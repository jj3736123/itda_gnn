from pathlib import Path
import json
from datetime import datetime
import random
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

try:
    from torch_geometric.nn import RGCNConv
except ImportError:
    raise ImportError("torch_geometric이 필요함. model2/model3가 돌아갔다면 보통 이미 설치되어 있음.")


# ============================================================
# 09_train_model3_final_grid_search.py
# 목적:
# - model3 안에서 진짜 최종 보완 실험 수행
# - TEXT_SIMILARITY edge의 SIM_THRESHOLD / TOP_K 조합을 grid search
# - valid PR-AUC 기준 최고 조합을 model3_final로 확정
#
# 후보:
# - SIM_THRESHOLD: 0.80 / 0.85 / 0.90
# - TOP_K: 3 / 5
#
# 최종 산출물:
# - model3_final_best.pt
# - model3_final_result.json
# - model3_final_grid_search_summary.csv
# - model3_final_all_predictions.csv
# - model3_final_edge_index.npy
# - model3_final_edge_type.npy
# - model3_final_text_similarity_edge_list.csv
# ============================================================


# ------------------------------------------------------------
# 0. 경로 설정
# ------------------------------------------------------------

NOTEBOOK_DIR = Path(__file__).resolve().parent
MODEL3_DIR = NOTEBOOK_DIR.parent

DATA_DIR = MODEL3_DIR / "data"
CONFIG_DIR = MODEL3_DIR / "configs"
OUTPUT_DIR = MODEL3_DIR / "outputs"
REPORT_DIR = MODEL3_DIR / "report_assets"

GRID_DIR = OUTPUT_DIR / "model3_final_grid_search"
GRID_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLED_REVIEWS_PATH = DATA_DIR / "model3_sampled_reviews.csv"
SOURCE_FEATURE_PATH = DATA_DIR / "model3_source_node_features_sbert384.csv"
SOURCE_FEATURE_CONFIG_PATH = CONFIG_DIR / "model3_source_feature_columns_sbert384.json"
TENSOR_DATA_PATH = DATA_DIR / "model3_tensor_data.pt"

FINAL_MODEL_PATH = DATA_DIR / "model3_final_best.pt"
FINAL_EDGE_INDEX_PATH = DATA_DIR / "model3_final_edge_index.npy"
FINAL_EDGE_TYPE_PATH = DATA_DIR / "model3_final_edge_type.npy"
FINAL_TEXT_EDGE_LIST_PATH = DATA_DIR / "model3_final_text_similarity_edge_list.csv"
FINAL_EDGE_CONFIG_PATH = CONFIG_DIR / "model3_final_edge_config.json"

FINAL_RESULT_PATH = OUTPUT_DIR / "model3_final_result.json"
GRID_SUMMARY_CSV_PATH = OUTPUT_DIR / "model3_final_grid_search_summary.csv"
GRID_SUMMARY_JSON_PATH = OUTPUT_DIR / "model3_final_grid_search_summary.json"
FINAL_PREDICTION_PATH = REPORT_DIR / "model3_final_all_predictions.csv"


# ------------------------------------------------------------
# 1. 실험 설정
# ------------------------------------------------------------

RANDOM_STATE = 42

SIM_THRESHOLD_LIST = [0.80, 0.85, 0.90]
TOP_K_LIST = [3, 5]
MIN_TEXT_LENGTH = 20

TEXT_EDGE_NAME = "TEXT_SIMILARITY"

HIDDEN_DIM = 32
DROPOUT = 0.5
LEARNING_RATE = 0.002
WEIGHT_DECAY = 5e-4
EPOCHS = 250
PATIENCE = 40

THRESHOLD_START = 0.05
THRESHOLD_END = 0.95
THRESHOLD_STEP = 0.01


# ------------------------------------------------------------
# 2. seed 고정
# ------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# 3. metric 함수
# ------------------------------------------------------------

def find_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_macro_f1 = -1.0

    thresholds = np.arange(
        THRESHOLD_START,
        THRESHOLD_END + THRESHOLD_STEP,
        THRESHOLD_STEP
    )

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = float(threshold)

    return best_threshold, best_macro_f1


def evaluate_with_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    result = {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

    return result


# ------------------------------------------------------------
# 4. 모델 정의
# ------------------------------------------------------------

class RGCNBinaryClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_relations, dropout):
        super().__init__()

        self.conv1 = RGCNConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            num_relations=num_relations
        )

        self.conv2 = RGCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_relations=num_relations
        )

        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_type):
        h = self.conv1(x, edge_index, edge_type)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_type)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.classifier(h)
        return out


# ------------------------------------------------------------
# 5. SBERT 컬럼 찾기
# ------------------------------------------------------------

def find_sbert_columns(feature_df, config_path):
    sbert_columns = []

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            source_config = json.load(f)

        if "sbert_feature_columns" in source_config:
            for col in source_config["sbert_feature_columns"]:
                if col in feature_df.columns:
                    sbert_columns.append(col)

    if len(sbert_columns) == 0:
        prefix_candidates = [
            "sbert_",
            "sbert",
            "text_sbert_",
            "text_emb_",
            "emb_",
            "embedding_"
        ]

        for col in feature_df.columns:
            lower_col = col.lower()
            for prefix in prefix_candidates:
                if lower_col.startswith(prefix):
                    sbert_columns.append(col)
                    break

    if len(sbert_columns) == 0:
        raise ValueError("SBERT feature column을 찾지 못함.")

    return sbert_columns


# ------------------------------------------------------------
# 6. TEXT_SIMILARITY edge 생성 함수
# ------------------------------------------------------------

def build_text_similarity_edges(work_df, sbert_columns, sim_threshold, top_k, min_text_length):
    text_edge_pair_dict = {}
    product_summaries = []

    total_products = work_df["prod_id"].nunique()
    processed_product_count = 0

    print("")
    print("[BUILD EDGE] SIM_THRESHOLD:", sim_threshold, "| TOP_K:", top_k)

    for prod_id, group in work_df.groupby("prod_id"):
        processed_product_count += 1

        if processed_product_count % 300 == 0:
            print("[EDGE PROGRESS]", processed_product_count, "/", total_products)

        group = group[group["text_length_for_edge"] >= min_text_length].copy()
        group = group.reset_index(drop=True)

        n = len(group)

        if n <= 1:
            product_summaries.append({
                "prod_id": int(prod_id),
                "candidate_reviews": int(n),
                "created_directed_edges": 0
            })
            continue

        X = group[sbert_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        sim_matrix = cosine_similarity(X)

        user_values = group["user_id"].values
        node_values = group["node_idx"].values
        review_values = group["review_id"].values

        created_for_product = 0

        for i in range(n):
            similarities = sim_matrix[i].copy()

            similarities[i] = -1.0

            same_user_mask = user_values == user_values[i]
            similarities[same_user_mask] = -1.0

            valid_indices = np.where(similarities >= sim_threshold)[0]

            if len(valid_indices) == 0:
                continue

            valid_scores = similarities[valid_indices]
            sorted_order = np.argsort(-valid_scores)
            sorted_indices = valid_indices[sorted_order]
            selected_indices = sorted_indices[:top_k]

            src_node = int(node_values[i])
            src_review_id = review_values[i]

            for j in selected_indices:
                dst_node = int(node_values[j])
                dst_review_id = review_values[j]
                score = float(similarities[j])

                if src_node == dst_node:
                    continue

                key1 = (src_node, dst_node)
                key2 = (dst_node, src_node)

                if key1 not in text_edge_pair_dict:
                    text_edge_pair_dict[key1] = {
                        "src": src_node,
                        "dst": dst_node,
                        "src_review_id": src_review_id,
                        "dst_review_id": dst_review_id,
                        "src_user_id": int(user_values[i]),
                        "dst_user_id": int(user_values[j]),
                        "prod_id": int(prod_id),
                        "similarity": score,
                        "edge_name": TEXT_EDGE_NAME
                    }
                    created_for_product += 1

                if key2 not in text_edge_pair_dict:
                    text_edge_pair_dict[key2] = {
                        "src": dst_node,
                        "dst": src_node,
                        "src_review_id": dst_review_id,
                        "dst_review_id": src_review_id,
                        "src_user_id": int(user_values[j]),
                        "dst_user_id": int(user_values[i]),
                        "prod_id": int(prod_id),
                        "similarity": score,
                        "edge_name": TEXT_EDGE_NAME
                    }
                    created_for_product += 1

        product_summaries.append({
            "prod_id": int(prod_id),
            "candidate_reviews": int(n),
            "created_directed_edges": int(created_for_product)
        })

    text_edge_records = list(text_edge_pair_dict.values())

    if len(text_edge_records) == 0:
        text_edge_df = pd.DataFrame(columns=[
            "src", "dst", "src_review_id", "dst_review_id",
            "src_user_id", "dst_user_id", "prod_id",
            "similarity", "edge_name"
        ])
    else:
        text_edge_df = pd.DataFrame(text_edge_records)

    product_summary_df = pd.DataFrame(product_summaries)

    products_with_edges = 0
    max_edges_product = None

    if len(product_summary_df) > 0:
        products_with_edges = int((product_summary_df["created_directed_edges"] > 0).sum())

        if product_summary_df["created_directed_edges"].max() > 0:
            max_row = product_summary_df.sort_values(
                "created_directed_edges",
                ascending=False
            ).iloc[0]

            max_edges_product = {
                "prod_id": int(max_row["prod_id"]),
                "candidate_reviews": int(max_row["candidate_reviews"]),
                "created_directed_edges": int(max_row["created_directed_edges"])
            }

    edge_info = {
        "text_edge_count": int(len(text_edge_df)),
        "products_with_text_edges": int(products_with_edges),
        "max_edges_product": max_edges_product
    }

    print("[EDGE DONE] text edges:", len(text_edge_df))
    print("[EDGE DONE] products with text edges:", products_with_edges)

    return text_edge_df, edge_info


# ------------------------------------------------------------
# 7. 학습 함수
# ------------------------------------------------------------

def train_one_candidate(
    candidate_name,
    x,
    y,
    train_mask,
    valid_mask,
    test_mask,
    edge_index,
    edge_type,
    num_relations,
    review_ids,
    device,
    sim_threshold,
    top_k,
    text_edge_count,
    products_with_text_edges
):
    set_seed(RANDOM_STATE)

    x = x.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    valid_mask = valid_mask.to(device)
    test_mask = test_mask.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)

    train_y = y[train_mask].view(-1)
    train_pos = int((train_y == 1).sum().item())
    train_neg = int((train_y == 0).sum().item())

    if train_pos == 0:
        pos_weight_value = 1.0
    else:
        pos_weight_value = train_neg / train_pos

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    model = RGCNBinaryClassifier(
        in_dim=x.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_relations=num_relations,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    candidate_model_path = GRID_DIR / (candidate_name + "_best.pt")
    candidate_result_path = GRID_DIR / (candidate_name + "_result.json")

    best_valid_pr_auc = -1.0
    best_epoch = -1
    patience_counter = 0
    history = []

    print("")
    print("[TRAIN START]", candidate_name)
    print("[INFO] edge_count:", int(edge_index.shape[1]))
    print("[INFO] num_relations:", int(num_relations))
    print("[INFO] text_edge_count:", int(text_edge_count))

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(x, edge_index, edge_type)
        loss = criterion(logits[train_mask], y[train_mask])

        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            logits_eval = model(x, edge_index, edge_type)
            prob_eval = torch.sigmoid(logits_eval).detach().cpu().numpy().reshape(-1)

        y_cpu = y.detach().cpu().numpy().reshape(-1)

        valid_mask_np = valid_mask.detach().cpu().numpy()
        valid_y = y_cpu[valid_mask_np]
        valid_prob = prob_eval[valid_mask_np]

        valid_pr_auc = average_precision_score(valid_y, valid_prob)
        valid_threshold, valid_macro_f1 = find_best_threshold(valid_y, valid_prob)

        record = {
            "epoch": int(epoch),
            "train_loss": float(loss.item()),
            "valid_pr_auc": float(valid_pr_auc),
            "valid_macro_f1_at_best_threshold": float(valid_macro_f1),
            "valid_best_threshold": float(valid_threshold)
        }

        history.append(record)

        if valid_pr_auc > best_valid_pr_auc:
            best_valid_pr_auc = float(valid_pr_auc)
            best_epoch = int(epoch)
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "in_dim": int(x.shape[1]),
                        "hidden_dim": HIDDEN_DIM,
                        "num_relations": int(num_relations),
                        "dropout": DROPOUT
                    },
                    "train_config": {
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "pos_weight": float(pos_weight_value),
                        "random_state": RANDOM_STATE
                    },
                    "edge_config": {
                        "sim_threshold": float(sim_threshold),
                        "top_k": int(top_k),
                        "min_text_length": int(MIN_TEXT_LENGTH),
                        "text_edge_count": int(text_edge_count),
                        "products_with_text_edges": int(products_with_text_edges)
                    },
                    "best_epoch": best_epoch,
                    "best_valid_pr_auc": best_valid_pr_auc
                },
                candidate_model_path
            )
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                "Epoch",
                epoch,
                "| loss:",
                round(float(loss.item()), 5),
                "| valid PR-AUC:",
                round(float(valid_pr_auc), 5),
                "| valid macro F1:",
                round(float(valid_macro_f1), 5),
                "| threshold:",
                round(float(valid_threshold), 2)
            )

        if patience_counter >= PATIENCE:
            print("[EARLY STOP]", candidate_name, "| epoch:", epoch)
            break

    print("[TRAIN DONE]", candidate_name)
    print("[BEST] epoch:", best_epoch)
    print("[BEST] valid PR-AUC:", best_valid_pr_auc)

    try:
        checkpoint = torch.load(candidate_model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(candidate_model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        final_logits = model(x, edge_index, edge_type)
        final_prob = torch.sigmoid(final_logits).detach().cpu().numpy().reshape(-1)

    y_all = y.detach().cpu().numpy().reshape(-1)

    train_mask_np = train_mask.detach().cpu().numpy()
    valid_mask_np = valid_mask.detach().cpu().numpy()
    test_mask_np = test_mask.detach().cpu().numpy()

    valid_y = y_all[valid_mask_np]
    valid_prob = final_prob[valid_mask_np]

    test_y = y_all[test_mask_np]
    test_prob = final_prob[test_mask_np]

    selected_threshold, valid_best_macro_f1 = find_best_threshold(valid_y, valid_prob)

    valid_result = evaluate_with_threshold(valid_y, valid_prob, selected_threshold)
    test_result = evaluate_with_threshold(test_y, test_prob, selected_threshold)

    result = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candidate_name": candidate_name,
        "model_name": "model3_final_candidate",
        "sim_threshold": float(sim_threshold),
        "top_k": int(top_k),
        "min_text_length": int(MIN_TEXT_LENGTH),
        "feature": {
            "manual_features": 8,
            "text_features": 128,
            "total_features": int(x.shape[1])
        },
        "edges": {
            "used_edges": [
                "RUR",
                "PRODUCT_TIME_7D",
                "NEW_USER_BURST_7D",
                "TEXT_SIMILARITY"
            ],
            "edge_count": int(edge_index.shape[1]),
            "base_edge_count": int(edge_index.shape[1] - text_edge_count),
            "text_similarity_edge_count": int(text_edge_count),
            "products_with_text_edges": int(products_with_text_edges),
            "num_relations": int(num_relations)
        },
        "train_config": {
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs_limit": EPOCHS,
            "patience": PATIENCE,
            "pos_weight": float(pos_weight_value),
            "random_state": RANDOM_STATE
        },
        "best": {
            "best_epoch": int(best_epoch),
            "best_valid_pr_auc": float(best_valid_pr_auc),
            "selected_threshold_from_valid": float(selected_threshold)
        },
        "valid_result": valid_result,
        "test_result": test_result,
        "model_path": str(candidate_model_path),
        "result_path": str(candidate_result_path),
        "history": history
    }

    with open(candidate_result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("[RESULT]", candidate_name)
    print("Valid PR-AUC:", round(valid_result["pr_auc"], 5))
    print("Test PR-AUC:", round(test_result["pr_auc"], 5))
    print("Test Macro F1:", round(test_result["macro_f1"], 5))
    print("Threshold:", round(test_result["threshold"], 2))
    print("Confusion Matrix:", test_result["confusion_matrix"])

    return result, final_prob, candidate_model_path


# ------------------------------------------------------------
# 8. 데이터 로드
# ------------------------------------------------------------

print("[LOAD] sampled reviews:", SAMPLED_REVIEWS_PATH)
sample_df = pd.read_csv(SAMPLED_REVIEWS_PATH)

print("[LOAD] source SBERT feature:", SOURCE_FEATURE_PATH)
feature_df = pd.read_csv(SOURCE_FEATURE_PATH)

print("[LOAD] tensor data:", TENSOR_DATA_PATH)
try:
    tensor_data = torch.load(TENSOR_DATA_PATH, weights_only=False)
except TypeError:
    tensor_data = torch.load(TENSOR_DATA_PATH)

print("[INFO] sample shape:", sample_df.shape)
print("[INFO] feature shape:", feature_df.shape)

required_cols = ["review_id", "user_id", "prod_id", "text"]

for col in required_cols:
    if col not in sample_df.columns:
        raise ValueError("sample_df에 필요한 컬럼이 없음: " + col)

sample_df = sample_df.reset_index(drop=True)
sample_df["node_idx"] = np.arange(len(sample_df))
sample_df["text"] = sample_df["text"].fillna("").astype(str)
sample_df["text_length_for_edge"] = sample_df["text"].str.len()

sbert_columns = find_sbert_columns(feature_df, SOURCE_FEATURE_CONFIG_PATH)
print("[OK] SBERT columns:", len(sbert_columns))

merge_cols = ["review_id"] + sbert_columns

work_df = sample_df[
    ["review_id", "node_idx", "user_id", "prod_id", "text_length_for_edge"]
].merge(
    feature_df[merge_cols],
    on="review_id",
    how="left"
)

if work_df[sbert_columns[0]].isna().sum() > 0:
    raise ValueError("SBERT feature가 없는 review가 있음.")

print("[OK] work df shape:", work_df.shape)


# ------------------------------------------------------------
# 9. tensor data 준비
# ------------------------------------------------------------

x = tensor_data["x"]
y = tensor_data["y"]

train_mask = tensor_data["train_mask"]
valid_mask = tensor_data["valid_mask"]
test_mask = tensor_data["test_mask"]

base_edge_index = tensor_data["base_edge_index"]
base_edge_type = tensor_data["base_edge_type"]
base_num_relations = tensor_data["base_num_relations"]

review_ids = tensor_data["review_ids"]

if base_edge_index.shape[0] != 2:
    base_edge_index = base_edge_index.T

base_edge_type = base_edge_type.reshape(-1)

text_edge_type_id = int(base_num_relations)

print("[INFO] x:", x.shape)
print("[INFO] y:", y.shape)
print("[INFO] base edge index:", base_edge_index.shape)
print("[INFO] base edge type:", base_edge_type.shape)
print("[INFO] base num relations:", base_num_relations)
print("[INFO] text edge type id:", text_edge_type_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] device:", device)


# ------------------------------------------------------------
# 10. Grid Search 실행
# ------------------------------------------------------------

all_results = []
candidate_prob_store = {}
candidate_model_path_store = {}
candidate_edge_store = {}

for sim_threshold in SIM_THRESHOLD_LIST:
    for top_k in TOP_K_LIST:
        candidate_name = "model3_final_thr" + str(sim_threshold).replace(".", "") + "_topk" + str(top_k)

        text_edge_df, edge_info = build_text_similarity_edges(
            work_df=work_df,
            sbert_columns=sbert_columns,
            sim_threshold=sim_threshold,
            top_k=top_k,
            min_text_length=MIN_TEXT_LENGTH
        )

        if len(text_edge_df) > 0:
            text_edge_index_np = text_edge_df[["src", "dst"]].values.T.astype(np.int64)
            text_edge_type_np = np.full(
                shape=(text_edge_index_np.shape[1],),
                fill_value=text_edge_type_id,
                dtype=np.int64
            )
        else:
            text_edge_index_np = np.empty((2, 0), dtype=np.int64)
            text_edge_type_np = np.empty((0,), dtype=np.int64)

        text_edge_index = torch.tensor(text_edge_index_np, dtype=torch.long)
        text_edge_type = torch.tensor(text_edge_type_np, dtype=torch.long)

        combined_edge_index = torch.cat(
            [base_edge_index.long(), text_edge_index.long()],
            dim=1
        )

        combined_edge_type = torch.cat(
            [base_edge_type.long(), text_edge_type.long()],
            dim=0
        )

        num_relations = int(text_edge_type_id + 1)

        result, final_prob, candidate_model_path = train_one_candidate(
            candidate_name=candidate_name,
            x=x,
            y=y,
            train_mask=train_mask,
            valid_mask=valid_mask,
            test_mask=test_mask,
            edge_index=combined_edge_index,
            edge_type=combined_edge_type,
            num_relations=num_relations,
            review_ids=review_ids,
            device=device,
            sim_threshold=sim_threshold,
            top_k=top_k,
            text_edge_count=edge_info["text_edge_count"],
            products_with_text_edges=edge_info["products_with_text_edges"]
        )

        all_results.append(result)
        candidate_prob_store[candidate_name] = final_prob
        candidate_model_path_store[candidate_name] = candidate_model_path
        candidate_edge_store[candidate_name] = {
            "text_edge_df": text_edge_df,
            "combined_edge_index": combined_edge_index.detach().cpu().numpy(),
            "combined_edge_type": combined_edge_type.detach().cpu().numpy(),
            "edge_info": edge_info,
            "sim_threshold": sim_threshold,
            "top_k": top_k
        }


# ------------------------------------------------------------
# 11. 결과 요약 및 best 선택
# ------------------------------------------------------------

summary_rows = []

for result in all_results:
    row = {
        "candidate_name": result["candidate_name"],
        "sim_threshold": result["sim_threshold"],
        "top_k": result["top_k"],
        "text_edge_count": result["edges"]["text_similarity_edge_count"],
        "products_with_text_edges": result["edges"]["products_with_text_edges"],
        "edge_count": result["edges"]["edge_count"],
        "best_epoch": result["best"]["best_epoch"],
        "valid_pr_auc": result["valid_result"]["pr_auc"],
        "valid_macro_f1": result["valid_result"]["macro_f1"],
        "test_pr_auc": result["test_result"]["pr_auc"],
        "test_macro_f1": result["test_result"]["macro_f1"],
        "test_accuracy": result["test_result"]["accuracy"],
        "test_precision": result["test_result"]["precision"],
        "test_recall": result["test_result"]["recall"],
        "threshold": result["test_result"]["threshold"],
        "confusion_matrix": str(result["test_result"]["confusion_matrix"]),
        "model_path": result["model_path"]
    }

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

summary_df = summary_df.sort_values(
    by=["valid_pr_auc", "valid_macro_f1"],
    ascending=[False, False]
).reset_index(drop=True)

summary_df.insert(0, "rank_by_valid", np.arange(1, len(summary_df) + 1))

summary_df.to_csv(GRID_SUMMARY_CSV_PATH, index=False, encoding="utf-8-sig")

print("")
print("[GRID SEARCH SUMMARY]")
print(summary_df)

best_row = summary_df.iloc[0]
best_candidate_name = best_row["candidate_name"]

print("")
print("[FINAL SELECTED BY VALID PR-AUC]")
print("best candidate:", best_candidate_name)
print("sim_threshold:", best_row["sim_threshold"])
print("top_k:", best_row["top_k"])
print("valid PR-AUC:", best_row["valid_pr_auc"])
print("test PR-AUC:", best_row["test_pr_auc"])
print("test Macro F1:", best_row["test_macro_f1"])


# ------------------------------------------------------------
# 12. 최종 model3_final 저장
# ------------------------------------------------------------

best_model_path = candidate_model_path_store[best_candidate_name]
shutil.copy2(best_model_path, FINAL_MODEL_PATH)

best_edge_item = candidate_edge_store[best_candidate_name]
best_prob = candidate_prob_store[best_candidate_name]

np.save(FINAL_EDGE_INDEX_PATH, best_edge_item["combined_edge_index"])
np.save(FINAL_EDGE_TYPE_PATH, best_edge_item["combined_edge_type"])
best_edge_item["text_edge_df"].to_csv(FINAL_TEXT_EDGE_LIST_PATH, index=False, encoding="utf-8-sig")

print("[SAVED] final model:", FINAL_MODEL_PATH)
print("[SAVED] final edge index:", FINAL_EDGE_INDEX_PATH)
print("[SAVED] final edge type:", FINAL_EDGE_TYPE_PATH)
print("[SAVED] final text edge list:", FINAL_TEXT_EDGE_LIST_PATH)


# ------------------------------------------------------------
# 13. 최종 prediction 저장
# ------------------------------------------------------------

y_all = y.detach().cpu().numpy().reshape(-1)
selected_threshold = float(best_row["threshold"])
final_pred = (best_prob >= selected_threshold).astype(int)

prediction_df = pd.DataFrame({
    "review_id": review_ids,
    "true_label": y_all.astype(int),
    "fake_probability": best_prob,
    "pred_label": final_pred
})

prediction_df.to_csv(FINAL_PREDICTION_PATH, index=False, encoding="utf-8-sig")

print("[SAVED] final prediction:", FINAL_PREDICTION_PATH)


# ------------------------------------------------------------
# 14. 최종 result json 저장
# ------------------------------------------------------------

best_result = None

for result in all_results:
    if result["candidate_name"] == best_candidate_name:
        best_result = result
        break

if best_result is None:
    raise ValueError("best_result를 찾지 못함.")

final_result = {
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_name": "model3_final",
    "internal_experiment_name": best_candidate_name,
    "selection_rule": "selected by highest validation PR-AUC, then validation Macro F1",
    "description": "Final RGCN model selected from TEXT_SIMILARITY edge grid search.",
    "grid_search_space": {
        "sim_threshold_list": SIM_THRESHOLD_LIST,
        "top_k_list": TOP_K_LIST,
        "min_text_length": MIN_TEXT_LENGTH
    },
    "selected_config": {
        "sim_threshold": float(best_row["sim_threshold"]),
        "top_k": int(best_row["top_k"]),
        "min_text_length": int(MIN_TEXT_LENGTH),
        "text_edge_count": int(best_row["text_edge_count"]),
        "products_with_text_edges": int(best_row["products_with_text_edges"])
    },
    "feature": {
        "manual_features": 8,
        "text_features": 128,
        "total_features": int(x.shape[1]),
        "text_feature_type": "SBERT PCA 128"
    },
    "edges": {
        "used_edges": [
            "RUR",
            "PRODUCT_TIME_7D",
            "NEW_USER_BURST_7D",
            "TEXT_SIMILARITY"
        ],
        "edge_count": int(best_row["edge_count"]),
        "num_relations": int(text_edge_type_id + 1)
    },
    "train_config": {
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs_limit": EPOCHS,
        "patience": PATIENCE,
        "random_state": RANDOM_STATE
    },
    "best": best_result["best"],
    "valid_result": best_result["valid_result"],
    "test_result": best_result["test_result"],
    "outputs": {
        "final_model_path": str(FINAL_MODEL_PATH),
        "final_prediction_path": str(FINAL_PREDICTION_PATH),
        "final_edge_index_path": str(FINAL_EDGE_INDEX_PATH),
        "final_edge_type_path": str(FINAL_EDGE_TYPE_PATH),
        "final_text_edge_list_path": str(FINAL_TEXT_EDGE_LIST_PATH),
        "grid_summary_csv_path": str(GRID_SUMMARY_CSV_PATH)
    },
    "all_candidates": summary_df.to_dict(orient="records")
}

with open(FINAL_RESULT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_result, f, ensure_ascii=False, indent=2)

with open(GRID_SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "selection_rule": "highest validation PR-AUC",
            "best_candidate": best_candidate_name,
            "summary": summary_df.to_dict(orient="records")
        },
        f,
        ensure_ascii=False,
        indent=2
    )

edge_config = {
    "model": "model3_final",
    "internal_experiment_name": best_candidate_name,
    "selected_by": "validation PR-AUC",
    "text_similarity_edge": {
        "edge_name": TEXT_EDGE_NAME,
        "same_prod_id": True,
        "different_user_id": True,
        "min_text_length": int(MIN_TEXT_LENGTH),
        "cosine_similarity_threshold": float(best_row["sim_threshold"]),
        "top_k_per_review": int(best_row["top_k"]),
        "bidirectional_directed_edges": True,
        "label_used": False,
        "prediction_used": False,
        "directed_edge_count": int(best_row["text_edge_count"]),
        "products_with_text_edges": int(best_row["products_with_text_edges"])
    },
    "edge_counts": {
        "combined_edge_count": int(best_row["edge_count"]),
        "num_relations": int(text_edge_type_id + 1)
    }
}

with open(FINAL_EDGE_CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(edge_config, f, ensure_ascii=False, indent=2)

print("[SAVED] final result:", FINAL_RESULT_PATH)
print("[SAVED] grid summary csv:", GRID_SUMMARY_CSV_PATH)
print("[SAVED] grid summary json:", GRID_SUMMARY_JSON_PATH)
print("[SAVED] final edge config:", FINAL_EDGE_CONFIG_PATH)

print("")
print("[DONE] model3_final grid search completed.")
print("최종 모델은 valid PR-AUC 기준으로 선택된 model3_final이다.")