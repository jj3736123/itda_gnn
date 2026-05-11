from pathlib import Path
import json
from datetime import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    raise ImportError("torch_geometric이 필요함. model2가 돌아갔다면 보통 이미 설치되어 있음.")


# ============================================================
# 05_train_rgcn_pca_original_edges.py
# 목적:
# - model3 PCA 128 node feature 사용
# - 기존 edge 3개만 사용
#   1. RUR
#   2. PRODUCT_TIME_7D
#   3. NEW_USER_BURST_7D
# - TEXT_SIMILARITY edge는 아직 사용하지 않음
# - model3_pca128_original_edges 기준 성능 측정
# ============================================================

NOTEBOOK_DIR = Path(__file__).resolve().parent
MODEL3_DIR = NOTEBOOK_DIR.parent

DATA_DIR = MODEL3_DIR / "data"
OUTPUT_DIR = MODEL3_DIR / "outputs"
REPORT_DIR = MODEL3_DIR / "report_assets"

TENSOR_DATA_PATH = DATA_DIR / "model3_tensor_data.pt"

BEST_MODEL_PATH = DATA_DIR / "model3_rgcn_pca_original_edges_best.pt"
RESULT_PATH = OUTPUT_DIR / "rgcn_pca_original_edges_result.json"
PREDICTION_PATH = REPORT_DIR / "model3_rgcn_pca_original_edges_all_predictions.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 1. 설정
# ------------------------------------------------------------

RANDOM_STATE = 42

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

set_seed(RANDOM_STATE)


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
# 5. 데이터 로드
# ------------------------------------------------------------

print("[LOAD] tensor data:", TENSOR_DATA_PATH)

try:
    tensor_data = torch.load(TENSOR_DATA_PATH, weights_only=False)
except TypeError:
    tensor_data = torch.load(TENSOR_DATA_PATH)

x = tensor_data["x"]
y = tensor_data["y"]

train_mask = tensor_data["train_mask"]
valid_mask = tensor_data["valid_mask"]
test_mask = tensor_data["test_mask"]

edge_index = tensor_data["base_edge_index"]
edge_type = tensor_data["base_edge_type"]
num_relations = tensor_data["base_num_relations"]

review_ids = tensor_data["review_ids"]

print("[INFO] x:", x.shape)
print("[INFO] y:", y.shape)
print("[INFO] edge_index:", edge_index.shape)
print("[INFO] edge_type:", edge_type.shape)
print("[INFO] num_relations:", num_relations)
print("[INFO] train:", int(train_mask.sum().item()))
print("[INFO] valid:", int(valid_mask.sum().item()))
print("[INFO] test:", int(test_mask.sum().item()))


# ------------------------------------------------------------
# 6. device 설정
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] device:", device)

x = x.to(device)
y = y.to(device)

train_mask = train_mask.to(device)
valid_mask = valid_mask.to(device)
test_mask = test_mask.to(device)

edge_index = edge_index.to(device)
edge_type = edge_type.to(device)


# ------------------------------------------------------------
# 7. pos_weight 계산
# ------------------------------------------------------------

train_y = y[train_mask].view(-1)
train_pos = int((train_y == 1).sum().item())
train_neg = int((train_y == 0).sum().item())

if train_pos == 0:
    pos_weight_value = 1.0
else:
    pos_weight_value = train_neg / train_pos

pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

print("[INFO] train negative:", train_neg)
print("[INFO] train positive:", train_pos)
print("[INFO] pos_weight:", pos_weight_value)


# ------------------------------------------------------------
# 8. 모델 / optimizer / loss
# ------------------------------------------------------------

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

print("[MODEL] RGCNBinaryClassifier")
print("[INFO] hidden_dim:", HIDDEN_DIM)
print("[INFO] dropout:", DROPOUT)
print("[INFO] learning_rate:", LEARNING_RATE)
print("[INFO] weight_decay:", WEIGHT_DECAY)


# ------------------------------------------------------------
# 9. 학습
# ------------------------------------------------------------

best_valid_pr_auc = -1.0
best_epoch = -1
patience_counter = 0

history = []

print("\n[START] training model3 RGCN PCA original edges")

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

    valid_y = y_cpu[valid_mask.detach().cpu().numpy()]
    valid_prob = prob_eval[valid_mask.detach().cpu().numpy()]

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
                "best_epoch": best_epoch,
                "best_valid_pr_auc": best_valid_pr_auc
            },
            BEST_MODEL_PATH
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
        print("[EARLY STOP] epoch:", epoch)
        break

print("[DONE] training finished")
print("[BEST] epoch:", best_epoch)
print("[BEST] valid PR-AUC:", best_valid_pr_auc)


# ------------------------------------------------------------
# 10. best model 로드 후 평가
# ------------------------------------------------------------

try:
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
except TypeError:
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)

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

best_threshold, valid_best_macro_f1 = find_best_threshold(valid_y, valid_prob)

valid_result = evaluate_with_threshold(valid_y, valid_prob, best_threshold)
test_result = evaluate_with_threshold(test_y, test_prob, best_threshold)

print("\n[VALID RESULT]")
print("PR-AUC:", round(valid_result["pr_auc"], 5))
print("Macro F1:", round(valid_result["macro_f1"], 5))
print("Accuracy:", round(valid_result["accuracy"], 5))
print("Threshold:", round(valid_result["threshold"], 2))
print("Confusion Matrix:", valid_result["confusion_matrix"])

print("\n[TEST RESULT]")
print("PR-AUC:", round(test_result["pr_auc"], 5))
print("Macro F1:", round(test_result["macro_f1"], 5))
print("Accuracy:", round(test_result["accuracy"], 5))
print("Precision:", round(test_result["precision"], 5))
print("Recall:", round(test_result["recall"], 5))
print("Threshold:", round(test_result["threshold"], 2))
print("Confusion Matrix:", test_result["confusion_matrix"])


# ------------------------------------------------------------
# 11. prediction 저장
# ------------------------------------------------------------

all_pred = (final_prob >= best_threshold).astype(int)

import pandas as pd

prediction_df = pd.DataFrame({
    "review_id": review_ids,
    "true_label": y_all.astype(int),
    "fake_probability": final_prob,
    "pred_label": all_pred
})

prediction_df.to_csv(PREDICTION_PATH, index=False, encoding="utf-8-sig")

print("[SAVED] predictions:", PREDICTION_PATH)


# ------------------------------------------------------------
# 12. result 저장
# ------------------------------------------------------------

result = {
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_name": "model3_rgcn_pca128_original_edges",
    "description": "RGCN with manual8 + SBERT PCA128 node features and original 3 edge relations.",
    "feature": {
        "manual_features": 8,
        "text_features": 128,
        "total_features": int(x.shape[1])
    },
    "edges": {
        "used_edges": [
            "RUR",
            "PRODUCT_TIME_7D",
            "NEW_USER_BURST_7D"
        ],
        "text_similarity_edge_used": False,
        "edge_count": int(edge_index.shape[1]),
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
        "selected_threshold_from_valid": float(best_threshold)
    },
    "valid_result": valid_result,
    "test_result": test_result,
    "outputs": {
        "best_model_path": str(BEST_MODEL_PATH),
        "prediction_path": str(PREDICTION_PATH)
    },
    "history": history
}

with open(RESULT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("[SAVED] result:", RESULT_PATH)

print("\n[DONE] model3 RGCN PCA original edges training completed.")
print("다음 단계: 06_train_rgcn_pca_text_similarity.py")