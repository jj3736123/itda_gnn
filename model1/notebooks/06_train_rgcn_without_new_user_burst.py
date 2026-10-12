import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)


# ============================================================
# 0. 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "model1"
MODEL_DATA_DIR = MODEL_DIR / "data"
MODEL_OUTPUT_DIR = MODEL_DIR / "outputs"

TENSOR_DATA_PATH = MODEL_DATA_DIR / "model1_tensor_data.pt"
NODE_FEATURE_PATH = MODEL_DATA_DIR / "model1_node_features.csv"

BEST_MODEL_PATH = MODEL_DATA_DIR / "model1_rgcn_without_new_user_burst_best.pt"
TRAIN_LOG_PATH = MODEL_OUTPUT_DIR / "rgcn_without_new_user_burst_training_log.csv"
RESULT_PATH = MODEL_OUTPUT_DIR / "rgcn_without_new_user_burst_result.json"
PRED_PATH = MODEL_OUTPUT_DIR / "rgcn_without_new_user_burst_test_predictions.csv"

RANDOM_STATE = 42

HIDDEN_CHANNELS = 64
DROPOUT = 0.30
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 150
PATIENCE = 25

SAVE_BEST_BY = "valid_pr_auc"

# 제거할 edge type
# 0 = RUR
# 1 = PRODUCT_TIME_7D
# 2 = NEW_USER_BURST_7D
REMOVE_EDGE_TYPE = 2


# ============================================================
# 1. seed 고정
# ============================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_STATE)


# ============================================================
# 2. 데이터 불러오기
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 device:", device)

try:
    data = torch.load(TENSOR_DATA_PATH, map_location="cpu", weights_only=False)
except TypeError:
    data = torch.load(TENSOR_DATA_PATH, map_location="cpu")

x = data["x"].to(device)
y = data["y"].to(device)

edge_index = data["edge_index"]
edge_type = data["edge_type"]

# New User Burst edge 제거
keep_mask = edge_type != REMOVE_EDGE_TYPE
edge_index = edge_index[:, keep_mask]
edge_type = edge_type[keep_mask]

# edge type을 0, 1만 쓰게 유지
edge_index = edge_index.to(device)
edge_type = edge_type.to(device)

train_mask = data["train_mask"].to(device)
valid_mask = data["valid_mask"].to(device)
test_mask = data["test_mask"].to(device)

feature_cols = data["feature_cols"]
pos_weight = data["pos_weight"].to(device)

num_nodes = x.size(0)
in_channels = x.size(1)
num_relations = int(edge_type.max().item()) + 1

print("x:", x.shape)
print("y:", y.shape)
print("edge_index:", edge_index.shape)
print("edge_type:", edge_type.shape)
print("사용 relation 수:", num_relations)
print("제거한 edge type:", REMOVE_EDGE_TYPE)
print("pos_weight:", pos_weight.item())


# ============================================================
# 3. RGCN Layer
# ============================================================

class RelationalGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super().__init__()

        self.num_relations = num_relations
        self.out_channels = out_channels

        self.self_linear = nn.Linear(in_channels, out_channels)

        self.relation_linears = nn.ModuleList()
        for _ in range(num_relations):
            self.relation_linears.append(
                nn.Linear(in_channels, out_channels, bias=False)
            )

    def forward(self, x, edge_index, edge_type):
        src_all = edge_index[0]
        dst_all = edge_index[1]

        out = self.self_linear(x)

        for rel in range(self.num_relations):
            mask = edge_type == rel

            if mask.sum().item() == 0:
                continue

            src = src_all[mask]
            dst = dst_all[mask]

            message = self.relation_linears[rel](x[src])

            agg = torch.zeros(
                x.size(0),
                self.out_channels,
                device=x.device,
                dtype=message.dtype
            )

            agg.index_add_(0, dst, message)

            degree = torch.zeros(
                x.size(0),
                1,
                device=x.device,
                dtype=message.dtype
            )

            ones = torch.ones(
                dst.size(0),
                1,
                device=x.device,
                dtype=message.dtype
            )

            degree.index_add_(0, dst, ones)
            agg = agg / degree.clamp(min=1.0)

            out = out + agg

        return out


class SimpleRGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_relations, dropout):
        super().__init__()

        self.conv1 = RelationalGraphConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            num_relations=num_relations
        )

        self.conv2 = RelationalGraphConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_relations=num_relations
        )

        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.classifier(x).view(-1)
        return logits


# ============================================================
# 4. 평가 함수
# ============================================================

def find_best_threshold(y_true, prob):
    best_threshold = 0.5
    best_macro_f1 = -1.0

    thresholds = np.arange(0.05, 0.951, 0.01)

    for threshold in thresholds:
        pred = (prob >= threshold).astype(int)
        macro_f1 = f1_score(y_true, pred, average="macro", zero_division=0)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = float(threshold)

    return best_threshold, best_macro_f1


def evaluate_with_threshold(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist()
    }


@torch.no_grad()
def get_probs(model):
    model.eval()
    logits = model(x, edge_index, edge_type)
    probs = torch.sigmoid(logits)
    return probs.detach().cpu().numpy()


def get_split_arrays(probs, mask):
    mask_np = mask.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    return y_np[mask_np], probs[mask_np]


# ============================================================
# 5. 모델 준비
# ============================================================

model = SimpleRGCN(
    in_channels=in_channels,
    hidden_channels=HIDDEN_CHANNELS,
    num_relations=num_relations,
    dropout=DROPOUT
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

best_score = -1.0
best_epoch = 0
bad_count = 0
logs = []


# ============================================================
# 6. 학습
# ============================================================

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()

    optimizer.zero_grad()

    logits = model(x, edge_index, edge_type)

    loss = criterion(
        logits[train_mask],
        y[train_mask].float()
    )

    loss.backward()
    optimizer.step()

    probs = get_probs(model)

    valid_y, valid_prob = get_split_arrays(probs, valid_mask)

    valid_pr_auc = average_precision_score(valid_y, valid_prob)
    valid_best_threshold, valid_best_macro_f1 = find_best_threshold(valid_y, valid_prob)

    if SAVE_BEST_BY == "valid_macro_f1":
        current_score = valid_best_macro_f1
    else:
        current_score = valid_pr_auc

    log_row = {
        "epoch": epoch,
        "loss": float(loss.item()),
        "valid_pr_auc": float(valid_pr_auc),
        "valid_best_macro_f1": float(valid_best_macro_f1),
        "valid_best_threshold": float(valid_best_threshold)
    }

    logs.append(log_row)

    if current_score > best_score:
        best_score = current_score
        best_epoch = epoch
        bad_count = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "config": {
                "hidden_channels": HIDDEN_CHANNELS,
                "dropout": DROPOUT,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "save_best_by": SAVE_BEST_BY,
                "num_relations": num_relations,
                "in_channels": in_channels,
                "removed_edge_type": REMOVE_EDGE_TYPE
            }
        }, BEST_MODEL_PATH)

    else:
        bad_count += 1

    if epoch == 1 or epoch % 5 == 0:
        print(
            "Epoch",
            epoch,
            "| loss:", round(loss.item(), 4),
            "| valid PR-AUC:", round(valid_pr_auc, 4),
            "| valid best Macro F1:", round(valid_best_macro_f1, 4),
            "| threshold:", round(valid_best_threshold, 2)
        )

    if bad_count >= PATIENCE:
        print("\nEarly stopping 발생")
        print("best epoch:", best_epoch)
        break


# ============================================================
# 7. best model 로드 후 최종 평가
# ============================================================

checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])

probs = get_probs(model)

train_y, train_prob = get_split_arrays(probs, train_mask)
valid_y, valid_prob = get_split_arrays(probs, valid_mask)
test_y, test_prob = get_split_arrays(probs, test_mask)

best_threshold, valid_best_macro_f1 = find_best_threshold(valid_y, valid_prob)

train_result = evaluate_with_threshold(train_y, train_prob, best_threshold)
valid_result = evaluate_with_threshold(valid_y, valid_prob, best_threshold)
test_result = evaluate_with_threshold(test_y, test_prob, best_threshold)


# ============================================================
# 8. 결과 저장
# ============================================================

log_df = pd.DataFrame(logs)
log_df.to_csv(TRAIN_LOG_PATH, index=False, encoding="utf-8-sig")

result = {
    "model_name": "SimpleRGCN_without_New_User_Burst_edge",
    "removed_edge_type": int(REMOVE_EDGE_TYPE),
    "best_epoch": int(checkpoint["epoch"]),
    "save_best_by": SAVE_BEST_BY,
    "best_validation_threshold": float(best_threshold),
    "num_nodes": int(num_nodes),
    "num_features": int(in_channels),
    "num_edges": int(edge_index.size(1)),
    "num_relations": int(num_relations),
    "hyperparameters": {
        "hidden_channels": HIDDEN_CHANNELS,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "pos_weight": float(pos_weight.item())
    },
    "train_result": train_result,
    "valid_result": valid_result,
    "test_result": test_result
}

with open(RESULT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

df_node = pd.read_csv(NODE_FEATURE_PATH)
df_node = df_node.sort_values("node_id").reset_index(drop=True)

test_mask_np = test_mask.detach().cpu().numpy()
test_pred = (test_prob >= best_threshold).astype(int)

pred_df = df_node.loc[test_mask_np, ["node_id", "review_id", "user_id", "prod_id", "rating", "label", "text"]].copy()
pred_df["prob_fake"] = test_prob
pred_df["pred_label"] = test_pred
pred_df["correct"] = (pred_df["label"] == pred_df["pred_label"]).astype(int)

pred_df.to_csv(PRED_PATH, index=False, encoding="utf-8-sig")


# ============================================================
# 9. 최종 출력
# ============================================================

print("\n==============================")
print("RGCN without New User Burst 학습 완료")
print("==============================")
print("best epoch:", checkpoint["epoch"])
print("best validation threshold:", round(best_threshold, 4))

print("\n[Train]")
print("PR-AUC:", round(train_result["pr_auc"], 4))
print("Macro F1:", round(train_result["macro_f1"], 4))
print("Accuracy:", round(train_result["accuracy"], 4))
print("Confusion Matrix:", train_result["confusion_matrix"])

print("\n[Valid]")
print("PR-AUC:", round(valid_result["pr_auc"], 4))
print("Macro F1:", round(valid_result["macro_f1"], 4))
print("Accuracy:", round(valid_result["accuracy"], 4))
print("Confusion Matrix:", valid_result["confusion_matrix"])

print("\n[Test]")
print("PR-AUC:", round(test_result["pr_auc"], 4))
print("Macro F1:", round(test_result["macro_f1"], 4))
print("Accuracy:", round(test_result["accuracy"], 4))
print("Confusion Matrix:", test_result["confusion_matrix"])

print("\n저장 완료:")
print("best model:", BEST_MODEL_PATH)
print("training log:", TRAIN_LOG_PATH)
print("result json:", RESULT_PATH)
print("test predictions:", PRED_PATH)