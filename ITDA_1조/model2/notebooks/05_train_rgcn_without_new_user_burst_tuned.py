import json
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


# =========================
# 0. Path setting
# =========================

CURRENT_FILE = Path(__file__).resolve()

MODEL2_DIR = CURRENT_FILE.parents[1]

DATA_DIR = MODEL2_DIR / "data"
OUTPUT_DIR = MODEL2_DIR / "outputs"
REPORT_ASSETS_DIR = MODEL2_DIR / "report_assets"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

PYG_DATA_PATH = DATA_DIR / "model2_pyg_data.pt"
TENSOR_DATA_PATH = DATA_DIR / "model2_tensor_data.pt"

BEST_MODEL_PATH = DATA_DIR / "model2_rgcn_without_new_user_burst_tuned_best.pt"
RESULT_OUTPUT_PATH = OUTPUT_DIR / "rgcn_without_new_user_burst_tuned_result.json"
PREDICTION_OUTPUT_PATH = REPORT_ASSETS_DIR / "final_rgcn_without_new_user_burst_tuned_all_predictions.csv"


# =========================
# 1. Config
# =========================

RANDOM_SEED = 42

HIDDEN_DIM = 32
DROPOUT = 0.5
LEARNING_RATE = 0.002
WEIGHT_DECAY = 5e-4

MAX_EPOCHS = 300
PATIENCE = 40

THRESHOLD_START = 0.05
THRESHOLD_END = 0.95
THRESHOLD_STEP = 0.01

# edge_type 2 = NEW_USER_BURST_7D 제거
REMOVE_EDGE_TYPE = 2


# =========================
# 2. Seed
# =========================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


# =========================
# 3. Load data
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

try:
    data = torch.load(PYG_DATA_PATH, map_location="cpu", weights_only=False)
    print("PyG data loaded:", PYG_DATA_PATH)
except Exception as e:
    print("PyG data load 실패. tensor_data로 대체 시도")
    print("error:", e)

    tensor_data = torch.load(TENSOR_DATA_PATH, map_location="cpu", weights_only=False)

    class SimpleData:
        pass

    data = SimpleData()
    data.x = tensor_data["x"]
    data.y = tensor_data["y"]
    data.edge_index = tensor_data["edge_index"]
    data.edge_type = tensor_data["edge_type"]
    data.train_mask = tensor_data["train_mask"]
    data.valid_mask = tensor_data["valid_mask"]
    data.test_mask = tensor_data["test_mask"]
    data.pos_weight = tensor_data["pos_weight"]

x = data.x.to(device)
y = data.y.to(device)

original_edge_index = data.edge_index
original_edge_type = data.edge_type

train_mask = data.train_mask.to(device)
valid_mask = data.valid_mask.to(device)
test_mask = data.test_mask.to(device)

pos_weight = data.pos_weight.to(device)


# =========================
# 4. Remove New User Burst edge
# =========================

keep_edge_mask = original_edge_type != REMOVE_EDGE_TYPE

edge_index_filtered = original_edge_index[:, keep_edge_mask]
edge_type_filtered = original_edge_type[keep_edge_mask]

# edge_type 0, 1만 남는 구조라서 그대로 사용 가능
# 혹시 모를 안전 체크
unique_edge_types = torch.unique(edge_type_filtered).tolist()

print("\nOriginal edge count:", original_edge_index.shape[1])
print("Filtered edge count:", edge_index_filtered.shape[1])
print("Removed edge count:", int((original_edge_type == REMOVE_EDGE_TYPE).sum().item()))
print("Remaining edge types:", unique_edge_types)

if REMOVE_EDGE_TYPE in unique_edge_types:
    raise ValueError("NEW_USER_BURST_7D edge가 제거되지 않음")

edge_index = edge_index_filtered.to(device)
edge_type = edge_type_filtered.to(device)

num_nodes = x.size(0)
input_dim = x.size(1)
num_relations = int(edge_type.max().item()) + 1

print("\nData check")
print("num_nodes:", num_nodes)
print("input_dim:", input_dim)
print("edge_index:", edge_index.shape)
print("edge_type:", edge_type.shape)
print("num_relations:", num_relations)
print("pos_weight:", pos_weight)

print("\nSplit count")
print("train:", int(train_mask.sum().item()))
print("valid:", int(valid_mask.sum().item()))
print("test:", int(test_mask.sum().item()))


# =========================
# 5. Model
# =========================

try:
    from torch_geometric.nn import RGCNConv
except ImportError as e:
    print("torch_geometric import 실패")
    print("PyTorch Geometric 설치가 필요함")
    raise e


class RGCNBinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_relations, dropout):
        super().__init__()

        self.conv1 = RGCNConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            num_relations=num_relations
        )

        self.conv2 = RGCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_relations=num_relations
        )

        self.lin = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        h = self.conv1(x, edge_index, edge_type)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_type)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.lin(h)

        return out


model = RGCNBinaryClassifier(
    input_dim=input_dim,
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

print("\nModel")
print(model)


# =========================
# 6. Metric functions
# =========================

def get_prob_and_label(logits, labels, mask):
    probs = torch.sigmoid(logits[mask]).detach().cpu().numpy().reshape(-1)
    true = labels[mask].detach().cpu().numpy().reshape(-1)

    return probs, true


def evaluate_with_threshold(probs, true, threshold):
    pred = (probs >= threshold).astype(int)

    acc = accuracy_score(true, pred)
    macro_f1 = f1_score(true, pred, average="macro", zero_division=0)
    precision = precision_score(true, pred, zero_division=0)
    recall = recall_score(true, pred, zero_division=0)

    try:
        pr_auc = average_precision_score(true, probs)
    except ValueError:
        pr_auc = 0.0

    cm = confusion_matrix(true, pred)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "precision": float(precision),
        "recall": float(recall),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist()
    }


def find_best_threshold(probs, true):
    thresholds = np.arange(
        THRESHOLD_START,
        THRESHOLD_END + 0.0001,
        THRESHOLD_STEP
    )

    best_threshold = 0.5
    best_macro_f1 = -1.0
    best_result = None

    for threshold in thresholds:
        result = evaluate_with_threshold(probs, true, threshold)

        if result["macro_f1"] > best_macro_f1:
            best_macro_f1 = result["macro_f1"]
            best_threshold = float(threshold)
            best_result = result

    return best_threshold, best_result


# =========================
# 7. Training
# =========================

best_valid_pr_auc = -1.0
best_epoch = -1
patience_counter = 0

history = []

print("\nTraining start")

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()

    optimizer.zero_grad()

    logits = model(x, edge_index, edge_type)
    loss = criterion(logits[train_mask], y[train_mask])

    loss.backward()
    optimizer.step()

    model.eval()

    with torch.no_grad():
        eval_logits = model(x, edge_index, edge_type)

        valid_probs, valid_true = get_prob_and_label(eval_logits, y, valid_mask)
        valid_pr_auc = average_precision_score(valid_true, valid_probs)

        valid_threshold, valid_result = find_best_threshold(valid_probs, valid_true)

    history.append({
        "epoch": epoch,
        "train_loss": float(loss.item()),
        "valid_pr_auc": float(valid_pr_auc),
        "valid_best_threshold": float(valid_threshold),
        "valid_macro_f1": float(valid_result["macro_f1"]),
        "valid_accuracy": float(valid_result["accuracy"])
    })

    if valid_pr_auc > best_valid_pr_auc:
        best_valid_pr_auc = valid_pr_auc
        best_epoch = epoch
        patience_counter = 0

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": HIDDEN_DIM,
            "num_relations": num_relations,
            "dropout": DROPOUT,
            "best_epoch": best_epoch,
            "best_valid_pr_auc": float(best_valid_pr_auc),
            "config": {
                "random_seed": RANDOM_SEED,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "removed_edge_type": REMOVE_EDGE_TYPE
            }
        }

        torch.save(checkpoint, BEST_MODEL_PATH)

    else:
        patience_counter += 1

    if epoch == 1 or epoch % 10 == 0:
        print(
            "Epoch",
            epoch,
            "| loss:",
            round(float(loss.item()), 4),
            "| valid PR-AUC:",
            round(float(valid_pr_auc), 4),
            "| valid Macro F1:",
            round(float(valid_result["macro_f1"]), 4),
            "| threshold:",
            round(float(valid_threshold), 2),
            "| patience:",
            patience_counter
        )

    if patience_counter >= PATIENCE:
        print("\nEarly stopping")
        print("best epoch:", best_epoch)
        print("best valid PR-AUC:", round(float(best_valid_pr_auc), 4))
        break


# =========================
# 8. Load best model
# =========================

print("\nBest model load:", BEST_MODEL_PATH)

checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    final_logits = model(x, edge_index, edge_type)

train_probs, train_true = get_prob_and_label(final_logits, y, train_mask)
valid_probs, valid_true = get_prob_and_label(final_logits, y, valid_mask)
test_probs, test_true = get_prob_and_label(final_logits, y, test_mask)

best_threshold, valid_best_result = find_best_threshold(valid_probs, valid_true)

train_result = evaluate_with_threshold(train_probs, train_true, best_threshold)
test_result = evaluate_with_threshold(test_probs, test_true, best_threshold)

print("\n==============================")
print("Final Result")
print("==============================")

print("best epoch:", checkpoint["best_epoch"])
print("best valid PR-AUC:", round(checkpoint["best_valid_pr_auc"], 4))
print("best threshold from valid:", round(best_threshold, 2))

print("\nTrain result:")
print(train_result)

print("\nValid result:")
print(valid_best_result)

print("\nTest result:")
print(test_result)


# =========================
# 9. Save predictions
# =========================

all_probs = torch.sigmoid(final_logits).detach().cpu().numpy().reshape(-1)
all_true = y.detach().cpu().numpy().reshape(-1)
all_pred = (all_probs >= best_threshold).astype(int)

split_array = np.empty(num_nodes, dtype=object)
split_array[train_mask.detach().cpu().numpy()] = "train"
split_array[valid_mask.detach().cpu().numpy()] = "valid"
split_array[test_mask.detach().cpu().numpy()] = "test"

prediction_df = pd.DataFrame({
    "node_id": np.arange(num_nodes),
    "label": all_true.astype(int),
    "prob_fake": all_probs,
    "pred": all_pred,
    "split": split_array
})

prediction_df.to_csv(PREDICTION_OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("\nprediction 저장 완료:")
print(PREDICTION_OUTPUT_PATH)


# =========================
# 10. Save result json
# =========================

result = {
    "model_name": "model2_rgcn_without_new_user_burst_tuned",
    "description": "Tuned RGCN using only RUR and PRODUCT_TIME_7D relations. NEW_USER_BURST_7D edge is removed.",
    "best_epoch": int(checkpoint["best_epoch"]),
    "best_valid_pr_auc": float(checkpoint["best_valid_pr_auc"]),
    "best_threshold_from_valid": float(best_threshold),
    "train_result": train_result,
    "valid_result": valid_best_result,
    "test_result": test_result,
    "config": {
        "random_seed": RANDOM_SEED,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "input_dim": input_dim,
        "num_relations": num_relations,
        "feature_type": "manual_8_plus_sbert_384",
        "uses_new_user_burst_edge": False,
        "removed_edge_type": REMOVE_EDGE_TYPE,
        "remaining_edge_types": unique_edge_types
    },
    "paths": {
        "best_model_path": str(BEST_MODEL_PATH),
        "prediction_output_path": str(PREDICTION_OUTPUT_PATH)
    },
    "history": history
}

with open(RESULT_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("\nresult 저장 완료:")
print(RESULT_OUTPUT_PATH)

print("\n==============================")
print("05_train_rgcn_without_new_user_burst_tuned.py 완료")
print("==============================")