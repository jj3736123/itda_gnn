import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import torch


# =========================
# 0. Path setting
# =========================

CURRENT_FILE = Path(__file__).resolve()

# ITDA/model2/notebooks/03_prepare_data.py 기준
MODEL2_DIR = CURRENT_FILE.parents[1]

DATA_DIR = MODEL2_DIR / "data"
CONFIG_DIR = MODEL2_DIR / "configs"
OUTPUT_DIR = MODEL2_DIR / "outputs"

NODE_FEATURE_PATH = DATA_DIR / "model2_node_features.csv"
SPLIT_PATH = DATA_DIR / "model2_split.csv"
EDGE_INDEX_PATH = DATA_DIR / "model2_edge_index.npy"
EDGE_TYPE_PATH = DATA_DIR / "model2_edge_type.npy"
FEATURE_COLUMNS_PATH = CONFIG_DIR / "model2_feature_columns.json"

SCALER_OUTPUT_PATH = DATA_DIR / "model2_standard_scaler.pkl"
TENSOR_DATA_OUTPUT_PATH = DATA_DIR / "model2_tensor_data.pt"
PYG_DATA_OUTPUT_PATH = DATA_DIR / "model2_pyg_data.pt"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "prepare_data_summary.json"


# =========================
# 1. Load files
# =========================

print("node feature path:", NODE_FEATURE_PATH)
print("split path:", SPLIT_PATH)
print("edge index path:", EDGE_INDEX_PATH)
print("edge type path:", EDGE_TYPE_PATH)
print("feature columns path:", FEATURE_COLUMNS_PATH)

node_df = pd.read_csv(NODE_FEATURE_PATH)
split_df = pd.read_csv(SPLIT_PATH)

edge_index = np.load(EDGE_INDEX_PATH)
edge_type = np.load(EDGE_TYPE_PATH)

with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
    feature_config = json.load(f)

feature_columns = feature_config["final_feature_columns"]

print("\nnode_df shape:", node_df.shape)
print("split_df shape:", split_df.shape)
print("edge_index shape:", edge_index.shape)
print("edge_type shape:", edge_type.shape)
print("feature column 수:", len(feature_columns))


# =========================
# 2. Basic checks
# =========================

required_node_cols = ["node_id", "review_id", "user_id", "prod_id", "label"]
required_split_cols = ["node_id", "review_id", "user_id", "prod_id", "label", "split"]

missing_node_cols = []
for col in required_node_cols:
    if col not in node_df.columns:
        missing_node_cols.append(col)

missing_split_cols = []
for col in required_split_cols:
    if col not in split_df.columns:
        missing_split_cols.append(col)

if len(missing_node_cols) > 0:
    raise ValueError("node_df에 필요한 컬럼이 없음: " + str(missing_node_cols))

if len(missing_split_cols) > 0:
    raise ValueError("split_df에 필요한 컬럼이 없음: " + str(missing_split_cols))

missing_feature_cols = []
for col in feature_columns:
    if col not in node_df.columns:
        missing_feature_cols.append(col)

if len(missing_feature_cols) > 0:
    raise ValueError("node_df에 feature column이 없음: " + str(missing_feature_cols[:10]))

if node_df["node_id"].nunique() != len(node_df):
    raise ValueError("node_df의 node_id가 unique하지 않음")

if split_df["node_id"].nunique() != len(split_df):
    raise ValueError("split_df의 node_id가 unique하지 않음")


# =========================
# 3. Sort by node_id
# =========================

node_df = node_df.sort_values("node_id").reset_index(drop=True)
split_df = split_df.sort_values("node_id").reset_index(drop=True)

node_ids = node_df["node_id"].to_numpy()
expected_node_ids = np.arange(len(node_df))

if not np.array_equal(node_ids, expected_node_ids):
    raise ValueError("node_id가 0부터 N-1까지 연속이 아님")

if not np.array_equal(node_df["node_id"].to_numpy(), split_df["node_id"].to_numpy()):
    raise ValueError("node_df와 split_df의 node_id 순서가 다름")

if not np.array_equal(node_df["label"].to_numpy(), split_df["label"].to_numpy()):
    raise ValueError("node_df와 split_df의 label이 다름")

print("\nnode_id / split 정렬 확인 완료")


# =========================
# 4. Feature / label / split
# =========================

X = node_df[feature_columns].to_numpy(dtype=np.float32)
y = node_df["label"].to_numpy(dtype=np.float32)

split_values = split_df["split"].to_numpy()

train_mask_np = split_values == "train"
valid_mask_np = split_values == "valid"
test_mask_np = split_values == "test"

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nsplit counts:")
print("train:", int(train_mask_np.sum()))
print("valid:", int(valid_mask_np.sum()))
print("test:", int(test_mask_np.sum()))

if train_mask_np.sum() == 0:
    raise ValueError("train split이 비어 있음")

if valid_mask_np.sum() == 0:
    raise ValueError("valid split이 비어 있음")

if test_mask_np.sum() == 0:
    raise ValueError("test split이 비어 있음")


# =========================
# 5. Standard scaling
#
# train에만 fit
# valid/test는 transform만 적용
# =========================

scaler = StandardScaler()

X_scaled = np.zeros_like(X, dtype=np.float32)

X_train = X[train_mask_np]
scaler.fit(X_train)

X_scaled[train_mask_np] = scaler.transform(X[train_mask_np]).astype(np.float32)
X_scaled[valid_mask_np] = scaler.transform(X[valid_mask_np]).astype(np.float32)
X_scaled[test_mask_np] = scaler.transform(X[test_mask_np]).astype(np.float32)

with open(SCALER_OUTPUT_PATH, "wb") as f:
    pickle.dump(scaler, f)

print("\nStandardScaler 저장 완료:")
print(SCALER_OUTPUT_PATH)

print("\nscaled feature check:")
print("X_scaled shape:", X_scaled.shape)
print("nan count:", int(np.isnan(X_scaled).sum()))
print("inf count:", int(np.isinf(X_scaled).sum()))

if np.isnan(X_scaled).sum() > 0:
    raise ValueError("X_scaled에 NaN 있음")

if np.isinf(X_scaled).sum() > 0:
    raise ValueError("X_scaled에 inf 있음")


# =========================
# 6. pos_weight
# =========================

y_train = y[train_mask_np]

train_pos = int((y_train == 1).sum())
train_neg = int((y_train == 0).sum())

pos_weight_value = train_neg / train_pos

print("\ntrain label count:")
print("negative:", train_neg)
print("positive:", train_pos)
print("pos_weight:", pos_weight_value)


# =========================
# 7. Edge index check
# =========================

# edge_index가 (E, 2)로 저장돼 있으면 (2, E)로 변환한다.
if len(edge_index.shape) != 2:
    raise ValueError("edge_index는 2차원이어야 함")

if edge_index.shape[0] == 2:
    edge_index_fixed = edge_index
elif edge_index.shape[1] == 2:
    edge_index_fixed = edge_index.T
else:
    raise ValueError("edge_index shape가 이상함: " + str(edge_index.shape))

edge_index_fixed = edge_index_fixed.astype(np.int64)
edge_type = edge_type.astype(np.int64)

num_edges = edge_index_fixed.shape[1]

if len(edge_type) != num_edges:
    raise ValueError("edge_type 길이와 edge 수가 다름")

if edge_index_fixed.min() < 0:
    raise ValueError("edge_index에 음수 node_id가 있음")

if edge_index_fixed.max() >= len(node_df):
    raise ValueError("edge_index에 node 수보다 큰 node_id가 있음")

print("\nedge check 완료")
print("edge_index_fixed shape:", edge_index_fixed.shape)
print("edge_type shape:", edge_type.shape)
print("edge_type unique:", np.unique(edge_type))


# =========================
# 8. Convert to torch tensor
# =========================

x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
valid_mask = torch.tensor(valid_mask_np, dtype=torch.bool)
test_mask = torch.tensor(test_mask_np, dtype=torch.bool)

edge_index_tensor = torch.tensor(edge_index_fixed, dtype=torch.long)
edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)

pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32)

print("\ntensor shape:")
print("x:", x_tensor.shape)
print("y:", y_tensor.shape)
print("train_mask:", train_mask.shape)
print("edge_index:", edge_index_tensor.shape)
print("edge_type:", edge_type_tensor.shape)
print("pos_weight:", pos_weight_tensor)


# =========================
# 9. Save tensor data
# =========================

tensor_data = {
    "x": x_tensor,
    "y": y_tensor,
    "train_mask": train_mask,
    "valid_mask": valid_mask,
    "test_mask": test_mask,
    "edge_index": edge_index_tensor,
    "edge_type": edge_type_tensor,
    "pos_weight": pos_weight_tensor,
    "feature_columns": feature_columns,
    "manual_feature_columns": feature_config["manual_feature_columns"],
    "sbert_feature_columns": feature_config["sbert_feature_columns"],
}

torch.save(tensor_data, TENSOR_DATA_OUTPUT_PATH)

print("\ntensor data 저장 완료:")
print(TENSOR_DATA_OUTPUT_PATH)


# =========================
# 10. Save PyG Data
# =========================

try:
    from torch_geometric.data import Data

    pyg_data = Data(
        x=x_tensor,
        edge_index=edge_index_tensor,
        edge_type=edge_type_tensor,
        y=y_tensor
    )

    pyg_data.train_mask = train_mask
    pyg_data.valid_mask = valid_mask
    pyg_data.test_mask = test_mask
    pyg_data.pos_weight = pos_weight_tensor

    torch.save(pyg_data, PYG_DATA_OUTPUT_PATH)

    pyg_saved = True

    print("\nPyG data 저장 완료:")
    print(PYG_DATA_OUTPUT_PATH)

except ImportError:
    pyg_saved = False

    print("\n주의: torch_geometric을 import하지 못함")
    print("model2_tensor_data.pt는 저장됨")
    print("PyG 학습 코드에서 torch_geometric이 필요할 수 있음")


# =========================
# 11. Save summary
# =========================

edge_type_counts = {}

unique_edge_types = np.unique(edge_type)
for edge_t in unique_edge_types:
    count = int((edge_type == edge_t).sum())
    edge_type_counts[str(int(edge_t))] = count

summary = {
    "node_count": int(len(node_df)),
    "feature_count": int(len(feature_columns)),
    "manual_feature_count": int(len(feature_config["manual_feature_columns"])),
    "sbert_feature_count": int(len(feature_config["sbert_feature_columns"])),
    "edge_count": int(num_edges),
    "edge_type_counts": edge_type_counts,
    "train_count": int(train_mask_np.sum()),
    "valid_count": int(valid_mask_np.sum()),
    "test_count": int(test_mask_np.sum()),
    "train_positive_count": train_pos,
    "train_negative_count": train_neg,
    "pos_weight": float(pos_weight_value),
    "scaler_fit_on": "train only",
    "tensor_data_output_path": str(TENSOR_DATA_OUTPUT_PATH),
    "pyg_data_output_path": str(PYG_DATA_OUTPUT_PATH),
    "pyg_saved": pyg_saved,
    "feature_columns_path": str(FEATURE_COLUMNS_PATH)
}

with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\nsummary 저장 완료:")
print(SUMMARY_OUTPUT_PATH)

print("\n==============================")
print("03_prepare_data.py 완료")
print("==============================")