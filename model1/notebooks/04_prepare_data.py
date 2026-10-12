import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 0. 기본 설정
# ============================================================

# 현재 파일 위치:
# ITDA/model1/notebooks/04_prepare_data.py
#
# parents[0] = notebooks
# parents[1] = model1
# parents[2] = ITDA

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "model1"
MODEL_DATA_DIR = MODEL_DIR / "data"
MODEL_CONFIG_DIR = MODEL_DIR / "configs"
MODEL_OUTPUT_DIR = MODEL_DIR / "outputs"

NODE_FEATURE_PATH = MODEL_DATA_DIR / "model1_node_features.csv"
EDGE_INDEX_PATH = MODEL_DATA_DIR / "model1_edge_index.npy"
EDGE_TYPE_PATH = MODEL_DATA_DIR / "model1_edge_type.npy"
FEATURE_COLUMNS_PATH = MODEL_CONFIG_DIR / "model1_feature_columns.json"

SPLIT_PATH = MODEL_DATA_DIR / "model1_split.csv"
SCALER_PATH = MODEL_DATA_DIR / "model1_standard_scaler.pkl"
TENSOR_DATA_PATH = MODEL_DATA_DIR / "model1_tensor_data.pt"
PYG_DATA_PATH = MODEL_DATA_DIR / "model1_pyg_data.pt"
PREPARE_SUMMARY_PATH = MODEL_OUTPUT_DIR / "prepare_data_summary.json"

RANDOM_STATE = 42

TEST_SIZE = 0.20
VALID_SIZE_WITHIN_TRAIN_VALID = 0.20
# 최종 비율:
# train = 0.8 * 0.8 = 0.64
# valid = 0.8 * 0.2 = 0.16
# test  = 0.20


# ============================================================
# 1. 데이터 불러오기
# ============================================================

df = pd.read_csv(NODE_FEATURE_PATH)

with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
    feature_info = json.load(f)

feature_cols = feature_info["all_feature_cols"]

edge_index = np.load(EDGE_INDEX_PATH)
edge_type = np.load(EDGE_TYPE_PATH)

print("node feature 데이터:", df.shape)
print("feature 수:", len(feature_cols))
print("edge_index shape:", edge_index.shape)
print("edge_type shape:", edge_type.shape)


# ============================================================
# 2. 기본 검증
# ============================================================

required_cols = ["node_id", "label"] + feature_cols

for col in required_cols:
    if col not in df.columns:
        raise ValueError(col + " 컬럼이 없다. 이전 단계 결과를 확인해야 한다.")

df = df.sort_values("node_id").reset_index(drop=True)

if not (df["node_id"].to_numpy() == np.arange(len(df))).all():
    raise ValueError("node_id가 0부터 연속적으로 정렬되어 있지 않다.")

num_nodes = len(df)

if edge_index.shape[0] != 2:
    raise ValueError("edge_index shape가 이상하다. 첫 번째 차원은 2여야 한다.")

if edge_index.shape[1] != len(edge_type):
    raise ValueError("edge_index의 edge 수와 edge_type 길이가 다르다.")

if edge_index.min() < 0 or edge_index.max() >= num_nodes:
    raise ValueError("edge_index 안에 node 범위를 벗어난 값이 있다.")


# ============================================================
# 3. train / valid / test split
# ============================================================
# 대회 가이드에 맞게 sampling 완료 후 split한다.
# label 비율이 유지되도록 stratify를 사용한다.

indices = np.arange(num_nodes)
labels = df["label"].to_numpy()

train_valid_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
)

train_idx, valid_idx = train_test_split(
    train_valid_idx,
    test_size=VALID_SIZE_WITHIN_TRAIN_VALID,
    random_state=RANDOM_STATE,
    stratify=labels[train_valid_idx]
)

split = np.array([""] * num_nodes, dtype=object)
split[train_idx] = "train"
split[valid_idx] = "valid"
split[test_idx] = "test"

df["split"] = split

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[train_idx] = True
valid_mask[valid_idx] = True
test_mask[test_idx] = True


# ============================================================
# 4. StandardScaler
# ============================================================
# 중요:
# scaler는 train set에만 fit한다.
# valid/test는 train 기준으로 transform만 한다.

X = df[feature_cols].to_numpy(dtype=np.float32)
y = df["label"].to_numpy(dtype=np.int64)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X[train_idx])
X_scaled = X.copy()

X_scaled[train_idx] = X_train_scaled
X_scaled[valid_idx] = scaler.transform(X[valid_idx])
X_scaled[test_idx] = scaler.transform(X[test_idx])

joblib.dump(scaler, SCALER_PATH)


# ============================================================
# 5. tensor 변환
# ============================================================

x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)

# BCEWithLogitsLoss용 pos_weight 계산
# fake = 1 이 minority class
train_labels = y[train_idx]
train_normal_count = int((train_labels == 0).sum())
train_fake_count = int((train_labels == 1).sum())

pos_weight_value = train_normal_count / train_fake_count
pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32)


# ============================================================
# 6. tensor 데이터 저장
# ============================================================

tensor_data = {
    "x": x_tensor,
    "y": y_tensor,
    "edge_index": edge_index_tensor,
    "edge_type": edge_type_tensor,
    "train_mask": train_mask,
    "valid_mask": valid_mask,
    "test_mask": test_mask,
    "feature_cols": feature_cols,
    "pos_weight": pos_weight_tensor,
}

torch.save(tensor_data, TENSOR_DATA_PATH)


# ============================================================
# 7. PyTorch Geometric Data 객체 저장
# ============================================================
# torch_geometric이 설치되어 있으면 바로 Data 객체로도 저장한다.
# 설치가 안 되어 있으면 tensor_data만 저장하고 넘어간다.

pyg_saved = False

try:
    from torch_geometric.data import Data

    pyg_data = Data(
        x=x_tensor,
        y=y_tensor,
        edge_index=edge_index_tensor,
        edge_type=edge_type_tensor,
        train_mask=train_mask,
        valid_mask=valid_mask,
        test_mask=test_mask,
    )

    pyg_data.feature_cols = feature_cols
    pyg_data.pos_weight = pos_weight_tensor

    torch.save(pyg_data, PYG_DATA_PATH)
    pyg_saved = True

except ImportError:
    print("\n주의: torch_geometric이 설치되어 있지 않아서 PyG Data 객체는 저장하지 않았다.")
    print("그래도 tensor_data는 저장되었다.")
    print("나중에 GNN 학습 전에 torch_geometric을 설치하면 된다.")


# ============================================================
# 8. split 파일 및 summary 저장
# ============================================================

df[["node_id", "review_id", "user_id", "prod_id", "label", "split"]].to_csv(
    SPLIT_PATH,
    index=False,
    encoding="utf-8-sig"
)


def split_summary(name):
    part = df[df["split"] == name]
    return {
        "count": int(len(part)),
        "normal_count": int((part["label"] == 0).sum()),
        "fake_count": int((part["label"] == 1).sum()),
        "fake_ratio": float(part["label"].mean())
    }


summary = {
    "num_nodes": int(num_nodes),
    "num_features": int(len(feature_cols)),
    "num_edges": int(edge_index.shape[1]),
    "num_edge_types": int(len(np.unique(edge_type))),
    "random_state": RANDOM_STATE,
    "split_ratio": {
        "train": 0.64,
        "valid": 0.16,
        "test": 0.20
    },
    "split_summary": {
        "train": split_summary("train"),
        "valid": split_summary("valid"),
        "test": split_summary("test")
    },
    "train_normal_count": train_normal_count,
    "train_fake_count": train_fake_count,
    "pos_weight": float(pos_weight_value),
    "pyg_data_saved": pyg_saved,
    "saved_split_path": str(SPLIT_PATH),
    "saved_scaler_path": str(SCALER_PATH),
    "saved_tensor_data_path": str(TENSOR_DATA_PATH),
    "saved_pyg_data_path": str(PYG_DATA_PATH) if pyg_saved else None
}

with open(PREPARE_SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=4)


# ============================================================
# 9. 결과 출력
# ============================================================

print("\n==============================")
print("data 준비 완료")
print("==============================")

print("x shape:", x_tensor.shape)
print("y shape:", y_tensor.shape)
print("edge_index shape:", edge_index_tensor.shape)
print("edge_type shape:", edge_type_tensor.shape)

print("\ntrain/valid/test 분포:")
print(df.groupby("split")["label"].value_counts())
print("\nsplit별 fake 비율:")
print(df.groupby("split")["label"].mean())

print("\npos_weight:", round(pos_weight_value, 4))

print("\n저장 완료:")
print("split file:", SPLIT_PATH)
print("scaler:", SCALER_PATH)
print("tensor data:", TENSOR_DATA_PATH)

if pyg_saved:
    print("pyg data:", PYG_DATA_PATH)
else:
    print("pyg data: 저장 안 됨")