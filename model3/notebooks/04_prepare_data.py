from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import joblib

from sklearn.preprocessing import StandardScaler

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ============================================================
# 04_prepare_data.py
# 목적:
# - model3_node_features.csv를 불러와 feature tensor 생성
# - train split에만 StandardScaler fit
# - valid/test에는 transform만 적용
# - base edge 버전과 text similarity 포함 버전을 모두 저장
#
# 생성 파일:
# - model3_standard_scaler.pkl
# - model3_tensor_data.pt
# - model3_pyg_data_base.pt
# - model3_pyg_data_text_similarity.pt
# - prepare_data_summary.json
# ============================================================

NOTEBOOK_DIR = Path(__file__).resolve().parent
MODEL3_DIR = NOTEBOOK_DIR.parent

DATA_DIR = MODEL3_DIR / "data"
CONFIG_DIR = MODEL3_DIR / "configs"
OUTPUT_DIR = MODEL3_DIR / "outputs"

SAMPLED_REVIEWS_PATH = DATA_DIR / "model3_sampled_reviews.csv"
SPLIT_PATH = DATA_DIR / "model3_split.csv"
NODE_FEATURE_PATH = DATA_DIR / "model3_node_features.csv"
FEATURE_CONFIG_PATH = CONFIG_DIR / "model3_feature_columns.json"

BASE_EDGE_INDEX_PATH = DATA_DIR / "model3_edge_index_base.npy"
BASE_EDGE_TYPE_PATH = DATA_DIR / "model3_edge_type_base.npy"

TEXT_EDGE_INDEX_PATH = DATA_DIR / "model3_edge_index.npy"
TEXT_EDGE_TYPE_PATH = DATA_DIR / "model3_edge_type.npy"

SCALER_PATH = DATA_DIR / "model3_standard_scaler.pkl"
TENSOR_DATA_PATH = DATA_DIR / "model3_tensor_data.pt"

PYG_BASE_PATH = DATA_DIR / "model3_pyg_data_base.pt"
PYG_TEXT_PATH = DATA_DIR / "model3_pyg_data_text_similarity.pt"

SUMMARY_PATH = OUTPUT_DIR / "prepare_data_summary.json"


# ------------------------------------------------------------
# 1. 파일 로드
# ------------------------------------------------------------

print("[LOAD] sampled reviews:", SAMPLED_REVIEWS_PATH)
sample_df = pd.read_csv(SAMPLED_REVIEWS_PATH)

print("[LOAD] split:", SPLIT_PATH)
split_df = pd.read_csv(SPLIT_PATH)

print("[LOAD] node features:", NODE_FEATURE_PATH)
feature_df = pd.read_csv(NODE_FEATURE_PATH)

print("[LOAD] feature config:", FEATURE_CONFIG_PATH)
with open(FEATURE_CONFIG_PATH, "r", encoding="utf-8") as f:
    feature_config = json.load(f)

print("[INFO] sample shape:", sample_df.shape)
print("[INFO] split shape:", split_df.shape)
print("[INFO] feature shape:", feature_df.shape)


# ------------------------------------------------------------
# 2. 기본 컬럼 확인
# ------------------------------------------------------------

for col in ["review_id", "label"]:
    if col not in sample_df.columns:
        raise ValueError("sampled reviews에 필요한 컬럼이 없음: " + col)

if "review_id" not in split_df.columns:
    raise ValueError("split 파일에 review_id 컬럼이 없음.")

if "review_id" not in feature_df.columns:
    raise ValueError("feature 파일에 review_id 컬럼이 없음.")

feature_columns = feature_config["feature_columns"]

missing_features = []

for col in feature_columns:
    if col not in feature_df.columns:
        missing_features.append(col)

if len(missing_features) > 0:
    print("[ERROR] missing feature columns:")
    for col in missing_features:
        print("-", col)
    raise ValueError("feature column 누락")

print("[OK] feature columns:", len(feature_columns))


# ------------------------------------------------------------
# 3. split 컬럼 찾기
# ------------------------------------------------------------

possible_split_cols = ["split", "set", "data_split"]
split_col = None

for col in possible_split_cols:
    if col in split_df.columns:
        split_col = col
        break

if split_col is None:
    raise ValueError("split 컬럼을 찾지 못함. 가능한 이름: split, set, data_split")

print("[INFO] split column:", split_col)
print("[INFO] split counts:")
print(split_df[split_col].value_counts())


# ------------------------------------------------------------
# 4. sample 순서 기준으로 feature / split 정렬
# ------------------------------------------------------------
# node index는 sample_df의 row 순서를 기준으로 잡는다.
# edge_index도 기존 model1/model2 sample 순서 기준이라고 보고 유지한다.

sample_df = sample_df.reset_index(drop=True)
sample_df["node_idx"] = np.arange(len(sample_df))

work_df = sample_df[["review_id", "label", "node_idx"]].merge(
    split_df[["review_id", split_col]],
    on="review_id",
    how="left"
)

work_df = work_df.merge(
    feature_df[["review_id"] + feature_columns],
    on="review_id",
    how="left"
)

if work_df[split_col].isna().sum() > 0:
    raise ValueError("split 정보가 없는 review가 있음.")

if work_df[feature_columns[0]].isna().sum() > 0:
    raise ValueError("feature 정보가 없는 review가 있음.")

print("[OK] merged work df shape:", work_df.shape)


# ------------------------------------------------------------
# 5. label 확인
# ------------------------------------------------------------

label_values = sorted(work_df["label"].unique().tolist())
print("[INFO] label unique values:", label_values)

# 이미 전처리된 sample이라고 가정: normal=0, fake=1
# 혹시 원본 label인 -1/1이면 여기서 변환
if set(label_values) == set([-1, 1]):
    print("[WARNING] original label detected. converting -1 -> 1, 1 -> 0")
    work_df["label"] = work_df["label"].map({-1: 1, 1: 0})

label_counts = work_df["label"].value_counts().sort_index()
print("[INFO] label counts:")
print(label_counts)


# ------------------------------------------------------------
# 6. train / valid / test mask 생성
# ------------------------------------------------------------

split_values = work_df[split_col].astype(str).str.lower()

train_mask_np = split_values == "train"
valid_mask_np = split_values.isin(["valid", "val", "validation"])
test_mask_np = split_values == "test"

print("[INFO] train:", int(train_mask_np.sum()))
print("[INFO] valid:", int(valid_mask_np.sum()))
print("[INFO] test:", int(test_mask_np.sum()))

if train_mask_np.sum() == 0:
    raise ValueError("train mask가 0개임.")

if valid_mask_np.sum() == 0:
    raise ValueError("valid mask가 0개임.")

if test_mask_np.sum() == 0:
    raise ValueError("test mask가 0개임.")


# ------------------------------------------------------------
# 7. feature scaling
# ------------------------------------------------------------

X = work_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
y = work_df["label"].astype(int).values

print("[INFO] raw X shape:", X.shape)
print("[INFO] y shape:", y.shape)

scaler = StandardScaler()

X_scaled = X.copy()
X_scaled[train_mask_np.values] = scaler.fit_transform(X[train_mask_np.values])
X_scaled[valid_mask_np.values] = scaler.transform(X[valid_mask_np.values])
X_scaled[test_mask_np.values] = scaler.transform(X[test_mask_np.values])

joblib.dump(scaler, SCALER_PATH)
print("[SAVED] scaler:", SCALER_PATH)


# ------------------------------------------------------------
# 8. tensor 생성
# ------------------------------------------------------------

x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

train_mask = torch.tensor(train_mask_np.values, dtype=torch.bool)
valid_mask = torch.tensor(valid_mask_np.values, dtype=torch.bool)
test_mask = torch.tensor(test_mask_np.values, dtype=torch.bool)

print("[INFO] x tensor:", x_tensor.shape)
print("[INFO] y tensor:", y_tensor.shape)


# ------------------------------------------------------------
# 9. edge 파일 로드
# ------------------------------------------------------------

base_edge_index_np = np.load(BASE_EDGE_INDEX_PATH)
base_edge_type_np = np.load(BASE_EDGE_TYPE_PATH)

text_edge_index_np = np.load(TEXT_EDGE_INDEX_PATH)
text_edge_type_np = np.load(TEXT_EDGE_TYPE_PATH)

if base_edge_index_np.shape[0] != 2:
    base_edge_index_np = base_edge_index_np.T

if text_edge_index_np.shape[0] != 2:
    text_edge_index_np = text_edge_index_np.T

base_edge_type_np = base_edge_type_np.reshape(-1)
text_edge_type_np = text_edge_type_np.reshape(-1)

if base_edge_index_np.shape[1] != len(base_edge_type_np):
    raise ValueError("base edge index와 edge type 길이가 다름.")

if text_edge_index_np.shape[1] != len(text_edge_type_np):
    raise ValueError("text edge index와 edge type 길이가 다름.")

base_edge_index = torch.tensor(base_edge_index_np, dtype=torch.long)
base_edge_type = torch.tensor(base_edge_type_np, dtype=torch.long)

text_edge_index = torch.tensor(text_edge_index_np, dtype=torch.long)
text_edge_type = torch.tensor(text_edge_type_np, dtype=torch.long)

base_num_relations = int(base_edge_type.max().item()) + 1
text_num_relations = int(text_edge_type.max().item()) + 1

print("[INFO] base edge index:", base_edge_index.shape)
print("[INFO] base edge type:", base_edge_type.shape)
print("[INFO] base num relations:", base_num_relations)

print("[INFO] text edge index:", text_edge_index.shape)
print("[INFO] text edge type:", text_edge_type.shape)
print("[INFO] text num relations:", text_num_relations)


# ------------------------------------------------------------
# 10. tensor data 저장
# ------------------------------------------------------------

tensor_data = {
    "x": x_tensor,
    "y": y_tensor,
    "train_mask": train_mask,
    "valid_mask": valid_mask,
    "test_mask": test_mask,
    "base_edge_index": base_edge_index,
    "base_edge_type": base_edge_type,
    "text_edge_index": text_edge_index,
    "text_edge_type": text_edge_type,
    "base_num_relations": base_num_relations,
    "text_num_relations": text_num_relations,
    "feature_columns": feature_columns,
    "review_ids": work_df["review_id"].tolist(),
    "node_idx": work_df["node_idx"].tolist()
}

torch.save(tensor_data, TENSOR_DATA_PATH)
print("[SAVED] tensor data:", TENSOR_DATA_PATH)


# ------------------------------------------------------------
# 11. PyG Data 저장
# ------------------------------------------------------------

if HAS_PYG:
    data_base = Data(
        x=x_tensor,
        y=y_tensor,
        edge_index=base_edge_index,
        edge_type=base_edge_type,
        train_mask=train_mask,
        valid_mask=valid_mask,
        test_mask=test_mask
    )

    data_base.num_relations = base_num_relations

    data_text = Data(
        x=x_tensor,
        y=y_tensor,
        edge_index=text_edge_index,
        edge_type=text_edge_type,
        train_mask=train_mask,
        valid_mask=valid_mask,
        test_mask=test_mask
    )

    data_text.num_relations = text_num_relations

    torch.save(data_base, PYG_BASE_PATH)
    torch.save(data_text, PYG_TEXT_PATH)

    print("[SAVED] PyG base data:", PYG_BASE_PATH)
    print("[SAVED] PyG text similarity data:", PYG_TEXT_PATH)

else:
    print("[WARNING] torch_geometric이 설치되어 있지 않아서 PyG Data 저장은 생략됨.")


# ------------------------------------------------------------
# 12. pos_weight 계산
# ------------------------------------------------------------

train_y = y[train_mask_np.values]
train_pos = int((train_y == 1).sum())
train_neg = int((train_y == 0).sum())

if train_pos == 0:
    pos_weight = 1.0
else:
    pos_weight = train_neg / train_pos

print("[INFO] train negative:", train_neg)
print("[INFO] train positive:", train_pos)
print("[INFO] pos_weight:", pos_weight)


# ------------------------------------------------------------
# 13. summary 저장
# ------------------------------------------------------------

summary = {
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "model3",
    "step": "04_prepare_data",
    "feature": {
        "num_features": len(feature_columns),
        "feature_columns_path": str(FEATURE_CONFIG_PATH),
        "node_feature_path": str(NODE_FEATURE_PATH),
        "scaler_fit": "train only"
    },
    "data": {
        "num_nodes": int(len(work_df)),
        "label_counts": {
            str(k): int(v) for k, v in label_counts.to_dict().items()
        },
        "split_counts": {
            "train": int(train_mask.sum().item()),
            "valid": int(valid_mask.sum().item()),
            "test": int(test_mask.sum().item())
        },
        "pos_weight": float(pos_weight)
    },
    "edges": {
        "base": {
            "edge_count": int(base_edge_index.shape[1]),
            "num_relations": int(base_num_relations)
        },
        "text_similarity": {
            "edge_count": int(text_edge_index.shape[1]),
            "num_relations": int(text_num_relations)
        }
    },
    "outputs": {
        "scaler": str(SCALER_PATH),
        "tensor_data": str(TENSOR_DATA_PATH),
        "pyg_base": str(PYG_BASE_PATH) if HAS_PYG else None,
        "pyg_text_similarity": str(PYG_TEXT_PATH) if HAS_PYG else None
    }
}

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("[SAVED] summary:", SUMMARY_PATH)

print("\n[DONE] model3 prepare data completed.")
print("다음 단계: 05_train_rgcn_pca_original_edges.py")