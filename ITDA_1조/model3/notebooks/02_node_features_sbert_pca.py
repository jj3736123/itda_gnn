from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import joblib

# ============================================================
# 02_node_features_sbert_pca.py
# 목적:
# - model2에서 가져온 SBERT 384차원 feature를 PCA 128차원으로 축소
# - manual feature 8개 + SBERT PCA 128개 = 총 136개 feature 생성
# - PCA는 train split에만 fit하고, 전체 데이터에 transform 적용
# ============================================================

NOTEBOOK_DIR = Path(__file__).resolve().parent
MODEL3_DIR = NOTEBOOK_DIR.parent

DATA_DIR = MODEL3_DIR / "data"
CONFIG_DIR = MODEL3_DIR / "configs"
OUTPUT_DIR = MODEL3_DIR / "outputs"

SOURCE_FEATURE_PATH = DATA_DIR / "model3_source_node_features_sbert384.csv"
SPLIT_PATH = DATA_DIR / "model3_split.csv"
SOURCE_FEATURE_CONFIG_PATH = CONFIG_DIR / "model3_source_feature_columns_sbert384.json"

OUTPUT_FEATURE_PATH = DATA_DIR / "model3_node_features.csv"
OUTPUT_FEATURE_CONFIG_PATH = CONFIG_DIR / "model3_feature_columns.json"
OUTPUT_PCA_PATH = DATA_DIR / "model3_sbert_pca128.pkl"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "node_feature_pca_summary.json"

N_COMPONENTS = 128
RANDOM_STATE = 42

# ------------------------------------------------------------
# 1. 파일 로드
# ------------------------------------------------------------

print("[LOAD] source feature:", SOURCE_FEATURE_PATH)
features_df = pd.read_csv(SOURCE_FEATURE_PATH)

print("[LOAD] split:", SPLIT_PATH)
split_df = pd.read_csv(SPLIT_PATH)

print("[INFO] source feature shape:", features_df.shape)
print("[INFO] split shape:", split_df.shape)

if "review_id" not in features_df.columns:
    raise ValueError("model3_source_node_features_sbert384.csv 안에 review_id 컬럼이 없음.")

if "review_id" not in split_df.columns:
    raise ValueError("model3_split.csv 안에 review_id 컬럼이 없음.")

# ------------------------------------------------------------
# 2. split 컬럼 찾기
# ------------------------------------------------------------

possible_split_cols = ["split", "set", "data_split"]

split_col = None

for col in possible_split_cols:
    if col in split_df.columns:
        split_col = col
        break

if split_col is None:
    raise ValueError("model3_split.csv에서 split 컬럼을 찾지 못함. 가능한 이름: split, set, data_split")

print("[INFO] split column:", split_col)
print("[INFO] split counts:")
print(split_df[split_col].value_counts())

# ------------------------------------------------------------
# 3. manual feature 컬럼 확인
# ------------------------------------------------------------

manual_feature_columns = [
    "rating",
    "rating_extreme",
    "rating_abs_deviation_from_prod_avg",
    "same_rating_prod_7d_count",
    "review_length_log",
    "user_review_count_global",
    "prod_review_count_global",
    "user_age_days_at_review",
]

missing_manual = []

for col in manual_feature_columns:
    if col not in features_df.columns:
        missing_manual.append(col)

if len(missing_manual) > 0:
    print("[ERROR] missing manual feature columns:")
    for col in missing_manual:
        print("-", col)
    raise ValueError("manual feature 컬럼이 부족함.")

print("[OK] manual feature columns:", len(manual_feature_columns))

# ------------------------------------------------------------
# 4. SBERT 384 feature 컬럼 찾기
# ------------------------------------------------------------

sbert_columns = []

# 4-1. config 파일에서 먼저 찾기
if SOURCE_FEATURE_CONFIG_PATH.exists():
    with open(SOURCE_FEATURE_CONFIG_PATH, "r", encoding="utf-8") as f:
        source_config = json.load(f)

    possible_keys = [
        "sbert_feature_columns",
        "text_feature_columns",
        "embedding_feature_columns",
        "feature_columns"
    ]

    for key in possible_keys:
        if key in source_config and isinstance(source_config[key], list):
            temp_cols = []
            for col in source_config[key]:
                if col in features_df.columns:
                    lower_col = col.lower()
                    if "sbert" in lower_col or "emb" in lower_col or "text" in lower_col:
                        temp_cols.append(col)

            if len(temp_cols) > 0:
                sbert_columns = temp_cols
                print("[INFO] SBERT columns found from config key:", key)
                break

# 4-2. config에서 못 찾으면 컬럼 이름으로 추정
if len(sbert_columns) == 0:
    prefix_candidates = [
        "sbert_",
        "sbert",
        "text_sbert_",
        "text_emb_",
        "emb_",
        "embedding_"
    ]

    for col in features_df.columns:
        lower_col = col.lower()
        for prefix in prefix_candidates:
            if lower_col.startswith(prefix):
                sbert_columns.append(col)
                break

    if len(sbert_columns) > 0:
        print("[INFO] SBERT columns found by prefix")

# 4-3. 그래도 못 찾으면 numeric 컬럼에서 manual/meta 제외 후 추정
if len(sbert_columns) == 0:
    meta_columns = [
        "review_id",
        "user_id",
        "prod_id",
        "label",
        "date",
        "text",
        "tag",
        "split",
        "set",
        "data_split"
    ]

    exclude_columns = set(meta_columns + manual_feature_columns)

    numeric_candidates = []

    for col in features_df.columns:
        if col not in exclude_columns:
            if pd.api.types.is_numeric_dtype(features_df[col]):
                numeric_candidates.append(col)

    if len(numeric_candidates) >= 128:
        sbert_columns = numeric_candidates
        print("[WARNING] SBERT columns inferred from numeric columns.")
    else:
        raise ValueError("SBERT feature 컬럼을 찾지 못함.")

print("[OK] SBERT source columns:", len(sbert_columns))

if len(sbert_columns) < N_COMPONENTS:
    raise ValueError("SBERT 컬럼 수가 PCA 차원보다 적음. SBERT 컬럼 수: " + str(len(sbert_columns)))

# ------------------------------------------------------------
# 5. split 정보 merge
# ------------------------------------------------------------

split_for_merge = split_df[["review_id", split_col]].copy()

feature_split_df = features_df[["review_id"]].merge(
    split_for_merge,
    on="review_id",
    how="left"
)

missing_split_count = feature_split_df[split_col].isna().sum()

if missing_split_count > 0:
    raise ValueError("feature 파일의 review_id 중 split 정보가 없는 행이 있음: " + str(missing_split_count))

train_mask = feature_split_df[split_col].astype(str).str.lower() == "train"
valid_mask = feature_split_df[split_col].astype(str).str.lower().isin(["valid", "val", "validation"])
test_mask = feature_split_df[split_col].astype(str).str.lower() == "test"

print("[INFO] train rows:", int(train_mask.sum()))
print("[INFO] valid rows:", int(valid_mask.sum()))
print("[INFO] test rows:", int(test_mask.sum()))

if train_mask.sum() == 0:
    raise ValueError("train split이 0개임. split 컬럼 값을 확인해야 함.")

# ------------------------------------------------------------
# 6. PCA fit / transform
# ------------------------------------------------------------

print("[PCA] preparing SBERT matrix...")

X_sbert = features_df[sbert_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
X_train_sbert = X_sbert[train_mask.values]

print("[INFO] X_sbert shape:", X_sbert.shape)
print("[INFO] X_train_sbert shape:", X_train_sbert.shape)

print("[PCA] fitting PCA on train split only...")

pca = PCA(
    n_components=N_COMPONENTS,
    random_state=RANDOM_STATE
)

pca.fit(X_train_sbert)

print("[PCA] transforming all data...")

X_pca = pca.transform(X_sbert)

pca_columns = []

for i in range(N_COMPONENTS):
    pca_columns.append("sbert_pca_" + str(i))

pca_df = pd.DataFrame(X_pca, columns=pca_columns)

explained_variance_sum = float(np.sum(pca.explained_variance_ratio_))

print("[OK] PCA completed")
print("[INFO] PCA components:", N_COMPONENTS)
print("[INFO] explained variance ratio sum:", explained_variance_sum)

# ------------------------------------------------------------
# 7. manual feature + PCA feature 저장
# ------------------------------------------------------------

manual_df = features_df[manual_feature_columns].copy()

for col in manual_feature_columns:
    manual_df[col] = pd.to_numeric(manual_df[col], errors="coerce").fillna(0.0)

output_df = pd.concat(
    [
        features_df[["review_id"]].reset_index(drop=True),
        manual_df.reset_index(drop=True),
        pca_df.reset_index(drop=True)
    ],
    axis=1
)

print("[INFO] output feature shape:", output_df.shape)

output_df.to_csv(OUTPUT_FEATURE_PATH, index=False, encoding="utf-8-sig")

print("[SAVED] node features:", OUTPUT_FEATURE_PATH)

# ------------------------------------------------------------
# 8. PCA 모델 저장
# ------------------------------------------------------------

joblib.dump(pca, OUTPUT_PCA_PATH)

print("[SAVED] PCA model:", OUTPUT_PCA_PATH)

# ------------------------------------------------------------
# 9. feature config 저장
# ------------------------------------------------------------

feature_columns = manual_feature_columns + pca_columns

feature_config = {
    "model": "model3",
    "feature_version": "manual8_sbert_pca128",
    "manual_feature_columns": manual_feature_columns,
    "text_feature_columns": pca_columns,
    "feature_columns": feature_columns,
    "num_manual_features": len(manual_feature_columns),
    "num_text_features": len(pca_columns),
    "num_total_features": len(feature_columns),
    "pca": {
        "source": "model2 SBERT 384 feature",
        "n_components": N_COMPONENTS,
        "random_state": RANDOM_STATE,
        "fit_split": "train only",
        "source_sbert_columns_count": len(sbert_columns),
        "explained_variance_ratio_sum": explained_variance_sum
    }
}

with open(OUTPUT_FEATURE_CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(feature_config, f, ensure_ascii=False, indent=2)

print("[SAVED] feature config:", OUTPUT_FEATURE_CONFIG_PATH)

# ------------------------------------------------------------
# 10. summary 저장
# ------------------------------------------------------------

summary = {
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "model3",
    "step": "02_node_features_sbert_pca",
    "input": {
        "source_feature_path": str(SOURCE_FEATURE_PATH),
        "split_path": str(SPLIT_PATH),
        "source_feature_shape": list(features_df.shape),
        "split_shape": list(split_df.shape)
    },
    "feature_design": {
        "manual_features": manual_feature_columns,
        "text_feature": "SBERT PCA",
        "sbert_source_columns_count": len(sbert_columns),
        "pca_components": N_COMPONENTS,
        "total_feature_count": len(feature_columns)
    },
    "pca": {
        "fit_on": "train split only",
        "random_state": RANDOM_STATE,
        "explained_variance_ratio_sum": explained_variance_sum
    },
    "split_counts": {
        "train": int(train_mask.sum()),
        "valid": int(valid_mask.sum()),
        "test": int(test_mask.sum())
    },
    "output": {
        "node_feature_path": str(OUTPUT_FEATURE_PATH),
        "feature_config_path": str(OUTPUT_FEATURE_CONFIG_PATH),
        "pca_model_path": str(OUTPUT_PCA_PATH),
        "output_feature_shape": list(output_df.shape)
    }
}

with open(OUTPUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("[SAVED] summary:", OUTPUT_SUMMARY_PATH)

print("\n[DONE] model3 SBERT PCA node feature generation completed.")
print("다음 단계: 03_edges_text_similarity.py")