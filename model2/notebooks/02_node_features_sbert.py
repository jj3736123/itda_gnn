import json
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 0. Path setting
# =========================

CURRENT_FILE = Path(__file__).resolve()

# ITDA/model2/notebooks/02_node_features_sbert.py 기준
MODEL2_DIR = CURRENT_FILE.parents[1]
ITDA_DIR = CURRENT_FILE.parents[2]

RAW_DATA_PATH = ITDA_DIR / "data" / "yelpzip_preprocessed.csv"
SAMPLE_PATH = MODEL2_DIR / "data" / "model2_sampled_reviews.csv"

OUTPUT_DATA_DIR = MODEL2_DIR / "data"
OUTPUT_CONFIG_DIR = MODEL2_DIR / "configs"
OUTPUT_SUMMARY_DIR = MODEL2_DIR / "outputs"

OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

NODE_FEATURE_OUTPUT_PATH = OUTPUT_DATA_DIR / "model2_node_features.csv"
FEATURE_COLUMNS_OUTPUT_PATH = OUTPUT_CONFIG_DIR / "model2_feature_columns.json"
SUMMARY_OUTPUT_PATH = OUTPUT_SUMMARY_DIR / "node_feature_summary.json"


# =========================
# 1. Config
# =========================

SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_BATCH_SIZE = 64

MANUAL_FEATURE_COLUMNS = [
    "rating",
    "rating_extreme",
    "rating_abs_deviation_from_prod_avg",
    "same_rating_prod_7d_count",
    "review_length_log",
    "user_review_count_global",
    "prod_review_count_global",
    "user_age_days_at_review",
]


# =========================
# 2. Load data
# =========================

print("raw data path:", RAW_DATA_PATH)
print("sample path:", SAMPLE_PATH)

raw_df = pd.read_csv(RAW_DATA_PATH)
sample_df = pd.read_csv(SAMPLE_PATH)

print("\nraw shape:", raw_df.shape)
print("sample shape:", sample_df.shape)
print("sample columns:", list(sample_df.columns))

required_sample_cols = [
    "node_id", "review_id", "user_id", "prod_id",
    "rating", "label", "date", "text"
]

missing_cols = [col for col in required_sample_cols if col not in sample_df.columns]

if missing_cols:
    raise ValueError("sample_df에 필요한 컬럼이 없음: " + str(missing_cols))

raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
sample_df["date"] = pd.to_datetime(sample_df["date"], errors="coerce")

if raw_df["date"].isna().sum() > 0:
    raise ValueError("raw_df date 변환 실패 row 수: " + str(raw_df["date"].isna().sum()))

if sample_df["date"].isna().sum() > 0:
    raise ValueError("sample_df date 변환 실패 row 수: " + str(sample_df["date"].isna().sum()))

raw_df["text"] = raw_df["text"].fillna("")
sample_df["text"] = sample_df["text"].fillna("")

sample_df = sample_df.sort_values("node_id").reset_index(drop=True)

print("\nnode_id min:", sample_df["node_id"].min())
print("node_id max:", sample_df["node_id"].max())
print("node_id unique:", sample_df["node_id"].nunique())


# =========================
# 3. Manual feature 1
# rating
# =========================

feature_df = sample_df[
    ["node_id", "review_id", "user_id", "prod_id", "label", "date", "text", "rating"]
].copy()

feature_df["rating"] = feature_df["rating"].astype(float)


# =========================
# 4. Manual feature 2
# rating_extreme
# =========================

feature_df["rating_extreme"] = feature_df["rating"].isin([1, 5, 1.0, 5.0]).astype(int)


# =========================
# 5. Manual feature 3
# rating_abs_deviation_from_prod_avg
# 
# product 평균 rating은 전체 raw data 기준으로 계산한다.
# label은 사용하지 않는다.
# =========================

prod_rating_avg = raw_df.groupby("prod_id")["rating"].mean()

feature_df["prod_rating_avg_global"] = feature_df["prod_id"].map(prod_rating_avg)
feature_df["rating_abs_deviation_from_prod_avg"] = (
    feature_df["rating"] - feature_df["prod_rating_avg_global"]
).abs()

feature_df = feature_df.drop(columns=["prod_rating_avg_global"])


# =========================
# 6. Manual feature 4
# same_rating_prod_7d_count
#
# 같은 prod_id, 같은 rating 안에서
# 현재 review 날짜 기준 ±7일 안에 있는 리뷰 수를 센다.
# 자기 자신은 제외한다.
# sample 내부 기준으로 계산한다.
# =========================

def compute_same_rating_prod_7d_count(df):
    result = np.zeros(len(df), dtype=np.int32)

    temp = df[["node_id", "prod_id", "rating", "date"]].copy()
    temp = temp.sort_values(["prod_id", "rating", "date", "node_id"]).reset_index(drop=True)

    for _, group in temp.groupby(["prod_id", "rating"], sort=False):
        idx_list = group["node_id"].to_numpy()
        dates = group["date"].to_numpy(dtype="datetime64[ns]")

        left = 0
        right = 0
        n = len(group)

        window = np.timedelta64(7, "D")

        for i in range(n):
            current_date = dates[i]

            while left < n and dates[left] < current_date - window:
                left += 1

            if right < i:
                right = i

            while right < n and dates[right] <= current_date + window:
                right += 1

            count = right - left - 1
            node_id = idx_list[i]
            result[node_id] = count

    return result


print("\nsame_rating_prod_7d_count 계산 중...")
feature_df["same_rating_prod_7d_count"] = compute_same_rating_prod_7d_count(feature_df)
print("same_rating_prod_7d_count 계산 완료")


# =========================
# 7. Manual feature 5
# review_length_log
# =========================

review_char_length = feature_df["text"].astype(str).str.len()
feature_df["review_length_log"] = np.log1p(review_char_length)


# =========================
# 8. Manual feature 6
# user_review_count_global
# =========================

user_review_count_global = raw_df.groupby("user_id").size()
feature_df["user_review_count_global"] = feature_df["user_id"].map(user_review_count_global)


# =========================
# 9. Manual feature 7
# prod_review_count_global
# =========================

prod_review_count_global = raw_df.groupby("prod_id").size()
feature_df["prod_review_count_global"] = feature_df["prod_id"].map(prod_review_count_global)


# =========================
# 10. Manual feature 8
# user_age_days_at_review
#
# 전체 raw data에서 해당 user의 첫 리뷰 날짜를 구한 뒤,
# 현재 review 날짜와의 차이를 계산한다.
# =========================

user_first_date = raw_df.groupby("user_id")["date"].min()
feature_df["user_first_date_global"] = feature_df["user_id"].map(user_first_date)

feature_df["user_age_days_at_review"] = (
    feature_df["date"] - feature_df["user_first_date_global"]
).dt.days

feature_df = feature_df.drop(columns=["user_first_date_global"])


# =========================
# 11. Manual feature missing check
# =========================

manual_missing = feature_df[MANUAL_FEATURE_COLUMNS].isna().sum()

print("\nmanual feature 결측치:")
print(manual_missing)

if manual_missing.sum() > 0:
    raise ValueError("manual feature에 결측치가 있음")


# =========================
# 12. SBERT embedding
# =========================

try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print("\n필요 패키지가 설치되지 않음")
    print("아래 명령어를 실행해라:")
    print("& C:/Users/user/AppData/Local/Programs/Python/Python313/python.exe -m pip install -U sentence-transformers tqdm")
    raise e

device = "cuda" if torch.cuda.is_available() else "cpu"

print("\nSBERT model:", SBERT_MODEL_NAME)
print("device:", device)

model = SentenceTransformer(SBERT_MODEL_NAME, device=device)

texts = feature_df["text"].astype(str).tolist()

print("\nSBERT embedding 생성 시작")
print("text 개수:", len(texts))
print("batch size:", SBERT_BATCH_SIZE)

embeddings = model.encode(
    texts,
    batch_size=SBERT_BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

embeddings = embeddings.astype(np.float32)

print("SBERT embedding shape:", embeddings.shape)

sbert_dim = embeddings.shape[1]
sbert_columns = []

for i in range(sbert_dim):
    col_name = "sbert_" + str(i)
    sbert_columns.append(col_name)
    feature_df[col_name] = embeddings[:, i]


# =========================
# 13. Final feature dataframe
# =========================

FINAL_FEATURE_COLUMNS = MANUAL_FEATURE_COLUMNS + sbert_columns

final_df = feature_df[
    ["node_id", "review_id", "user_id", "prod_id", "label"] + FINAL_FEATURE_COLUMNS
].copy()

print("\nfinal node feature shape:", final_df.shape)
print("manual feature 수:", len(MANUAL_FEATURE_COLUMNS))
print("sbert feature 수:", len(sbert_columns))
print("total feature 수:", len(FINAL_FEATURE_COLUMNS))

missing_total = final_df[FINAL_FEATURE_COLUMNS].isna().sum().sum()
print("전체 feature 결측치 수:", missing_total)

if missing_total > 0:
    raise ValueError("최종 feature에 결측치가 있음")


# =========================
# 14. Save
# =========================

final_df.to_csv(NODE_FEATURE_OUTPUT_PATH, index=False, encoding="utf-8-sig")

feature_config = {
    "manual_feature_columns": MANUAL_FEATURE_COLUMNS,
    "text_feature_type": "SBERT",
    "sbert_model_name": SBERT_MODEL_NAME,
    "sbert_batch_size": SBERT_BATCH_SIZE,
    "sbert_feature_columns": sbert_columns,
    "final_feature_columns": FINAL_FEATURE_COLUMNS,
    "manual_feature_count": len(MANUAL_FEATURE_COLUMNS),
    "sbert_feature_count": len(sbert_columns),
    "total_feature_count": len(FINAL_FEATURE_COLUMNS),
    "node_feature_path": str(NODE_FEATURE_OUTPUT_PATH),
    "notes": [
        "model2 reuses model1 sample, split, and edges.",
        "model2 replaces TF-IDF/SVD text features with SBERT embeddings.",
        "rating_deviation_from_prod_avg is replaced by rating_abs_deviation_from_prod_avg.",
        "prod_burst30_ratio is excluded from node features.",
        "new_low_activity_user and user_fake_prior_like_score are excluded."
    ]
}

with open(FEATURE_COLUMNS_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(feature_config, f, ensure_ascii=False, indent=2)

summary = {
    "node_feature_shape": list(final_df.shape),
    "manual_feature_columns": MANUAL_FEATURE_COLUMNS,
    "manual_feature_count": len(MANUAL_FEATURE_COLUMNS),
    "sbert_model_name": SBERT_MODEL_NAME,
    "sbert_feature_count": len(sbert_columns),
    "total_feature_count": len(FINAL_FEATURE_COLUMNS),
    "missing_feature_count": int(missing_total),
    "sample_node_count": int(len(final_df)),
    "label_counts": {
        str(k): int(v)
        for k, v in final_df["label"].value_counts().sort_index().to_dict().items()
    }
}

with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n저장 완료:")
print("node features:", NODE_FEATURE_OUTPUT_PATH)
print("feature config:", FEATURE_COLUMNS_OUTPUT_PATH)
print("summary:", SUMMARY_OUTPUT_PATH)

print("\n==============================")
print("02_node_features_sbert.py 완료")
print("==============================")