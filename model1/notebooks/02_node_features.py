import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# ============================================================
# 0. 기본 설정
# ============================================================


BASE_DIR = Path(__file__).resolve().parents[2]   # ITDA 폴더

FULL_DATA_PATH = BASE_DIR / "data" / "yelpzip_preprocessed.csv"
SAMPLED_PATH = BASE_DIR / "model1" / "data" / "model1_sampled_reviews.csv"

MODEL_DIR = BASE_DIR / "model1"
MODEL_DATA_DIR = MODEL_DIR / "data"
MODEL_CONFIG_DIR = MODEL_DIR / "configs"
MODEL_OUTPUT_DIR = MODEL_DIR / "outputs"

MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NODE_FEATURE_PATH = MODEL_DATA_DIR / "model1_node_features.csv"
FEATURE_COLUMNS_PATH = MODEL_CONFIG_DIR / "model1_feature_columns.json"
SUMMARY_PATH = MODEL_OUTPUT_DIR / "node_feature_summary.json"

RANDOM_STATE = 42
TEXT_SVD_DIM = 32
TIME_WINDOW_DAYS = 7


# ============================================================
# 1. 데이터 불러오기
# ============================================================

df_full = pd.read_csv(FULL_DATA_PATH)
df_sample = pd.read_csv(SAMPLED_PATH)

df_full["date"] = pd.to_datetime(df_full["date"], errors="coerce")
df_sample["date"] = pd.to_datetime(df_sample["date"], errors="coerce")

df_full = df_full.dropna(subset=["date"])
df_sample = df_sample.dropna(subset=["date"])

df_full["text"] = df_full["text"].astype(str)
df_sample["text"] = df_sample["text"].astype(str)

print("전체 데이터:", df_full.shape)
print("샘플 데이터:", df_sample.shape)
print("샘플 label 분포:")
print(df_sample["label"].value_counts())
print("샘플 fake 비율:", round(df_sample["label"].mean(), 4))


# ============================================================
# 2. 전체 데이터 기준 user 통계 생성
# ============================================================
# sample 안에서만 계산하면 오래 활동한 유저가 신규 유저처럼 보일 수 있다.
# 따라서 user 활동량은 반드시 전체 데이터 기준으로 계산한다.

user_stats = (
    df_full.groupby("user_id")
    .agg(
        user_review_count_global=("review_id", "count"),
        user_first_review_date=("date", "min")
    )
    .reset_index()
)

df_sample = df_sample.merge(user_stats, on="user_id", how="left")

df_sample["user_age_days_at_review"] = (
    df_sample["date"] - df_sample["user_first_review_date"]
).dt.days

df_sample["user_age_days_at_review"] = df_sample["user_age_days_at_review"].clip(lower=0)


# ============================================================
# 3. 전체 데이터 기준 product 통계 생성
# ============================================================

prod_stats = (
    df_full.groupby("prod_id")
    .agg(
        prod_review_count_global=("review_id", "count"),
        prod_avg_rating_global=("rating", "mean")
    )
    .reset_index()
)

df_sample = df_sample.merge(prod_stats, on="prod_id", how="left")

df_sample["rating_deviation_from_prod_avg"] = (
    df_sample["rating"] - df_sample["prod_avg_rating_global"]
)


# ============================================================
# 4. rating 관련 feature 생성
# ============================================================

df_sample["rating_extreme"] = df_sample["rating"].isin([1, 5]).astype(int)


# ============================================================
# 5. 같은 product + 같은 rating + 7일 이내 리뷰 수
# ============================================================
# Product-Time-Rating edge를 따로 만들지 않는 대신,
# 같은 product에서 같은 rating이 짧은 기간에 얼마나 몰렸는지를 node feature로 요약한다.
# 여기서는 sample subgraph 안에서만 계산한다.

def calculate_same_rating_prod_7d_count(data):
    result = pd.Series(0, index=data.index, dtype=int)
    time_window = pd.Timedelta(days=TIME_WINDOW_DAYS)

    grouped = data.groupby(["prod_id", "rating"])

    for _, group in grouped:
        group_sorted = group.sort_values("date")
        index_list = group_sorted.index.to_list()
        date_list = group_sorted["date"].to_list()

        left = 0
        right = 0

        for i in range(len(date_list)):
            while date_list[i] - date_list[left] > time_window:
                left += 1

            if right < i:
                right = i

            while right + 1 < len(date_list) and date_list[right + 1] - date_list[i] <= time_window:
                right += 1

            # right - left 는 자기 자신을 제외한 개수다.
            result.loc[index_list[i]] = right - left

    return result


df_sample["same_rating_prod_7d_count"] = calculate_same_rating_prod_7d_count(df_sample)


# ============================================================
# 6. text 길이 feature 생성
# ============================================================

df_sample["review_length"] = df_sample["text"].str.len()
df_sample["review_length_log"] = np.log1p(df_sample["review_length"])


# ============================================================
# 7. text TF-IDF + SVD feature 생성
# ============================================================
# BERT까지 바로 가면 코드가 무거워진다.
# 일단은 TF-IDF를 만들고, SVD로 32차원으로 줄여서 node feature로 사용한다.

tfidf = TfidfVectorizer(
    max_features=5000,
    min_df=3,
    max_df=0.90,
    stop_words="english"
)

text_tfidf = tfidf.fit_transform(df_sample["text"])

actual_svd_dim = min(TEXT_SVD_DIM, text_tfidf.shape[1] - 1)

if actual_svd_dim < 1:
    print("TF-IDF feature 수가 너무 적어서 text_svd feature를 만들지 않는다.")
    text_feature_cols = []
else:
    svd = TruncatedSVD(
        n_components=actual_svd_dim,
        random_state=RANDOM_STATE
    )

    text_svd = svd.fit_transform(text_tfidf)

    text_feature_cols = []

    for i in range(actual_svd_dim):
        col_name = "text_svd_" + str(i)
        df_sample[col_name] = text_svd[:, i]
        text_feature_cols.append(col_name)

    explained = svd.explained_variance_ratio_.sum()
    print("TF-IDF shape:", text_tfidf.shape)
    print("SVD 차원:", actual_svd_dim)
    print("SVD explained variance ratio 합:", round(explained, 4))


# ============================================================
# 8. 최종 feature column 정리
# ============================================================

manual_feature_cols = [
    "rating",
    "rating_extreme",
    "rating_deviation_from_prod_avg",
    "same_rating_prod_7d_count",
    "review_length_log",
    "user_review_count_global",
    "prod_review_count_global",
    "user_age_days_at_review"
]

feature_cols = manual_feature_cols + text_feature_cols

# 혹시 모를 결측치 / 무한값 처리
df_sample[feature_cols] = df_sample[feature_cols].replace([np.inf, -np.inf], np.nan)
df_sample[feature_cols] = df_sample[feature_cols].fillna(0)


# ============================================================
# 9. 저장
# ============================================================

df_sample.to_csv(NODE_FEATURE_PATH, index=False, encoding="utf-8-sig")

feature_info = {
    "manual_feature_cols": manual_feature_cols,
    "text_feature_cols": text_feature_cols,
    "all_feature_cols": feature_cols,
    "num_features": len(feature_cols),
    "text_svd_dim_requested": TEXT_SVD_DIM,
    "text_svd_dim_actual": len(text_feature_cols),
    "time_window_days_for_same_rating": TIME_WINDOW_DAYS
}

with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
    json.dump(feature_info, f, ensure_ascii=False, indent=4)

summary = {
    "num_nodes": int(len(df_sample)),
    "num_features": int(len(feature_cols)),
    "manual_feature_count": int(len(manual_feature_cols)),
    "text_feature_count": int(len(text_feature_cols)),
    "normal_count": int((df_sample["label"] == 0).sum()),
    "fake_count": int((df_sample["label"] == 1).sum()),
    "fake_ratio": float(df_sample["label"].mean()),
    "saved_node_feature_path": str(NODE_FEATURE_PATH),
    "saved_feature_columns_path": str(FEATURE_COLUMNS_PATH)
}

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=4)


# ============================================================
# 10. 결과 확인
# ============================================================

print("\n==============================")
print("node feature 생성 완료")
print("==============================")
print("node 수:", len(df_sample))
print("feature 수:", len(feature_cols))
print("manual feature 수:", len(manual_feature_cols))
print("text feature 수:", len(text_feature_cols))

print("\n사용 feature:")
for col in feature_cols:
    print("-", col)

print("\n결측치 개수:")
print(df_sample[feature_cols].isna().sum().sum())

print("\n저장 완료:")
print("node feature file:", NODE_FEATURE_PATH)
print("feature columns file:", FEATURE_COLUMNS_PATH)
print("summary file:", SUMMARY_PATH)