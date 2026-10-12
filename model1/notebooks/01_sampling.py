import json
from pathlib import Path

import pandas as pd


# ============================================================
# 0. 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]   # ITDA 폴더
DATA_PATH = BASE_DIR / "data" / "yelpzip_preprocessed.csv"

MODEL_DIR = BASE_DIR / "model1"
MODEL_DATA_DIR = MODEL_DIR / "data"
MODEL_OUTPUT_DIR = MODEL_DIR / "outputs"

MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NODES = 25000
MIN_PROD_REVIEW_COUNT = 35
BURST_DAYS = 30
RANDOM_STATE = 42

SAMPLED_PATH = MODEL_DATA_DIR / "model1_sampled_reviews.csv"
PRODUCT_BURST_PATH = MODEL_OUTPUT_DIR / "product_burst_candidates.csv"
SUMMARY_PATH = MODEL_OUTPUT_DIR / "sampling_summary.json"


# ============================================================
# 1. 데이터 불러오기
# ============================================================

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

print("전체 데이터 크기:", df.shape)
print("전체 label 분포:")
print(df["label"].value_counts())
print("전체 fake 비율:", round(df["label"].mean(), 4))


# ============================================================
# 2. product별 가장 리뷰가 몰린 30일 구간 찾기
# ============================================================

def find_best_30d_window(group):
    """
    한 product 안에서 리뷰가 가장 많이 몰린 30일 구간을 찾는다.
    label, rating, user 정보는 사용하지 않는다.
    sampling 기준에는 prod_id와 date만 사용한다.
    """

    group = group.sort_values("date")
    dates = group["date"].reset_index(drop=True)

    left = 0
    best_left = 0
    best_right = 0
    best_count = 0

    for right in range(len(dates)):
        while dates.iloc[right] - dates.iloc[left] > pd.Timedelta(days=BURST_DAYS):
            left += 1

        current_count = right - left + 1

        if current_count > best_count:
            best_count = current_count
            best_left = left
            best_right = right

    return pd.Series({
        "prod_review_count": len(group),
        "max_30d_reviews": best_count,
        "burst_start": dates.iloc[best_left],
        "burst_end": dates.iloc[best_right],
    })


product_burst = (
    df.groupby("prod_id", group_keys=False)
      .apply(find_best_30d_window)
      .reset_index()
)

product_burst["burst30_ratio"] = (
    product_burst["max_30d_reviews"] / product_burst["prod_review_count"]
)

product_burst = product_burst[
    product_burst["prod_review_count"] >= MIN_PROD_REVIEW_COUNT
].copy()

product_burst = product_burst.sort_values(
    by=["burst30_ratio", "max_30d_reviews", "prod_review_count"],
    ascending=[False, False, False]
).reset_index(drop=True)

product_burst.to_csv(PRODUCT_BURST_PATH, index=False, encoding="utf-8-sig")

print("\n후보 product 수:", len(product_burst))
print(product_burst.head(10))


# ============================================================
# 3. Product-Burst Ratio Sampling
# ============================================================

sampled_parts = []
current_nodes = 0

for _, row in product_burst.iterrows():
    if current_nodes >= TARGET_NODES:
        break

    prod_id = row["prod_id"]
    burst_start = row["burst_start"]
    burst_end = row["burst_end"]

    burst_reviews = df[
        (df["prod_id"] == prod_id) &
        (df["date"] >= burst_start) &
        (df["date"] <= burst_end)
    ].copy()

    if len(burst_reviews) == 0:
        continue

    remaining = TARGET_NODES - current_nodes

    if len(burst_reviews) > remaining:
        burst_reviews = burst_reviews.sample(
            n=remaining,
            random_state=RANDOM_STATE
        )

    sampled_parts.append(burst_reviews)
    current_nodes += len(burst_reviews)


sampled_df = pd.concat(sampled_parts, axis=0).copy()

sampled_df = sampled_df.sort_values(["prod_id", "date", "review_id"]).reset_index(drop=True)

# GNN에서 node index로 쓰기 좋은 새 번호
sampled_df["node_id"] = sampled_df.index

# 저장
sampled_df.to_csv(SAMPLED_PATH, index=False, encoding="utf-8-sig")


# ============================================================
# 4. 샘플링 결과 확인
# ============================================================

summary = {
    "sampling_method": "Product-Burst Ratio Sampling",
    "target_nodes": TARGET_NODES,
    "actual_nodes": int(len(sampled_df)),
    "min_prod_review_count": MIN_PROD_REVIEW_COUNT,
    "burst_days": BURST_DAYS,
    "random_state": RANDOM_STATE,
    "num_products": int(sampled_df["prod_id"].nunique()),
    "num_users": int(sampled_df["user_id"].nunique()),
    "normal_count": int((sampled_df["label"] == 0).sum()),
    "fake_count": int((sampled_df["label"] == 1).sum()),
    "fake_ratio": float(sampled_df["label"].mean()),
    "date_min": str(sampled_df["date"].min()),
    "date_max": str(sampled_df["date"].max()),
    "saved_sample_path": str(SAMPLED_PATH),
    "saved_product_burst_path": str(PRODUCT_BURST_PATH),
}

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=4)

print("\n==============================")
print("샘플링 완료")
print("==============================")
print("sample node 수:", len(sampled_df))
print("product 수:", sampled_df["prod_id"].nunique())
print("user 수:", sampled_df["user_id"].nunique())
print("label 분포:")
print(sampled_df["label"].value_counts())
print("fake 비율:", round(sampled_df["label"].mean(), 4))
print("기간:", sampled_df["date"].min(), "~", sampled_df["date"].max())

print("\n저장 완료:")
print("sampled reviews:", SAMPLED_PATH)
print("product burst candidates:", PRODUCT_BURST_PATH)
print("summary:", SUMMARY_PATH)