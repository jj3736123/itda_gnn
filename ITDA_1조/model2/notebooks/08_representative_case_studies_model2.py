import json
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 0. Path setting
# =========================

CURRENT_FILE = Path(__file__).resolve()

MODEL2_DIR = CURRENT_FILE.parents[1]

DATA_DIR = MODEL2_DIR / "data"
OUTPUT_DIR = MODEL2_DIR / "outputs"
REPORT_ASSETS_DIR = MODEL2_DIR / "report_assets"
REPRESENTATIVE_DIR = REPORT_ASSETS_DIR / "representative_case_studies"

SAMPLE_PATH = DATA_DIR / "model2_sampled_reviews.csv"
NODE_FEATURE_PATH = DATA_DIR / "model2_node_features.csv"
EDGE_LIST_PATH = DATA_DIR / "model2_edge_list.csv"
PREDICTION_PATH = REPORT_ASSETS_DIR / "final_rgcn_tuned_all_predictions.csv"

REP_PRODUCT_SUMMARY_PATH = REPRESENTATIVE_DIR / "representative_product_summary.csv"
REP_MARKDOWN_PATH = REPRESENTATIVE_DIR / "representative_case_study_summary.md"
REP_JSON_PATH = OUTPUT_DIR / "representative_case_study_summary.json"

REPRESENTATIVE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 1. Config
# =========================

MIN_REVIEW_COUNT = 20
HIGH_CONF_THRESHOLD = 0.80
TOP_PRODUCT_COUNT = 10
TOP_REVIEW_COUNT = 10
TOP_PAIR_COUNT = 10

EDGE_TYPE_NAME = {
    0: "RUR",
    1: "PRODUCT_TIME_7D",
    2: "NEW_USER_BURST_7D"
}


# =========================
# 2. Load data
# =========================

sample_df = pd.read_csv(SAMPLE_PATH)
node_feature_df = pd.read_csv(NODE_FEATURE_PATH)
edge_df = pd.read_csv(EDGE_LIST_PATH)
pred_df = pd.read_csv(PREDICTION_PATH)

sample_df["date"] = pd.to_datetime(sample_df["date"], errors="coerce")

print("sample shape:", sample_df.shape)
print("node feature shape:", node_feature_df.shape)
print("edge shape:", edge_df.shape)
print("prediction shape:", pred_df.shape)

sbert_cols = [col for col in node_feature_df.columns if col.startswith("sbert_")]

if len(sbert_cols) == 0:
    raise ValueError("sbert_ 컬럼을 찾지 못함")

print("sbert feature count:", len(sbert_cols))


# =========================
# 3. Merge base data
# =========================

base_cols = [
    "node_id", "review_id", "user_id", "prod_id",
    "rating", "label", "date", "text"
]

manual_cols = [
    "node_id",
    "rating_extreme",
    "rating_abs_deviation_from_prod_avg",
    "same_rating_prod_7d_count",
    "review_length_log",
    "user_review_count_global",
    "prod_review_count_global",
    "user_age_days_at_review"
]

merged_df = sample_df[base_cols].merge(
    pred_df[["node_id", "prob_fake", "pred", "split"]],
    on="node_id",
    how="left"
)

merged_df = merged_df.merge(
    node_feature_df[manual_cols],
    on="node_id",
    how="left"
)

merged_df["new_low_activity_user"] = (
    (merged_df["user_review_count_global"] <= 2)
    & (merged_df["user_age_days_at_review"] <= 30)
).astype(int)

merged_df["high_conf_fake"] = (
    merged_df["prob_fake"] >= HIGH_CONF_THRESHOLD
).astype(int)

if merged_df["prob_fake"].isna().sum() > 0:
    raise ValueError("prediction merge 실패")

sbert_df = node_feature_df[["node_id"] + sbert_cols].copy()


# =========================
# 4. Edge summary
# =========================

node_to_prod = merged_df.set_index("node_id")["prod_id"].to_dict()

edge_df["src_prod_id"] = edge_df["src"].map(node_to_prod)
edge_df["dst_prod_id"] = edge_df["dst"].map(node_to_prod)

internal_edge_df = edge_df[
    edge_df["src_prod_id"].notna()
    & edge_df["dst_prod_id"].notna()
    & (edge_df["src_prod_id"] == edge_df["dst_prod_id"])
].copy()

internal_edge_df["prod_id"] = internal_edge_df["src_prod_id"].astype(int)
internal_edge_df["edge_name"] = internal_edge_df["edge_type"].map(EDGE_TYPE_NAME)

edge_summary = (
    internal_edge_df
    .groupby(["prod_id", "edge_type"])
    .size()
    .reset_index(name="edge_count")
)

edge_pivot = edge_summary.pivot_table(
    index="prod_id",
    columns="edge_type",
    values="edge_count",
    fill_value=0
).reset_index()

for edge_type in [0, 1, 2]:
    if edge_type not in edge_pivot.columns:
        edge_pivot[edge_type] = 0

edge_pivot = edge_pivot.rename(columns={
    0: "internal_rur_edges",
    1: "internal_product_time_7d_edges",
    2: "internal_new_user_burst_7d_edges"
})

edge_pivot["internal_total_edges"] = (
    edge_pivot["internal_rur_edges"]
    + edge_pivot["internal_product_time_7d_edges"]
    + edge_pivot["internal_new_user_burst_7d_edges"]
)


# =========================
# 5. Semantic similarity helper
# =========================

def compute_semantic_metrics(prod_reviews, sbert_df):
    """
    product 내부 high probability review들 사이의 SBERT cosine similarity를 계산한다.
    embedding은 02_node_features_sbert.py에서 normalize_embeddings=True로 생성했으므로
    dot product를 cosine similarity로 볼 수 있다.
    """
    suspicious_reviews = prod_reviews[
        prod_reviews["prob_fake"] >= HIGH_CONF_THRESHOLD
    ].copy()

    if len(suspicious_reviews) < 2:
        suspicious_reviews = prod_reviews.sort_values(
            by="prob_fake",
            ascending=False
        ).head(min(TOP_REVIEW_COUNT, len(prod_reviews))).copy()
    else:
        suspicious_reviews = suspicious_reviews.sort_values(
            by="prob_fake",
            ascending=False
        ).head(min(20, len(suspicious_reviews))).copy()

    if len(suspicious_reviews) < 2:
        return {
            "semantic_review_count": len(suspicious_reviews),
            "semantic_mean_cosine": 0.0,
            "semantic_max_cosine": 0.0,
            "semantic_pair_records": []
        }

    emb = suspicious_reviews[["node_id"]].merge(
        sbert_df,
        on="node_id",
        how="left"
    )

    X = emb[sbert_cols].to_numpy(dtype=np.float32)

    sim_matrix = np.matmul(X, X.T)

    pair_records = []

    for i in range(len(suspicious_reviews)):
        for j in range(i + 1, len(suspicious_reviews)):
            r1 = suspicious_reviews.iloc[i]
            r2 = suspicious_reviews.iloc[j]

            pair_records.append({
                "node_id_1": int(r1["node_id"]),
                "node_id_2": int(r2["node_id"]),
                "prob_fake_1": float(r1["prob_fake"]),
                "prob_fake_2": float(r2["prob_fake"]),
                "label_1": int(r1["label"]),
                "label_2": int(r2["label"]),
                "cosine_similarity": float(sim_matrix[i, j]),
                "text_1": str(r1["text"]),
                "text_2": str(r2["text"])
            })

    pair_records = sorted(
        pair_records,
        key=lambda x: x["cosine_similarity"],
        reverse=True
    )

    cosine_values = [p["cosine_similarity"] for p in pair_records]

    return {
        "semantic_review_count": len(suspicious_reviews),
        "semantic_mean_cosine": float(np.mean(cosine_values)),
        "semantic_max_cosine": float(np.max(cosine_values)),
        "semantic_pair_records": pair_records
    }


# =========================
# 6. Product summary
# =========================

product_summary = (
    merged_df
    .groupby("prod_id")
    .agg(
        review_count=("node_id", "count"),
        avg_fake_probability=("prob_fake", "mean"),
        max_fake_probability=("prob_fake", "max"),
        pred_fake_ratio=("pred", "mean"),
        true_fake_ratio=("label", "mean"),
        high_conf_fake_count=("high_conf_fake", "sum"),
        high_conf_fake_ratio=("high_conf_fake", "mean"),
        new_low_activity_ratio=("new_low_activity_user", "mean"),
        avg_rating=("rating", "mean"),
        avg_rating_abs_deviation=("rating_abs_deviation_from_prod_avg", "mean"),
        avg_same_rating_prod_7d_count=("same_rating_prod_7d_count", "mean"),
        first_review_date=("date", "min"),
        last_review_date=("date", "max")
    )
    .reset_index()
)

product_summary = product_summary.merge(
    edge_pivot,
    on="prod_id",
    how="left"
)

edge_cols = [
    "internal_rur_edges",
    "internal_product_time_7d_edges",
    "internal_new_user_burst_7d_edges",
    "internal_total_edges"
]

for col in edge_cols:
    product_summary[col] = product_summary[col].fillna(0).astype(int)

product_summary["date_span_days"] = (
    pd.to_datetime(product_summary["last_review_date"])
    - pd.to_datetime(product_summary["first_review_date"])
).dt.days


# =========================
# 7. Add semantic metrics for eligible products
# =========================

eligible_products = product_summary[
    product_summary["review_count"] >= MIN_REVIEW_COUNT
]["prod_id"].tolist()

print("eligible product count:", len(eligible_products))
print("condition: review_count >=", MIN_REVIEW_COUNT)

semantic_rows = []

for idx, prod_id in enumerate(eligible_products):
    if (idx + 1) % 50 == 0:
        print("semantic metrics:", idx + 1, "/", len(eligible_products))

    prod_reviews = merged_df[
        merged_df["prod_id"] == prod_id
    ].copy()

    metrics = compute_semantic_metrics(prod_reviews, sbert_df)

    semantic_rows.append({
        "prod_id": prod_id,
        "semantic_review_count": metrics["semantic_review_count"],
        "semantic_mean_cosine": metrics["semantic_mean_cosine"],
        "semantic_max_cosine": metrics["semantic_max_cosine"]
    })

semantic_df = pd.DataFrame(semantic_rows)

product_summary = product_summary.merge(
    semantic_df,
    on="prod_id",
    how="left"
)

product_summary["semantic_review_count"] = product_summary["semantic_review_count"].fillna(0).astype(int)
product_summary["semantic_mean_cosine"] = product_summary["semantic_mean_cosine"].fillna(0.0)
product_summary["semantic_max_cosine"] = product_summary["semantic_max_cosine"].fillna(0.0)


# =========================
# 8. Representative score
# =========================

candidate_df = product_summary[
    product_summary["review_count"] >= MIN_REVIEW_COUNT
].copy()

if len(candidate_df) == 0:
    raise ValueError("대표 case study 후보가 없음. MIN_REVIEW_COUNT를 낮춰야 함")

# scale terms
candidate_df["review_count_score"] = np.log1p(candidate_df["review_count"]) / np.log1p(
    candidate_df["review_count"].max()
)

candidate_df["new_user_edge_density"] = (
    candidate_df["internal_new_user_burst_7d_edges"]
    / (candidate_df["review_count"] * candidate_df["review_count"])
)

candidate_df["new_user_edge_density"] = candidate_df["new_user_edge_density"].clip(0, 1)

# model2는 text 심화 모델이므로 semantic similarity도 대표 사례 선정에 포함한다.
candidate_df["representative_score"] = (
    0.30 * candidate_df["avg_fake_probability"]
    + 0.20 * candidate_df["pred_fake_ratio"]
    + 0.15 * candidate_df["high_conf_fake_ratio"]
    + 0.15 * candidate_df["semantic_mean_cosine"]
    + 0.10 * candidate_df["new_low_activity_ratio"]
    + 0.05 * candidate_df["new_user_edge_density"]
    + 0.05 * candidate_df["review_count_score"]
)

candidate_df = candidate_df.sort_values(
    by=[
        "representative_score",
        "review_count",
        "avg_fake_probability",
        "semantic_mean_cosine",
        "internal_new_user_burst_7d_edges"
    ],
    ascending=[False, False, False, False, False]
).reset_index(drop=True)

candidate_df.insert(0, "representative_rank", range(1, len(candidate_df) + 1))

candidate_df.to_csv(
    REP_PRODUCT_SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\n대표 case study 후보 저장 완료:")
print(REP_PRODUCT_SUMMARY_PATH)

print("\n상위 대표 case study 후보:")
display_cols = [
    "representative_rank",
    "prod_id",
    "review_count",
    "avg_fake_probability",
    "pred_fake_ratio",
    "true_fake_ratio",
    "high_conf_fake_ratio",
    "semantic_mean_cosine",
    "semantic_max_cosine",
    "new_low_activity_ratio",
    "internal_new_user_burst_7d_edges",
    "internal_total_edges",
    "representative_score"
]

print(candidate_df.head(TOP_PRODUCT_COUNT)[display_cols])


# =========================
# 9. Save per-product representative files
# =========================

top_products = candidate_df.head(TOP_PRODUCT_COUNT)["prod_id"].tolist()

case_records = []
markdown_lines = []

markdown_lines.append("# Model2 Representative Case Studies")
markdown_lines.append("")
markdown_lines.append("대표 case study는 단순 avg_fake_probability 순위가 아니라, review_count가 일정 수준 이상인 product만 대상으로 선정하였다.")
markdown_lines.append("")
markdown_lines.append("선정 조건과 기준은 다음과 같다.")
markdown_lines.append("")
markdown_lines.append("```text")
markdown_lines.append("MIN_REVIEW_COUNT = " + str(MIN_REVIEW_COUNT))
markdown_lines.append("HIGH_CONF_THRESHOLD = " + str(HIGH_CONF_THRESHOLD))
markdown_lines.append("selection includes avg_fake_probability, pred_fake_ratio, high_conf_fake_ratio, SBERT semantic similarity, new_low_activity_ratio, New User Burst edge density, review_count")
markdown_lines.append("```")
markdown_lines.append("")

for rank, prod_id in enumerate(top_products, start=1):
    prod_row = candidate_df[candidate_df["prod_id"] == prod_id].iloc[0].to_dict()

    prod_reviews = merged_df[
        merged_df["prod_id"] == prod_id
    ].copy()

    prod_reviews = prod_reviews.sort_values(
        by=["prob_fake", "date"],
        ascending=[False, True]
    ).reset_index(drop=True)

    semantic_metrics = compute_semantic_metrics(prod_reviews, sbert_df)
    pair_records = semantic_metrics["semantic_pair_records"]

    product_dir = REPRESENTATIVE_DIR / ("product_" + str(prod_id))
    product_dir.mkdir(parents=True, exist_ok=True)

    review_output_path = product_dir / "reviews.csv"
    pair_output_path = product_dir / "semantic_pairs.csv"
    summary_output_path = product_dir / "summary.txt"

    review_cols = [
        "node_id",
        "review_id",
        "user_id",
        "prod_id",
        "split",
        "label",
        "pred",
        "prob_fake",
        "rating",
        "rating_extreme",
        "rating_abs_deviation_from_prod_avg",
        "same_rating_prod_7d_count",
        "review_length_log",
        "user_review_count_global",
        "user_age_days_at_review",
        "new_low_activity_user",
        "date",
        "text"
    ]

    prod_reviews[review_cols].to_csv(
        review_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    pair_df = pd.DataFrame(pair_records[:TOP_PAIR_COUNT])

    if len(pair_df) > 0:
        pair_df.to_csv(
            pair_output_path,
            index=False,
            encoding="utf-8-sig"
        )
    else:
        pd.DataFrame(columns=[
            "node_id_1", "node_id_2", "cosine_similarity"
        ]).to_csv(
            pair_output_path,
            index=False,
            encoding="utf-8-sig"
        )

    summary_lines = []
    summary_lines.append("Representative Product Case Study")
    summary_lines.append("================================")
    summary_lines.append("")
    summary_lines.append("rank: " + str(rank))
    summary_lines.append("prod_id: " + str(prod_id))
    summary_lines.append("review_count: " + str(int(prod_row["review_count"])))
    summary_lines.append("avg_fake_probability: " + str(round(float(prod_row["avg_fake_probability"]), 4)))
    summary_lines.append("pred_fake_ratio: " + str(round(float(prod_row["pred_fake_ratio"]), 4)))
    summary_lines.append("true_fake_ratio: " + str(round(float(prod_row["true_fake_ratio"]), 4)))
    summary_lines.append("high_conf_fake_ratio: " + str(round(float(prod_row["high_conf_fake_ratio"]), 4)))
    summary_lines.append("semantic_mean_cosine: " + str(round(float(prod_row["semantic_mean_cosine"]), 4)))
    summary_lines.append("semantic_max_cosine: " + str(round(float(prod_row["semantic_max_cosine"]), 4)))
    summary_lines.append("new_low_activity_ratio: " + str(round(float(prod_row["new_low_activity_ratio"]), 4)))
    summary_lines.append("internal_new_user_burst_7d_edges: " + str(int(prod_row["internal_new_user_burst_7d_edges"])))
    summary_lines.append("internal_total_edges: " + str(int(prod_row["internal_total_edges"])))
    summary_lines.append("representative_score: " + str(round(float(prod_row["representative_score"]), 4)))
    summary_lines.append("")

    summary_lines.append("Top suspicious reviews")
    summary_lines.append("----------------------")

    for _, r in prod_reviews.head(5).iterrows():
        text = str(r["text"]).replace("\n", " ").replace("\r", " ")
        if len(text) > 300:
            text = text[:300] + "..."

        summary_lines.append("")
        summary_lines.append("node_id: " + str(r["node_id"]))
        summary_lines.append("prob_fake: " + str(round(float(r["prob_fake"]), 4)))
        summary_lines.append("label: " + str(int(r["label"])))
        summary_lines.append("pred: " + str(int(r["pred"])))
        summary_lines.append("rating: " + str(r["rating"]))
        summary_lines.append("date: " + str(r["date"]))
        summary_lines.append("new_low_activity_user: " + str(int(r["new_low_activity_user"])))
        summary_lines.append("text: " + text)

    summary_lines.append("")
    summary_lines.append("Top semantic similar pairs")
    summary_lines.append("--------------------------")

    for pair in pair_records[:5]:
        text_1 = str(pair["text_1"]).replace("\n", " ").replace("\r", " ")
        text_2 = str(pair["text_2"]).replace("\n", " ").replace("\r", " ")

        if len(text_1) > 200:
            text_1 = text_1[:200] + "..."

        if len(text_2) > 200:
            text_2 = text_2[:200] + "..."

        summary_lines.append("")
        summary_lines.append("node_id_1: " + str(pair["node_id_1"]))
        summary_lines.append("node_id_2: " + str(pair["node_id_2"]))
        summary_lines.append("cosine_similarity: " + str(round(float(pair["cosine_similarity"]), 4)))
        summary_lines.append("prob_fake_1: " + str(round(float(pair["prob_fake_1"]), 4)))
        summary_lines.append("prob_fake_2: " + str(round(float(pair["prob_fake_2"]), 4)))
        summary_lines.append("text_1: " + text_1)
        summary_lines.append("text_2: " + text_2)

    with open(summary_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    case_records.append({
        "rank": rank,
        "prod_id": int(prod_id),
        "review_count": int(prod_row["review_count"]),
        "avg_fake_probability": float(prod_row["avg_fake_probability"]),
        "pred_fake_ratio": float(prod_row["pred_fake_ratio"]),
        "true_fake_ratio": float(prod_row["true_fake_ratio"]),
        "high_conf_fake_ratio": float(prod_row["high_conf_fake_ratio"]),
        "semantic_mean_cosine": float(prod_row["semantic_mean_cosine"]),
        "semantic_max_cosine": float(prod_row["semantic_max_cosine"]),
        "new_low_activity_ratio": float(prod_row["new_low_activity_ratio"]),
        "internal_new_user_burst_7d_edges": int(prod_row["internal_new_user_burst_7d_edges"]),
        "internal_total_edges": int(prod_row["internal_total_edges"]),
        "representative_score": float(prod_row["representative_score"]),
        "review_output_path": str(review_output_path),
        "semantic_pair_output_path": str(pair_output_path),
        "summary_output_path": str(summary_output_path)
    })

    markdown_lines.append("## Rank " + str(rank) + ": product " + str(prod_id))
    markdown_lines.append("")
    markdown_lines.append("```text")
    markdown_lines.append("review_count: " + str(int(prod_row["review_count"])))
    markdown_lines.append("avg_fake_probability: " + str(round(float(prod_row["avg_fake_probability"]), 4)))
    markdown_lines.append("pred_fake_ratio: " + str(round(float(prod_row["pred_fake_ratio"]), 4)))
    markdown_lines.append("true_fake_ratio: " + str(round(float(prod_row["true_fake_ratio"]), 4)))
    markdown_lines.append("high_conf_fake_ratio: " + str(round(float(prod_row["high_conf_fake_ratio"]), 4)))
    markdown_lines.append("semantic_mean_cosine: " + str(round(float(prod_row["semantic_mean_cosine"]), 4)))
    markdown_lines.append("semantic_max_cosine: " + str(round(float(prod_row["semantic_max_cosine"]), 4)))
    markdown_lines.append("new_low_activity_ratio: " + str(round(float(prod_row["new_low_activity_ratio"]), 4)))
    markdown_lines.append("internal_new_user_burst_7d_edges: " + str(int(prod_row["internal_new_user_burst_7d_edges"])))
    markdown_lines.append("internal_total_edges: " + str(int(prod_row["internal_total_edges"])))
    markdown_lines.append("```")
    markdown_lines.append("")


with open(REP_MARKDOWN_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(markdown_lines))

summary_json = {
    "description": "Representative model2 case studies with minimum review_count and SBERT semantic similarity metrics.",
    "model_name": "model2_rgcn_tuned_with_new_user_burst",
    "min_review_count": MIN_REVIEW_COUNT,
    "high_conf_threshold": HIGH_CONF_THRESHOLD,
    "top_product_count": TOP_PRODUCT_COUNT,
    "representative_product_summary_path": str(REP_PRODUCT_SUMMARY_PATH),
    "representative_markdown_path": str(REP_MARKDOWN_PATH),
    "case_records": case_records
}

with open(REP_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(summary_json, f, ensure_ascii=False, indent=2)

print("\n대표 case study 파일 저장 완료:")
print("summary csv:", REP_PRODUCT_SUMMARY_PATH)
print("markdown:", REP_MARKDOWN_PATH)
print("json:", REP_JSON_PATH)
print("case study dir:", REPRESENTATIVE_DIR)

print("\n==============================")
print("08_representative_case_studies_model2.py 완료")
print("==============================")