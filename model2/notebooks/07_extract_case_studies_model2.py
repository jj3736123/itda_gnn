import json
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# 0. Path setting
# =========================

CURRENT_FILE = Path(__file__).resolve()

# ITDA/model2/notebooks/07_extract_case_studies_model2.py 기준
MODEL2_DIR = CURRENT_FILE.parents[1]
ITDA_DIR = CURRENT_FILE.parents[2]

RAW_DATA_PATH = ITDA_DIR / "data" / "yelpzip_preprocessed.csv"

DATA_DIR = MODEL2_DIR / "data"
OUTPUT_DIR = MODEL2_DIR / "outputs"
REPORT_ASSETS_DIR = MODEL2_DIR / "report_assets"
CASE_STUDY_DIR = REPORT_ASSETS_DIR / "case_studies"

SAMPLE_PATH = DATA_DIR / "model2_sampled_reviews.csv"
NODE_FEATURE_PATH = DATA_DIR / "model2_node_features.csv"
EDGE_LIST_PATH = DATA_DIR / "model2_edge_list.csv"
PREDICTION_PATH = REPORT_ASSETS_DIR / "final_rgcn_tuned_all_predictions.csv"

PRODUCT_SUMMARY_OUTPUT_PATH = REPORT_ASSETS_DIR / "product_suspicious_summary.csv"
HIGH_CONF_OUTPUT_PATH = REPORT_ASSETS_DIR / "high_confidence_fake_reviews.csv"
DASHBOARD_NODES_OUTPUT_PATH = REPORT_ASSETS_DIR / "dashboard_nodes.csv"
DASHBOARD_EDGES_OUTPUT_PATH = REPORT_ASSETS_DIR / "dashboard_edges.csv"
SUMMARY_JSON_PATH = OUTPUT_DIR / "case_study_summary.json"

REPORT_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
CASE_STUDY_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 1. Config
# =========================

HIGH_CONF_THRESHOLD = 0.80
TOP_PRODUCT_COUNT = 10

EDGE_TYPE_NAME = {
    0: "RUR",
    1: "PRODUCT_TIME_7D",
    2: "NEW_USER_BURST_7D"
}


# =========================
# 2. Load data
# =========================

print("sample path:", SAMPLE_PATH)
print("node feature path:", NODE_FEATURE_PATH)
print("edge list path:", EDGE_LIST_PATH)
print("prediction path:", PREDICTION_PATH)

sample_df = pd.read_csv(SAMPLE_PATH)
node_feature_df = pd.read_csv(NODE_FEATURE_PATH)
edge_df = pd.read_csv(EDGE_LIST_PATH)
pred_df = pd.read_csv(PREDICTION_PATH)

print("\nsample shape:", sample_df.shape)
print("node feature shape:", node_feature_df.shape)
print("edge shape:", edge_df.shape)
print("prediction shape:", pred_df.shape)

sample_df["date"] = pd.to_datetime(sample_df["date"], errors="coerce")

if sample_df["date"].isna().sum() > 0:
    raise ValueError("sample_df date 변환 실패 row 수: " + str(sample_df["date"].isna().sum()))

required_pred_cols = ["node_id", "label", "prob_fake", "pred", "split"]

missing_pred_cols = []
for col in required_pred_cols:
    if col not in pred_df.columns:
        missing_pred_cols.append(col)

if len(missing_pred_cols) > 0:
    raise ValueError("prediction 파일에 필요한 컬럼 없음: " + str(missing_pred_cols))


# =========================
# 3. Merge prediction + sample + node features
# =========================

base_cols = [
    "node_id", "review_id", "user_id", "prod_id",
    "rating", "label", "date", "text"
]

sample_small = sample_df[base_cols].copy()

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

node_manual = node_feature_df[manual_cols].copy()

merged_df = sample_small.merge(
    pred_df[["node_id", "prob_fake", "pred", "split"]],
    on="node_id",
    how="left"
)

merged_df = merged_df.merge(
    node_manual,
    on="node_id",
    how="left"
)

missing_pred = merged_df["prob_fake"].isna().sum()

if missing_pred > 0:
    raise ValueError("prediction merge 실패 row 수: " + str(missing_pred))

# NEW_USER_BURST 조건과 같은 기준으로 case study용 indicator 생성
merged_df["new_low_activity_user"] = (
    (merged_df["user_review_count_global"] <= 2)
    & (merged_df["user_age_days_at_review"] <= 30)
).astype(int)

merged_df["high_conf_fake"] = (merged_df["prob_fake"] >= HIGH_CONF_THRESHOLD).astype(int)

print("\nmerged shape:", merged_df.shape)
print("merged columns:", list(merged_df.columns))


# =========================
# 4. Product-level edge summary
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
# 5. Product suspicious summary
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
        avg_review_length_log=("review_length_log", "mean"),
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

# 너무 적은 review 수 product가 상위에 뜨는 것을 줄이기 위해 review_count도 함께 고려
product_summary = product_summary.sort_values(
    by=[
        "avg_fake_probability",
        "pred_fake_ratio",
        "high_conf_fake_ratio",
        "review_count",
        "internal_new_user_burst_7d_edges"
    ],
    ascending=[False, False, False, False, False]
).reset_index(drop=True)

product_summary.insert(0, "suspicious_rank", range(1, len(product_summary) + 1))

product_summary.to_csv(PRODUCT_SUMMARY_OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("\nproduct suspicious summary 저장 완료:")
print(PRODUCT_SUMMARY_OUTPUT_PATH)

print("\n상위 suspicious product 10개:")
print(product_summary.head(10)[[
    "suspicious_rank",
    "prod_id",
    "review_count",
    "avg_fake_probability",
    "pred_fake_ratio",
    "true_fake_ratio",
    "new_low_activity_ratio",
    "internal_new_user_burst_7d_edges",
    "internal_total_edges"
]])


# =========================
# 6. High confidence fake reviews
# =========================

high_conf_df = merged_df[
    merged_df["prob_fake"] >= HIGH_CONF_THRESHOLD
].copy()

high_conf_df = high_conf_df.sort_values(
    by=["prob_fake", "prod_id"],
    ascending=[False, True]
).reset_index(drop=True)

high_conf_cols = [
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

high_conf_df[high_conf_cols].to_csv(
    HIGH_CONF_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\nhigh confidence fake reviews 저장 완료:")
print(HIGH_CONF_OUTPUT_PATH)
print("high confidence fake review 수:", len(high_conf_df))


# =========================
# 7. Dashboard nodes / edges
# =========================

dashboard_nodes = merged_df.copy()

dashboard_nodes["node_label"] = dashboard_nodes["label"].map({
    0: "normal",
    1: "fake"
})

dashboard_nodes["pred_label"] = dashboard_nodes["pred"].map({
    0: "pred_normal",
    1: "pred_fake"
})

dashboard_nodes = dashboard_nodes[[
    "node_id",
    "review_id",
    "user_id",
    "prod_id",
    "split",
    "label",
    "node_label",
    "pred",
    "pred_label",
    "prob_fake",
    "rating",
    "rating_extreme",
    "rating_abs_deviation_from_prod_avg",
    "same_rating_prod_7d_count",
    "review_length_log",
    "user_review_count_global",
    "prod_review_count_global",
    "user_age_days_at_review",
    "new_low_activity_user",
    "date"
]].copy()

dashboard_nodes.to_csv(
    DASHBOARD_NODES_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

dashboard_edges = edge_df.copy()

dashboard_edges["edge_name"] = dashboard_edges["edge_type"].map(EDGE_TYPE_NAME)

dashboard_edges = dashboard_edges[[
    "src",
    "dst",
    "edge_type",
    "edge_name"
]].copy()

dashboard_edges.to_csv(
    DASHBOARD_EDGES_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\ndashboard nodes 저장 완료:")
print(DASHBOARD_NODES_OUTPUT_PATH)
print("dashboard nodes shape:", dashboard_nodes.shape)

print("\ndashboard edges 저장 완료:")
print(DASHBOARD_EDGES_OUTPUT_PATH)
print("dashboard edges shape:", dashboard_edges.shape)


# =========================
# 8. Top product case study files
# =========================

top_products = product_summary.head(TOP_PRODUCT_COUNT)["prod_id"].tolist()

case_study_records = []

for rank, prod_id in enumerate(top_products, start=1):
    prod_summary_row = product_summary[
        product_summary["prod_id"] == prod_id
    ].iloc[0].to_dict()

    prod_reviews = merged_df[
        merged_df["prod_id"] == prod_id
    ].copy()

    prod_reviews = prod_reviews.sort_values(
        by=["prob_fake", "date"],
        ascending=[False, True]
    ).reset_index(drop=True)

    prod_edges = internal_edge_df[
        internal_edge_df["prod_id"] == prod_id
    ].copy()

    review_output_path = CASE_STUDY_DIR / (
        "product_" + str(prod_id) + "_reviews.csv"
    )

    edge_output_path = CASE_STUDY_DIR / (
        "product_" + str(prod_id) + "_internal_edges.csv"
    )

    summary_output_path = CASE_STUDY_DIR / (
        "product_" + str(prod_id) + "_summary.txt"
    )

    prod_reviews[high_conf_cols].to_csv(
        review_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    prod_edges[[
        "src",
        "dst",
        "edge_type",
        "edge_name",
        "prod_id"
    ]].to_csv(
        edge_output_path,
        index=False,
        encoding="utf-8-sig"
    )

    summary_lines = []
    summary_lines.append("Product Case Study")
    summary_lines.append("==================")
    summary_lines.append("")
    summary_lines.append("rank: " + str(rank))
    summary_lines.append("prod_id: " + str(prod_id))
    summary_lines.append("review_count: " + str(int(prod_summary_row["review_count"])))
    summary_lines.append("avg_fake_probability: " + str(round(float(prod_summary_row["avg_fake_probability"]), 4)))
    summary_lines.append("pred_fake_ratio: " + str(round(float(prod_summary_row["pred_fake_ratio"]), 4)))
    summary_lines.append("true_fake_ratio: " + str(round(float(prod_summary_row["true_fake_ratio"]), 4)))
    summary_lines.append("high_conf_fake_count: " + str(int(prod_summary_row["high_conf_fake_count"])))
    summary_lines.append("high_conf_fake_ratio: " + str(round(float(prod_summary_row["high_conf_fake_ratio"]), 4)))
    summary_lines.append("new_low_activity_ratio: " + str(round(float(prod_summary_row["new_low_activity_ratio"]), 4)))
    summary_lines.append("internal_rur_edges: " + str(int(prod_summary_row["internal_rur_edges"])))
    summary_lines.append("internal_product_time_7d_edges: " + str(int(prod_summary_row["internal_product_time_7d_edges"])))
    summary_lines.append("internal_new_user_burst_7d_edges: " + str(int(prod_summary_row["internal_new_user_burst_7d_edges"])))
    summary_lines.append("internal_total_edges: " + str(int(prod_summary_row["internal_total_edges"])))
    summary_lines.append("")
    summary_lines.append("Top suspicious reviews")
    summary_lines.append("----------------------")

    top_review_rows = prod_reviews.head(5)

    for _, r in top_review_rows.iterrows():
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

    with open(summary_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    case_study_records.append({
        "rank": rank,
        "prod_id": int(prod_id),
        "review_file": str(review_output_path),
        "edge_file": str(edge_output_path),
        "summary_file": str(summary_output_path)
    })

print("\ncase study files 저장 완료:")
print(CASE_STUDY_DIR)


# =========================
# 9. Save summary json
# =========================

top_product_records = product_summary.head(TOP_PRODUCT_COUNT).to_dict(orient="records")

summary = {
    "model_name": "model2_rgcn_tuned_with_new_user_burst",
    "prediction_path": str(PREDICTION_PATH),
    "high_conf_threshold": HIGH_CONF_THRESHOLD,
    "top_product_count": TOP_PRODUCT_COUNT,
    "product_summary_output_path": str(PRODUCT_SUMMARY_OUTPUT_PATH),
    "high_conf_output_path": str(HIGH_CONF_OUTPUT_PATH),
    "dashboard_nodes_output_path": str(DASHBOARD_NODES_OUTPUT_PATH),
    "dashboard_edges_output_path": str(DASHBOARD_EDGES_OUTPUT_PATH),
    "case_study_dir": str(CASE_STUDY_DIR),
    "total_high_conf_fake_reviews": int(len(high_conf_df)),
    "top_products": [
        {
            "suspicious_rank": int(row["suspicious_rank"]),
            "prod_id": int(row["prod_id"]),
            "review_count": int(row["review_count"]),
            "avg_fake_probability": float(row["avg_fake_probability"]),
            "pred_fake_ratio": float(row["pred_fake_ratio"]),
            "true_fake_ratio": float(row["true_fake_ratio"]),
            "new_low_activity_ratio": float(row["new_low_activity_ratio"]),
            "internal_new_user_burst_7d_edges": int(row["internal_new_user_burst_7d_edges"]),
            "internal_total_edges": int(row["internal_total_edges"])
        }
        for row in top_product_records
    ],
    "case_study_records": case_study_records
}

with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\nsummary json 저장 완료:")
print(SUMMARY_JSON_PATH)

print("\n==============================")
print("07_extract_case_studies_model2.py 완료")
print("==============================")