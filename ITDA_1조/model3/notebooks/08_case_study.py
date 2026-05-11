from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np


# ============================================================
# 08_case_study.py
# 목적:
# - model3 최종 모델 prediction 결과 기반 suspicious product case study 생성
# - model3의 핵심인 TEXT_SIMILARITY edge와 NEW_USER_BURST edge를 함께 해석
# - 보고서 / 대시보드용 CSV 생성
# ============================================================

NOTEBOOK_DIR = Path(__file__).resolve().parent
MODEL3_DIR = NOTEBOOK_DIR.parent

DATA_DIR = MODEL3_DIR / "data"
OUTPUT_DIR = MODEL3_DIR / "outputs"
REPORT_DIR = MODEL3_DIR / "report_assets"

CASE_DIR = REPORT_DIR / "case_studies"
CASE_DIR.mkdir(parents=True, exist_ok=True)

SAMPLED_REVIEWS_PATH = DATA_DIR / "model3_sampled_reviews.csv"
PREDICTION_PATH = REPORT_DIR / "model3_rgcn_pca_text_similarity_all_predictions.csv"
EDGE_LIST_PATH = DATA_DIR / "model3_edge_list.csv"
TEXT_EDGE_LIST_PATH = DATA_DIR / "model3_text_similarity_edge_list.csv"

OUTPUT_PRODUCT_SUMMARY_PATH = CASE_DIR / "model3_product_suspicious_summary.csv"
OUTPUT_HIGH_CONF_REVIEWS_PATH = CASE_DIR / "model3_high_confidence_fake_reviews.csv"
OUTPUT_TOP_PRODUCTS_MD_PATH = CASE_DIR / "model3_representative_case_study_summary.md"
OUTPUT_DASHBOARD_NODES_PATH = CASE_DIR / "model3_dashboard_nodes.csv"
OUTPUT_DASHBOARD_EDGES_PATH = CASE_DIR / "model3_dashboard_edges.csv"
OUTPUT_SUMMARY_JSON_PATH = OUTPUT_DIR / "case_study_summary.json"


MIN_REVIEW_COUNT = 20
HIGH_CONF_THRESHOLD = 0.80
TOP_N_PRODUCTS = 10


# ------------------------------------------------------------
# 1. 파일 로드
# ------------------------------------------------------------

print("[LOAD] sampled reviews:", SAMPLED_REVIEWS_PATH)
sample_df = pd.read_csv(SAMPLED_REVIEWS_PATH)

print("[LOAD] predictions:", PREDICTION_PATH)
pred_df = pd.read_csv(PREDICTION_PATH)

print("[LOAD] edge list:", EDGE_LIST_PATH)
edge_df = pd.read_csv(EDGE_LIST_PATH)

print("[LOAD] text edge list:", TEXT_EDGE_LIST_PATH)
text_edge_df = pd.read_csv(TEXT_EDGE_LIST_PATH)

print("[INFO] sample shape:", sample_df.shape)
print("[INFO] prediction shape:", pred_df.shape)
print("[INFO] edge shape:", edge_df.shape)
print("[INFO] text edge shape:", text_edge_df.shape)


# ------------------------------------------------------------
# 2. 기본 컬럼 확인
# ------------------------------------------------------------

required_sample_cols = ["review_id", "user_id", "prod_id", "rating", "label", "date", "text"]

for col in required_sample_cols:
    if col not in sample_df.columns:
        raise ValueError("sample_df에 필요한 컬럼이 없음: " + col)

required_pred_cols = ["review_id", "true_label", "fake_probability", "pred_label"]

for col in required_pred_cols:
    if col not in pred_df.columns:
        raise ValueError("pred_df에 필요한 컬럼이 없음: " + col)

sample_df = sample_df.reset_index(drop=True)
sample_df["node_idx"] = np.arange(len(sample_df))

work_df = sample_df.merge(
    pred_df,
    on="review_id",
    how="left"
)

if work_df["fake_probability"].isna().sum() > 0:
    raise ValueError("prediction이 없는 review가 있음.")

work_df["date"] = pd.to_datetime(work_df["date"], errors="coerce")
work_df["text"] = work_df["text"].fillna("").astype(str)
work_df["text_length"] = work_df["text"].str.len()
work_df["high_conf_fake"] = work_df["fake_probability"] >= HIGH_CONF_THRESHOLD

print("[OK] merged work shape:", work_df.shape)


# ------------------------------------------------------------
# 3. edge 컬럼 정리
# ------------------------------------------------------------

def normalize_edge_columns(df):
    df = df.copy()

    # src 컬럼
    if "src" not in df.columns:
        for candidate in ["source", "source_node", "src_node", "from"]:
            if candidate in df.columns:
                df["src"] = df[candidate]
                break

    # dst 컬럼
    if "dst" not in df.columns:
        for candidate in ["target", "target_node", "dst_node", "to"]:
            if candidate in df.columns:
                df["dst"] = df[candidate]
                break

    # edge_name 컬럼
    if "edge_name" not in df.columns:
        if "relation" in df.columns:
            df["edge_name"] = df["relation"]
        elif "edge_type_name" in df.columns:
            df["edge_name"] = df["edge_type_name"]
        elif "edge_type" in df.columns:
            df["edge_name"] = df["edge_type"].astype(str)
        else:
            df["edge_name"] = "UNKNOWN"

    if "src" not in df.columns or "dst" not in df.columns:
        raise ValueError("edge list에서 src/dst 컬럼을 찾지 못함.")

    df["src"] = pd.to_numeric(df["src"], errors="coerce").astype("Int64")
    df["dst"] = pd.to_numeric(df["dst"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["src", "dst"]).copy()
    df["src"] = df["src"].astype(int)
    df["dst"] = df["dst"].astype(int)

    return df


edge_df = normalize_edge_columns(edge_df)
text_edge_df = normalize_edge_columns(text_edge_df)

print("[OK] normalized edge shape:", edge_df.shape)
print("[OK] normalized text edge shape:", text_edge_df.shape)

node_to_prod = work_df.set_index("node_idx")["prod_id"].to_dict()
node_to_review = work_df.set_index("node_idx")["review_id"].to_dict()

edge_df["src_prod_id"] = edge_df["src"].map(node_to_prod)
edge_df["dst_prod_id"] = edge_df["dst"].map(node_to_prod)
edge_df["internal_same_product"] = edge_df["src_prod_id"] == edge_df["dst_prod_id"]

text_edge_df["src_prod_id"] = text_edge_df["src"].map(node_to_prod)
text_edge_df["dst_prod_id"] = text_edge_df["dst"].map(node_to_prod)
text_edge_df["internal_same_product"] = text_edge_df["src_prod_id"] == text_edge_df["dst_prod_id"]


# ------------------------------------------------------------
# 4. product별 prediction summary
# ------------------------------------------------------------

product_summary = work_df.groupby("prod_id").agg(
    review_count=("review_id", "count"),
    true_fake_count=("true_label", "sum"),
    pred_fake_count=("pred_label", "sum"),
    high_conf_fake_count=("high_conf_fake", "sum"),
    avg_fake_probability=("fake_probability", "mean"),
    max_fake_probability=("fake_probability", "max"),
    avg_rating=("rating", "mean"),
    avg_text_length=("text_length", "mean"),
    unique_user_count=("user_id", "nunique")
).reset_index()

product_summary["true_fake_ratio"] = product_summary["true_fake_count"] / product_summary["review_count"]
product_summary["pred_fake_ratio"] = product_summary["pred_fake_count"] / product_summary["review_count"]
product_summary["high_conf_fake_ratio"] = product_summary["high_conf_fake_count"] / product_summary["review_count"]


# ------------------------------------------------------------
# 5. product별 edge summary
# ------------------------------------------------------------

internal_edge_df = edge_df[edge_df["internal_same_product"] == True].copy()

edge_count_by_product = internal_edge_df.groupby("src_prod_id").size().reset_index(name="internal_total_edges")
edge_count_by_product = edge_count_by_product.rename(columns={"src_prod_id": "prod_id"})

# edge name 기준 카운트
edge_name_counts = internal_edge_df.groupby(["src_prod_id", "edge_name"]).size().reset_index(name="edge_count")
edge_name_pivot = edge_name_counts.pivot_table(
    index="src_prod_id",
    columns="edge_name",
    values="edge_count",
    fill_value=0
).reset_index()

edge_name_pivot = edge_name_pivot.rename(columns={"src_prod_id": "prod_id"})

product_summary = product_summary.merge(edge_count_by_product, on="prod_id", how="left")
product_summary = product_summary.merge(edge_name_pivot, on="prod_id", how="left")

product_summary["internal_total_edges"] = product_summary["internal_total_edges"].fillna(0).astype(int)

# 존재할 수도 있고 이름이 다를 수도 있는 edge들 처리
for col in ["RUR", "PRODUCT_TIME_7D", "NEW_USER_BURST_7D", "TEXT_SIMILARITY"]:
    if col not in product_summary.columns:
        product_summary[col] = 0

product_summary["internal_rur_edges"] = product_summary["RUR"].fillna(0).astype(int)
product_summary["internal_product_time_7d_edges"] = product_summary["PRODUCT_TIME_7D"].fillna(0).astype(int)
product_summary["internal_new_user_burst_7d_edges"] = product_summary["NEW_USER_BURST_7D"].fillna(0).astype(int)
product_summary["internal_text_similarity_edges"] = product_summary["TEXT_SIMILARITY"].fillna(0).astype(int)


# ------------------------------------------------------------
# 6. text similarity 평균값 추가
# ------------------------------------------------------------

if "similarity" in text_edge_df.columns:
    text_edge_df["similarity"] = pd.to_numeric(text_edge_df["similarity"], errors="coerce")

    text_sim_summary = text_edge_df[text_edge_df["internal_same_product"] == True].groupby("src_prod_id").agg(
        avg_text_similarity=("similarity", "mean"),
        max_text_similarity=("similarity", "max")
    ).reset_index()

    text_sim_summary = text_sim_summary.rename(columns={"src_prod_id": "prod_id"})

    product_summary = product_summary.merge(text_sim_summary, on="prod_id", how="left")
else:
    product_summary["avg_text_similarity"] = np.nan
    product_summary["max_text_similarity"] = np.nan

product_summary["avg_text_similarity"] = product_summary["avg_text_similarity"].fillna(0.0)
product_summary["max_text_similarity"] = product_summary["max_text_similarity"].fillna(0.0)


# ------------------------------------------------------------
# 7. density 계산
# ------------------------------------------------------------

def possible_directed_edges(n):
    if n <= 1:
        return 1
    return n * (n - 1)

product_summary["possible_internal_directed_edges"] = product_summary["review_count"].apply(possible_directed_edges)

product_summary["internal_edge_density"] = (
    product_summary["internal_total_edges"] / product_summary["possible_internal_directed_edges"]
)

product_summary["new_user_burst_edge_density"] = (
    product_summary["internal_new_user_burst_7d_edges"] / product_summary["possible_internal_directed_edges"]
)

product_summary["text_similarity_edge_density"] = (
    product_summary["internal_text_similarity_edges"] / product_summary["possible_internal_directed_edges"]
)


# ------------------------------------------------------------
# 8. representative score 계산
# ------------------------------------------------------------

eligible_df = product_summary[product_summary["review_count"] >= MIN_REVIEW_COUNT].copy()

print("[INFO] eligible products:", len(eligible_df))
print("[INFO] condition: review_count >=", MIN_REVIEW_COUNT)

score_cols = [
    "avg_fake_probability",
    "pred_fake_ratio",
    "high_conf_fake_ratio",
    "new_user_burst_edge_density",
    "text_similarity_edge_density",
    "avg_text_similarity"
]

def minmax(series):
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return pd.Series(np.zeros(len(series)), index=series.index)

    return (series - min_val) / (max_val - min_val)

for col in score_cols:
    eligible_df[col + "_norm"] = minmax(eligible_df[col])

eligible_df["review_count_norm"] = minmax(np.log1p(eligible_df["review_count"]))

eligible_df["representative_score"] = (
    0.25 * eligible_df["avg_fake_probability_norm"]
    + 0.20 * eligible_df["pred_fake_ratio_norm"]
    + 0.15 * eligible_df["high_conf_fake_ratio_norm"]
    + 0.15 * eligible_df["new_user_burst_edge_density_norm"]
    + 0.15 * eligible_df["text_similarity_edge_density_norm"]
    + 0.05 * eligible_df["avg_text_similarity_norm"]
    + 0.05 * eligible_df["review_count_norm"]
)

eligible_df = eligible_df.sort_values(
    by="representative_score",
    ascending=False
).reset_index(drop=True)

eligible_df["case_rank"] = np.arange(1, len(eligible_df) + 1)

eligible_df.to_csv(OUTPUT_PRODUCT_SUMMARY_PATH, index=False, encoding="utf-8-sig")

print("[SAVED] product summary:", OUTPUT_PRODUCT_SUMMARY_PATH)


# ------------------------------------------------------------
# 9. high confidence fake review 저장
# ------------------------------------------------------------

top_product_ids = eligible_df.head(TOP_N_PRODUCTS)["prod_id"].tolist()

high_conf_reviews = work_df[
    (work_df["prod_id"].isin(top_product_ids))
    & (work_df["high_conf_fake"] == True)
].copy()

high_conf_reviews = high_conf_reviews.sort_values(
    by=["prod_id", "fake_probability"],
    ascending=[True, False]
)

high_conf_reviews.to_csv(OUTPUT_HIGH_CONF_REVIEWS_PATH, index=False, encoding="utf-8-sig")

print("[SAVED] high confidence reviews:", OUTPUT_HIGH_CONF_REVIEWS_PATH)


# ------------------------------------------------------------
# 10. dashboard용 nodes / edges 저장
# ------------------------------------------------------------

dashboard_nodes = work_df[work_df["prod_id"].isin(top_product_ids)].copy()

dashboard_nodes = dashboard_nodes[
    [
        "node_idx",
        "review_id",
        "user_id",
        "prod_id",
        "rating",
        "label",
        "true_label",
        "pred_label",
        "fake_probability",
        "date",
        "text"
    ]
].copy()

dashboard_nodes.to_csv(OUTPUT_DASHBOARD_NODES_PATH, index=False, encoding="utf-8-sig")

top_node_set = set(dashboard_nodes["node_idx"].tolist())

dashboard_edges = edge_df[
    edge_df["src"].isin(top_node_set)
    & edge_df["dst"].isin(top_node_set)
].copy()

dashboard_edges.to_csv(OUTPUT_DASHBOARD_EDGES_PATH, index=False, encoding="utf-8-sig")

print("[SAVED] dashboard nodes:", OUTPUT_DASHBOARD_NODES_PATH)
print("[SAVED] dashboard edges:", OUTPUT_DASHBOARD_EDGES_PATH)


# ------------------------------------------------------------
# 11. markdown summary 저장
# ------------------------------------------------------------

md_lines = []

md_lines.append("# Model3 Representative Case Study")
md_lines.append("")
md_lines.append("## Selection Rule")
md_lines.append("")
md_lines.append("- Minimum review count: " + str(MIN_REVIEW_COUNT))
md_lines.append("- High confidence fake threshold: " + str(HIGH_CONF_THRESHOLD))
md_lines.append("- Representative score uses prediction signal, New User Burst structure, and TEXT_SIMILARITY structure.")
md_lines.append("")

md_lines.append("## Top Suspicious Products")
md_lines.append("")
md_lines.append("| Rank | prod_id | Reviews | Avg Fake Prob | Pred Fake Ratio | High Conf Ratio | New User Burst Edges | Text Similarity Edges | Representative Score |")
md_lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

top_cases = eligible_df.head(TOP_N_PRODUCTS).copy()

for _, row in top_cases.iterrows():
    line = (
        "| "
        + str(int(row["case_rank"]))
        + " | "
        + str(int(row["prod_id"]))
        + " | "
        + str(int(row["review_count"]))
        + " | "
        + str(round(float(row["avg_fake_probability"]), 4))
        + " | "
        + str(round(float(row["pred_fake_ratio"]), 4))
        + " | "
        + str(round(float(row["high_conf_fake_ratio"]), 4))
        + " | "
        + str(int(row["internal_new_user_burst_7d_edges"]))
        + " | "
        + str(int(row["internal_text_similarity_edges"]))
        + " | "
        + str(round(float(row["representative_score"]), 4))
        + " |"
    )
    md_lines.append(line)

md_lines.append("")
md_lines.append("## Interpretation")
md_lines.append("")
md_lines.append("Model3 selects representative suspicious products by combining three kinds of evidence.")
md_lines.append("")
md_lines.append("1. Prediction signal: average fake probability and predicted fake ratio.")
md_lines.append("2. Behavioral structure: New User Burst edge density.")
md_lines.append("3. Semantic structure: TEXT_SIMILARITY edge density and average semantic similarity.")
md_lines.append("")
md_lines.append("This supports the model3 argument that fake review detection benefits from combining semantic text representation and graph-based behavioral relations.")

md_text = "\n".join(md_lines)

with open(OUTPUT_TOP_PRODUCTS_MD_PATH, "w", encoding="utf-8") as f:
    f.write(md_text)

print("[SAVED] markdown summary:", OUTPUT_TOP_PRODUCTS_MD_PATH)


# ------------------------------------------------------------
# 12. summary json 저장
# ------------------------------------------------------------

summary = {
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "model3_rgcn_pca128_text_similarity",
    "case_study_rule": {
        "min_review_count": MIN_REVIEW_COUNT,
        "high_conf_threshold": HIGH_CONF_THRESHOLD,
        "top_n_products": TOP_N_PRODUCTS
    },
    "counts": {
        "total_products": int(product_summary["prod_id"].nunique()),
        "eligible_products": int(len(eligible_df)),
        "dashboard_nodes": int(len(dashboard_nodes)),
        "dashboard_edges": int(len(dashboard_edges)),
        "high_confidence_reviews": int(len(high_conf_reviews))
    },
    "top_products": top_cases[
        [
            "case_rank",
            "prod_id",
            "review_count",
            "avg_fake_probability",
            "pred_fake_ratio",
            "high_conf_fake_ratio",
            "internal_new_user_burst_7d_edges",
            "internal_text_similarity_edges",
            "representative_score"
        ]
    ].to_dict(orient="records"),
    "outputs": {
        "product_summary": str(OUTPUT_PRODUCT_SUMMARY_PATH),
        "high_conf_reviews": str(OUTPUT_HIGH_CONF_REVIEWS_PATH),
        "dashboard_nodes": str(OUTPUT_DASHBOARD_NODES_PATH),
        "dashboard_edges": str(OUTPUT_DASHBOARD_EDGES_PATH),
        "markdown_summary": str(OUTPUT_TOP_PRODUCTS_MD_PATH)
    }
}

with open(OUTPUT_SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("[SAVED] summary json:", OUTPUT_SUMMARY_JSON_PATH)

print("\n[TOP CASES]")
print(
    top_cases[
        [
            "case_rank",
            "prod_id",
            "review_count",
            "avg_fake_probability",
            "pred_fake_ratio",
            "high_conf_fake_ratio",
            "internal_new_user_burst_7d_edges",
            "internal_text_similarity_edges",
            "representative_score"
        ]
    ]
)

print("\n[DONE] model3 case study completed.")