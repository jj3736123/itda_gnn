from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np


# ============================================================
# 10_case_study_model3_final.py
# 목적:
# - 최종 모델 model3_final 기준 case study 생성
# - model3_final_all_predictions.csv 사용
# - model3_final_text_similarity_edge_list.csv 사용
# - base edge + final text edge를 함께 해석
#
# 최종 모델:
# - model3_final
# - SIM_THRESHOLD = 0.85
# - TOP_K = 5
# - TEXT_SIMILARITY edge count = 822
# ============================================================


# ------------------------------------------------------------
# 0. 경로 설정
# ------------------------------------------------------------

NOTEBOOK_DIR = Path(__file__).resolve().parent
MODEL3_DIR = NOTEBOOK_DIR.parent

DATA_DIR = MODEL3_DIR / "data"
OUTPUT_DIR = MODEL3_DIR / "outputs"
REPORT_DIR = MODEL3_DIR / "report_assets"

CASE_DIR = REPORT_DIR / "model3_final_case_studies"
CASE_DIR.mkdir(parents=True, exist_ok=True)

SAMPLED_REVIEWS_PATH = DATA_DIR / "model3_sampled_reviews.csv"

# 최종 model3_final 결과
FINAL_RESULT_PATH = OUTPUT_DIR / "model3_final_result.json"
FINAL_PREDICTION_PATH = REPORT_DIR / "model3_final_all_predictions.csv"
FINAL_TEXT_EDGE_LIST_PATH = DATA_DIR / "model3_final_text_similarity_edge_list.csv"

# base edge는 기존 model2/model3 base edge 사용
BASE_EDGE_LIST_PATH = DATA_DIR / "model3_edge_list_base.csv"

OUTPUT_PRODUCT_SUMMARY_PATH = CASE_DIR / "model3_final_product_suspicious_summary.csv"
OUTPUT_HIGH_CONF_REVIEWS_PATH = CASE_DIR / "model3_final_high_confidence_fake_reviews.csv"
OUTPUT_DASHBOARD_NODES_PATH = CASE_DIR / "model3_final_dashboard_nodes.csv"
OUTPUT_DASHBOARD_EDGES_PATH = CASE_DIR / "model3_final_dashboard_edges.csv"
OUTPUT_MD_PATH = CASE_DIR / "model3_final_representative_case_study_summary.md"
OUTPUT_SUMMARY_JSON_PATH = OUTPUT_DIR / "model3_final_case_study_summary.json"


# ------------------------------------------------------------
# 1. 설정
# ------------------------------------------------------------

MIN_REVIEW_COUNT = 20
HIGH_CONF_THRESHOLD = 0.80
TOP_N_PRODUCTS = 10


# ------------------------------------------------------------
# 2. 파일 로드
# ------------------------------------------------------------

print("[LOAD] sampled reviews:", SAMPLED_REVIEWS_PATH)
sample_df = pd.read_csv(SAMPLED_REVIEWS_PATH)

print("[LOAD] final predictions:", FINAL_PREDICTION_PATH)
pred_df = pd.read_csv(FINAL_PREDICTION_PATH)

print("[LOAD] base edge list:", BASE_EDGE_LIST_PATH)
base_edge_df = pd.read_csv(BASE_EDGE_LIST_PATH)

print("[LOAD] final text similarity edge list:", FINAL_TEXT_EDGE_LIST_PATH)
text_edge_df = pd.read_csv(FINAL_TEXT_EDGE_LIST_PATH)

print("[LOAD] final result:", FINAL_RESULT_PATH)
with open(FINAL_RESULT_PATH, "r", encoding="utf-8") as f:
    final_result = json.load(f)

print("[INFO] sample shape:", sample_df.shape)
print("[INFO] prediction shape:", pred_df.shape)
print("[INFO] base edge shape:", base_edge_df.shape)
print("[INFO] final text edge shape:", text_edge_df.shape)


# ------------------------------------------------------------
# 3. 기본 컬럼 확인
# ------------------------------------------------------------

required_sample_cols = [
    "review_id",
    "user_id",
    "prod_id",
    "rating",
    "label",
    "date",
    "text"
]

for col in required_sample_cols:
    if col not in sample_df.columns:
        raise ValueError("sample_df에 필요한 컬럼이 없음: " + col)

required_pred_cols = [
    "review_id",
    "true_label",
    "fake_probability",
    "pred_label"
]

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
# 4. edge column 정리 함수
# ------------------------------------------------------------

def normalize_edge_columns(df, is_text_edge=False):
    df = df.copy()

    if "src" not in df.columns:
        for candidate in ["source", "source_node", "src_node", "from"]:
            if candidate in df.columns:
                df["src"] = df[candidate]
                break

    if "dst" not in df.columns:
        for candidate in ["target", "target_node", "dst_node", "to"]:
            if candidate in df.columns:
                df["dst"] = df[candidate]
                break

    if "src" not in df.columns or "dst" not in df.columns:
        raise ValueError("edge list에서 src/dst 컬럼을 찾지 못함.")

    df["src"] = pd.to_numeric(df["src"], errors="coerce")
    df["dst"] = pd.to_numeric(df["dst"], errors="coerce")
    df = df.dropna(subset=["src", "dst"]).copy()
    df["src"] = df["src"].astype(int)
    df["dst"] = df["dst"].astype(int)

    if "edge_name" not in df.columns:
        if "relation" in df.columns:
            df["edge_name"] = df["relation"]
        elif "edge_type_name" in df.columns:
            df["edge_name"] = df["edge_type_name"]
        elif "edge_type" in df.columns:
            # base edge_type이 숫자로만 있을 경우 mapping
            edge_type_numeric = pd.to_numeric(df["edge_type"], errors="coerce")

            if edge_type_numeric.notna().all() and is_text_edge == False:
                mapping = {
                    0: "RUR",
                    1: "PRODUCT_TIME_7D",
                    2: "NEW_USER_BURST_7D",
                    3: "TEXT_SIMILARITY"
                }

                df["edge_name"] = edge_type_numeric.astype(int).map(mapping).fillna(
                    df["edge_type"].astype(str)
                )
            else:
                df["edge_name"] = df["edge_type"].astype(str)
        else:
            if is_text_edge:
                df["edge_name"] = "TEXT_SIMILARITY"
            else:
                df["edge_name"] = "UNKNOWN"

    if is_text_edge:
        df["edge_name"] = "TEXT_SIMILARITY"

    return df


base_edge_df = normalize_edge_columns(base_edge_df, is_text_edge=False)
text_edge_df = normalize_edge_columns(text_edge_df, is_text_edge=True)

print("[OK] normalized base edge shape:", base_edge_df.shape)
print("[OK] normalized text edge shape:", text_edge_df.shape)


# ------------------------------------------------------------
# 5. final combined edge list 생성
# ------------------------------------------------------------

base_for_save = base_edge_df.copy()

if "similarity" not in base_for_save.columns:
    base_for_save["similarity"] = np.nan

text_for_save = text_edge_df.copy()

combined_edge_df = pd.concat(
    [base_for_save, text_for_save],
    axis=0,
    ignore_index=True
)

print("[INFO] combined edge shape:", combined_edge_df.shape)


# ------------------------------------------------------------
# 6. node to product mapping
# ------------------------------------------------------------

node_to_prod = work_df.set_index("node_idx")["prod_id"].to_dict()
node_to_review = work_df.set_index("node_idx")["review_id"].to_dict()

combined_edge_df["src_prod_id"] = combined_edge_df["src"].map(node_to_prod)
combined_edge_df["dst_prod_id"] = combined_edge_df["dst"].map(node_to_prod)
combined_edge_df["internal_same_product"] = (
    combined_edge_df["src_prod_id"] == combined_edge_df["dst_prod_id"]
)

text_edge_df["src_prod_id"] = text_edge_df["src"].map(node_to_prod)
text_edge_df["dst_prod_id"] = text_edge_df["dst"].map(node_to_prod)
text_edge_df["internal_same_product"] = (
    text_edge_df["src_prod_id"] == text_edge_df["dst_prod_id"]
)


# ------------------------------------------------------------
# 7. product별 prediction summary
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

product_summary["true_fake_ratio"] = (
    product_summary["true_fake_count"] / product_summary["review_count"]
)

product_summary["pred_fake_ratio"] = (
    product_summary["pred_fake_count"] / product_summary["review_count"]
)

product_summary["high_conf_fake_ratio"] = (
    product_summary["high_conf_fake_count"] / product_summary["review_count"]
)


# ------------------------------------------------------------
# 8. product별 edge summary
# ------------------------------------------------------------

internal_edge_df = combined_edge_df[
    combined_edge_df["internal_same_product"] == True
].copy()

edge_count_by_product = internal_edge_df.groupby("src_prod_id").size().reset_index(
    name="internal_total_edges"
)

edge_count_by_product = edge_count_by_product.rename(
    columns={"src_prod_id": "prod_id"}
)

edge_name_counts = internal_edge_df.groupby(
    ["src_prod_id", "edge_name"]
).size().reset_index(name="edge_count")

edge_name_pivot = edge_name_counts.pivot_table(
    index="src_prod_id",
    columns="edge_name",
    values="edge_count",
    fill_value=0
).reset_index()

edge_name_pivot = edge_name_pivot.rename(columns={"src_prod_id": "prod_id"})

product_summary = product_summary.merge(
    edge_count_by_product,
    on="prod_id",
    how="left"
)

product_summary = product_summary.merge(
    edge_name_pivot,
    on="prod_id",
    how="left"
)

product_summary["internal_total_edges"] = (
    product_summary["internal_total_edges"].fillna(0).astype(int)
)

for col in ["RUR", "PRODUCT_TIME_7D", "NEW_USER_BURST_7D", "TEXT_SIMILARITY"]:
    if col not in product_summary.columns:
        product_summary[col] = 0

product_summary["internal_rur_edges"] = product_summary["RUR"].fillna(0).astype(int)
product_summary["internal_product_time_7d_edges"] = (
    product_summary["PRODUCT_TIME_7D"].fillna(0).astype(int)
)
product_summary["internal_new_user_burst_7d_edges"] = (
    product_summary["NEW_USER_BURST_7D"].fillna(0).astype(int)
)
product_summary["internal_text_similarity_edges"] = (
    product_summary["TEXT_SIMILARITY"].fillna(0).astype(int)
)


# ------------------------------------------------------------
# 9. text similarity 평균값 추가
# ------------------------------------------------------------

if "similarity" in text_edge_df.columns:
    text_edge_df["similarity"] = pd.to_numeric(
        text_edge_df["similarity"],
        errors="coerce"
    )

    text_sim_summary = text_edge_df[
        text_edge_df["internal_same_product"] == True
    ].groupby("src_prod_id").agg(
        avg_text_similarity=("similarity", "mean"),
        max_text_similarity=("similarity", "max")
    ).reset_index()

    text_sim_summary = text_sim_summary.rename(columns={"src_prod_id": "prod_id"})

    product_summary = product_summary.merge(
        text_sim_summary,
        on="prod_id",
        how="left"
    )
else:
    product_summary["avg_text_similarity"] = np.nan
    product_summary["max_text_similarity"] = np.nan

product_summary["avg_text_similarity"] = (
    product_summary["avg_text_similarity"].fillna(0.0)
)

product_summary["max_text_similarity"] = (
    product_summary["max_text_similarity"].fillna(0.0)
)


# ------------------------------------------------------------
# 10. density 계산
# ------------------------------------------------------------

def possible_directed_edges(n):
    if n <= 1:
        return 1
    return n * (n - 1)


product_summary["possible_internal_directed_edges"] = (
    product_summary["review_count"].apply(possible_directed_edges)
)

product_summary["internal_edge_density"] = (
    product_summary["internal_total_edges"]
    / product_summary["possible_internal_directed_edges"]
)

product_summary["new_user_burst_edge_density"] = (
    product_summary["internal_new_user_burst_7d_edges"]
    / product_summary["possible_internal_directed_edges"]
)

product_summary["text_similarity_edge_density"] = (
    product_summary["internal_text_similarity_edges"]
    / product_summary["possible_internal_directed_edges"]
)


# ------------------------------------------------------------
# 11. representative score 계산
# ------------------------------------------------------------

eligible_df = product_summary[
    product_summary["review_count"] >= MIN_REVIEW_COUNT
].copy()

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

eligible_df.to_csv(
    OUTPUT_PRODUCT_SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("[SAVED] product summary:", OUTPUT_PRODUCT_SUMMARY_PATH)


# ------------------------------------------------------------
# 12. high confidence review 저장
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

high_conf_reviews.to_csv(
    OUTPUT_HIGH_CONF_REVIEWS_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("[SAVED] high confidence reviews:", OUTPUT_HIGH_CONF_REVIEWS_PATH)


# ------------------------------------------------------------
# 13. dashboard용 nodes / edges 저장
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

dashboard_nodes.to_csv(
    OUTPUT_DASHBOARD_NODES_PATH,
    index=False,
    encoding="utf-8-sig"
)

top_node_set = set(dashboard_nodes["node_idx"].tolist())

dashboard_edges = combined_edge_df[
    combined_edge_df["src"].isin(top_node_set)
    & combined_edge_df["dst"].isin(top_node_set)
].copy()

dashboard_edges.to_csv(
    OUTPUT_DASHBOARD_EDGES_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("[SAVED] dashboard nodes:", OUTPUT_DASHBOARD_NODES_PATH)
print("[SAVED] dashboard edges:", OUTPUT_DASHBOARD_EDGES_PATH)


# ------------------------------------------------------------
# 14. markdown summary 저장
# ------------------------------------------------------------

selected_config = final_result.get("selected_config", {})
test_result = final_result.get("test_result", {})
valid_result = final_result.get("valid_result", {})

md_lines = []

md_lines.append("# Model3 Final Representative Case Study")
md_lines.append("")
md_lines.append("## Final Model")
md_lines.append("")
md_lines.append("- Model name: `model3_final`")
md_lines.append("- Internal experiment name: `" + str(final_result.get("internal_experiment_name", "")) + "`")
md_lines.append("- Selected by: validation PR-AUC")
md_lines.append("- SIM_THRESHOLD: " + str(selected_config.get("sim_threshold", "")))
md_lines.append("- TOP_K: " + str(selected_config.get("top_k", "")))
md_lines.append("- MIN_TEXT_LENGTH: " + str(selected_config.get("min_text_length", "")))
md_lines.append("- TEXT_SIMILARITY directed edges: " + str(selected_config.get("text_edge_count", "")))
md_lines.append("")
md_lines.append("## Final Test Performance")
md_lines.append("")
md_lines.append("- Test PR-AUC: " + str(round(float(test_result.get("pr_auc", 0)), 4)))
md_lines.append("- Test Macro F1: " + str(round(float(test_result.get("macro_f1", 0)), 4)))
md_lines.append("- Test Accuracy: " + str(round(float(test_result.get("accuracy", 0)), 4)))
md_lines.append("- Threshold: " + str(round(float(test_result.get("threshold", 0)), 2)))
md_lines.append("")
md_lines.append("## Selection Rule")
md_lines.append("")
md_lines.append("- Minimum review count: " + str(MIN_REVIEW_COUNT))
md_lines.append("- High confidence fake threshold: " + str(HIGH_CONF_THRESHOLD))
md_lines.append("- Representative score combines prediction signal, New User Burst structure, and TEXT_SIMILARITY structure.")
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
md_lines.append("model3_final does not assume that every suspicious product must have strong text similarity edges.")
md_lines.append("Instead, it interprets fake review suspicion through two complementary patterns.")
md_lines.append("")
md_lines.append("1. Behavioral burst pattern: new or low activity users concentrate on the same product within a short period.")
md_lines.append("2. Semantic similarity pattern: different users write semantically similar reviews for the same product.")
md_lines.append("")
md_lines.append("The final case study is based on the selected grid search configuration, not the earlier top_k=3 prototype.")

md_text = "\n".join(md_lines)

with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
    f.write(md_text)

print("[SAVED] markdown summary:", OUTPUT_MD_PATH)


# ------------------------------------------------------------
# 15. summary json 저장
# ------------------------------------------------------------

summary = {
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "model3_final",
    "internal_experiment_name": final_result.get("internal_experiment_name", ""),
    "selected_config": selected_config,
    "test_result": test_result,
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
        "markdown_summary": str(OUTPUT_MD_PATH)
    }
}

with open(OUTPUT_SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("[SAVED] summary json:", OUTPUT_SUMMARY_JSON_PATH)


# ------------------------------------------------------------
# 16. 최종 출력
# ------------------------------------------------------------

print("")
print("[TOP CASES]")
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

print("")
print("[DONE] model3_final case study completed.")