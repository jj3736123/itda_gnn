import json
from pathlib import Path

import pandas as pd


# ============================================================
# 0. 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "model1"
REPORT_DIR = MODEL_DIR / "report_assets"
CASE_DIR = REPORT_DIR / "case_studies"

CASE_DIR.mkdir(parents=True, exist_ok=True)

ALL_PRED_PATH = REPORT_DIR / "final_rgcn_all_predictions.csv"
PRODUCT_SUMMARY_PATH = REPORT_DIR / "product_suspicious_summary.csv"
EDGE_LIST_PATH = MODEL_DIR / "data" / "model1_edge_list.csv"

CASE_OVERVIEW_PATH = CASE_DIR / "case_study_overview.csv"
CASE_SUMMARY_JSON_PATH = CASE_DIR / "case_study_summary.json"
TOP_CASE_NODES_PATH = CASE_DIR / "top_case_nodes.csv"
TOP_CASE_EDGES_PATH = CASE_DIR / "top_case_edges.csv"

TOP_K_PRODUCTS = 3


# ============================================================
# 1. 데이터 불러오기
# ============================================================

df_pred = pd.read_csv(ALL_PRED_PATH)
product_summary = pd.read_csv(PRODUCT_SUMMARY_PATH)
edge_df = pd.read_csv(EDGE_LIST_PATH)

df_pred["date"] = pd.to_datetime(df_pred["date"], errors="coerce")

print("prediction 데이터:", df_pred.shape)
print("product summary:", product_summary.shape)
print("edge list:", edge_df.shape)


# ============================================================
# 2. 상위 suspicious product 선택
# ============================================================

top_products = product_summary.head(TOP_K_PRODUCTS)["prod_id"].tolist()

print("\n선택된 상위 product:")
print(top_products)


# ============================================================
# 3. product별 case study 파일 생성
# ============================================================

case_rows = []
all_case_nodes = []
all_case_edges = []

for prod_id in top_products:
    prod_nodes = df_pred[df_pred["prod_id"] == prod_id].copy()
    prod_nodes = prod_nodes.sort_values(
        ["prob_fake", "date"],
        ascending=[False, True]
    ).reset_index(drop=True)

    node_set = set(prod_nodes["node_id"].tolist())

    prod_edges = edge_df[
        edge_df["src"].isin(node_set) &
        edge_df["dst"].isin(node_set)
    ].copy()

    # product별 edge type count
    edge_type_counts = (
        prod_edges.groupby("edge_name")
        .size()
        .reset_index(name="directed_edge_count")
        .sort_values("directed_edge_count", ascending=False)
    )

    # rating 분포
    rating_counts = (
        prod_nodes["rating"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # 예측 fake 상위 리뷰 10개
    top_reviews = prod_nodes.head(10)[
        [
            "node_id",
            "review_id",
            "user_id",
            "prod_id",
            "date",
            "rating",
            "label",
            "pred_label",
            "prob_fake",
            "is_new_low_activity_user",
            "user_review_count_global",
            "user_age_days_at_review",
            "same_rating_prod_7d_count",
            "text_short",
        ]
    ].copy()

    # 파일 저장
    prod_nodes_path = CASE_DIR / ("product_" + str(prod_id) + "_nodes.csv")
    prod_edges_path = CASE_DIR / ("product_" + str(prod_id) + "_edges.csv")
    prod_top_reviews_path = CASE_DIR / ("product_" + str(prod_id) + "_top_reviews.csv")
    prod_edge_counts_path = CASE_DIR / ("product_" + str(prod_id) + "_edge_counts.csv")

    prod_nodes.to_csv(prod_nodes_path, index=False, encoding="utf-8-sig")
    prod_edges.to_csv(prod_edges_path, index=False, encoding="utf-8-sig")
    top_reviews.to_csv(prod_top_reviews_path, index=False, encoding="utf-8-sig")
    edge_type_counts.to_csv(prod_edge_counts_path, index=False, encoding="utf-8-sig")

    # overview row
    row = {
        "prod_id": int(prod_id),
        "review_count": int(len(prod_nodes)),
        "unique_user_count": int(prod_nodes["user_id"].nunique()),
        "date_min": str(prod_nodes["date"].min()),
        "date_max": str(prod_nodes["date"].max()),
        "avg_prob_fake": float(prod_nodes["prob_fake"].mean()),
        "max_prob_fake": float(prod_nodes["prob_fake"].max()),
        "pred_fake_count": int((prod_nodes["pred_label"] == 1).sum()),
        "pred_fake_ratio": float((prod_nodes["pred_label"] == 1).mean()),
        "true_fake_count": int((prod_nodes["label"] == 1).sum()),
        "true_fake_ratio": float((prod_nodes["label"] == 1).mean()),
        "new_low_activity_count": int(prod_nodes["is_new_low_activity_user"].sum()),
        "new_low_activity_ratio": float(prod_nodes["is_new_low_activity_user"].mean()),
        "same_rating_prod_7d_mean": float(prod_nodes["same_rating_prod_7d_count"].mean()),
        "directed_edge_count_inside_product": int(len(prod_edges)),
        "rating_counts": rating_counts,
        "nodes_file": str(prod_nodes_path),
        "edges_file": str(prod_edges_path),
        "top_reviews_file": str(prod_top_reviews_path),
        "edge_counts_file": str(prod_edge_counts_path),
    }

    case_rows.append(row)

    all_case_nodes.append(prod_nodes)
    all_case_edges.append(prod_edges)

    print("\n==============================")
    print("product", prod_id, "case 생성 완료")
    print("==============================")
    print("review 수:", row["review_count"])
    print("unique user 수:", row["unique_user_count"])
    print("기간:", row["date_min"], "~", row["date_max"])
    print("avg prob fake:", round(row["avg_prob_fake"], 4))
    print("pred fake ratio:", round(row["pred_fake_ratio"], 4))
    print("true fake ratio:", round(row["true_fake_ratio"], 4))
    print("new low activity ratio:", round(row["new_low_activity_ratio"], 4))
    print("내부 directed edge 수:", row["directed_edge_count_inside_product"])
    print("\nedge type count:")
    print(edge_type_counts)


# ============================================================
# 4. 전체 case study 파일 저장
# ============================================================

case_overview = pd.DataFrame(case_rows)
case_overview.to_csv(CASE_OVERVIEW_PATH, index=False, encoding="utf-8-sig")

if len(all_case_nodes) > 0:
    top_case_nodes = pd.concat(all_case_nodes, axis=0).reset_index(drop=True)
else:
    top_case_nodes = pd.DataFrame()

if len(all_case_edges) > 0:
    top_case_edges = pd.concat(all_case_edges, axis=0).drop_duplicates().reset_index(drop=True)
else:
    top_case_edges = pd.DataFrame()

top_case_nodes.to_csv(TOP_CASE_NODES_PATH, index=False, encoding="utf-8-sig")
top_case_edges.to_csv(TOP_CASE_EDGES_PATH, index=False, encoding="utf-8-sig")

summary = {
    "top_k_products": TOP_K_PRODUCTS,
    "selected_products": [int(x) for x in top_products],
    "case_overview": case_rows,
    "saved_files": {
        "case_overview": str(CASE_OVERVIEW_PATH),
        "case_summary_json": str(CASE_SUMMARY_JSON_PATH),
        "top_case_nodes": str(TOP_CASE_NODES_PATH),
        "top_case_edges": str(TOP_CASE_EDGES_PATH),
    }
}

with open(CASE_SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=4, default=str)




print("\n\n==============================")
print("case study 전체 요약")
print("==============================")
print(case_overview)

print("\n보고서에 쓸 해석 초안:")
for row in case_rows:
    print("\nproduct", row["prod_id"])
    print(
        "해당 product는 sample 내 리뷰",
        row["review_count"],
        "개 중",
        row["pred_fake_count"],
        "개가 fake로 예측되었고, 평균 fake 확률은",
        round(row["avg_prob_fake"], 4),
        "이다."
    )
    print(
        "또한 신규/저활동 user 비율이",
        round(row["new_low_activity_ratio"], 4),
        "로 높아, 단순히 개별 리뷰 문장만 의심된 것이 아니라 특정 product에 짧은 기간 신규 계정성 리뷰가 집중된 구조가 관찰된다."
    )
    print(
        "실제 label 기준 fake 비율도",
        round(row["true_fake_ratio"], 4),
        "로 나타나 모델의 의심 판단이 어느 정도 실제 fake label과도 일치한다."
    )

print("\n저장 완료:")
print("case overview:", CASE_OVERVIEW_PATH)
print("case summary json:", CASE_SUMMARY_JSON_PATH)
print("top case nodes:", TOP_CASE_NODES_PATH)
print("top case edges:", TOP_CASE_EDGES_PATH)