from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 03_edges_text_similarity.py
# 목적:
# - model3에 TEXT_SIMILARITY edge 추가
# - 조건:
#   1. 같은 prod_id 내부에서만 비교
#   2. 서로 다른 user_id끼리만 연결
#   3. text 길이 MIN_TEXT_LENGTH 이상
#   4. SBERT cosine similarity >= SIM_THRESHOLD
#   5. 각 review당 top_k개까지만 선택
#   6. 선택된 edge는 양방향 directed edge로 저장
# ============================================================

NOTEBOOK_DIR = Path(__file__).resolve().parent
MODEL3_DIR = NOTEBOOK_DIR.parent

DATA_DIR = MODEL3_DIR / "data"
CONFIG_DIR = MODEL3_DIR / "configs"
OUTPUT_DIR = MODEL3_DIR / "outputs"

SAMPLED_REVIEWS_PATH = DATA_DIR / "model3_sampled_reviews.csv"
SOURCE_FEATURE_PATH = DATA_DIR / "model3_source_node_features_sbert384.csv"
SOURCE_FEATURE_CONFIG_PATH = CONFIG_DIR / "model3_source_feature_columns_sbert384.json"

BASE_EDGE_LIST_PATH = DATA_DIR / "model3_edge_list_base.csv"
BASE_EDGE_INDEX_PATH = DATA_DIR / "model3_edge_index_base.npy"
BASE_EDGE_TYPE_PATH = DATA_DIR / "model3_edge_type_base.npy"
BASE_EDGE_CONFIG_PATH = CONFIG_DIR / "model3_edge_config_base.json"

TEXT_EDGE_LIST_PATH = DATA_DIR / "model3_text_similarity_edge_list.csv"

OUTPUT_EDGE_LIST_PATH = DATA_DIR / "model3_edge_list.csv"
OUTPUT_EDGE_INDEX_PATH = DATA_DIR / "model3_edge_index.npy"
OUTPUT_EDGE_TYPE_PATH = DATA_DIR / "model3_edge_type.npy"
OUTPUT_EDGE_CONFIG_PATH = CONFIG_DIR / "model3_edge_config.json"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "text_similarity_edge_summary.json"

# ------------------------------------------------------------
# 실험 설정
# ------------------------------------------------------------

SIM_THRESHOLD = 0.85
TOP_K = 3
MIN_TEXT_LENGTH = 20

TEXT_EDGE_NAME = "TEXT_SIMILARITY"

# ------------------------------------------------------------
# 1. 파일 로드
# ------------------------------------------------------------

print("[LOAD] sampled reviews:", SAMPLED_REVIEWS_PATH)
sample_df = pd.read_csv(SAMPLED_REVIEWS_PATH)

print("[LOAD] source SBERT feature:", SOURCE_FEATURE_PATH)
feature_df = pd.read_csv(SOURCE_FEATURE_PATH)

print("[LOAD] base edge list:", BASE_EDGE_LIST_PATH)
base_edge_list_df = pd.read_csv(BASE_EDGE_LIST_PATH)

print("[LOAD] base edge index:", BASE_EDGE_INDEX_PATH)
base_edge_index = np.load(BASE_EDGE_INDEX_PATH)

print("[LOAD] base edge type:", BASE_EDGE_TYPE_PATH)
base_edge_type = np.load(BASE_EDGE_TYPE_PATH)

print("[INFO] sampled reviews shape:", sample_df.shape)
print("[INFO] source feature shape:", feature_df.shape)
print("[INFO] base edge list shape:", base_edge_list_df.shape)
print("[INFO] base edge index shape:", base_edge_index.shape)
print("[INFO] base edge type shape:", base_edge_type.shape)

# ------------------------------------------------------------
# 2. 기본 컬럼 확인
# ------------------------------------------------------------

required_sample_cols = ["review_id", "user_id", "prod_id"]

for col in required_sample_cols:
    if col not in sample_df.columns:
        raise ValueError("sampled reviews에 필요한 컬럼이 없음: " + col)

if "review_id" not in feature_df.columns:
    raise ValueError("source feature에 review_id 컬럼이 없음.")

if "text" in sample_df.columns:
    sample_df["text"] = sample_df["text"].fillna("").astype(str)
    sample_df["text_length_for_edge"] = sample_df["text"].str.len()
else:
    print("[WARNING] sampled reviews에 text 컬럼이 없음. MIN_TEXT_LENGTH 조건은 적용하지 않음.")
    sample_df["text_length_for_edge"] = 999999

# ------------------------------------------------------------
# 3. node index 생성
# ------------------------------------------------------------
# 기존 model1/model2와 같은 sample 순서를 유지한다고 보고,
# sample_df의 row 순서를 node index로 사용한다.

sample_df = sample_df.reset_index(drop=True)
sample_df["node_idx"] = np.arange(len(sample_df))

if sample_df["review_id"].duplicated().sum() > 0:
    raise ValueError("review_id 중복이 있음. node index mapping 불가능.")

review_to_node = {}

for idx, row in sample_df.iterrows():
    review_to_node[row["review_id"]] = int(row["node_idx"])

print("[OK] node index mapping created:", len(review_to_node))

# ------------------------------------------------------------
# 4. SBERT 컬럼 찾기
# ------------------------------------------------------------

sbert_columns = []

if SOURCE_FEATURE_CONFIG_PATH.exists():
    with open(SOURCE_FEATURE_CONFIG_PATH, "r", encoding="utf-8") as f:
        source_config = json.load(f)

    if "sbert_feature_columns" in source_config:
        for col in source_config["sbert_feature_columns"]:
            if col in feature_df.columns:
                sbert_columns.append(col)

if len(sbert_columns) == 0:
    prefix_candidates = [
        "sbert_",
        "sbert",
        "text_sbert_",
        "text_emb_",
        "emb_",
        "embedding_"
    ]

    for col in feature_df.columns:
        lower_col = col.lower()
        for prefix in prefix_candidates:
            if lower_col.startswith(prefix):
                sbert_columns.append(col)
                break

if len(sbert_columns) == 0:
    raise ValueError("SBERT column을 찾지 못함.")

print("[OK] SBERT columns:", len(sbert_columns))

# ------------------------------------------------------------
# 5. sample 정보와 SBERT feature merge
# ------------------------------------------------------------

merge_cols = ["review_id"] + sbert_columns

work_df = sample_df[
    ["review_id", "node_idx", "user_id", "prod_id", "text_length_for_edge"]
].merge(
    feature_df[merge_cols],
    on="review_id",
    how="left"
)

missing_feature_count = work_df[sbert_columns[0]].isna().sum()

if missing_feature_count > 0:
    raise ValueError("SBERT feature가 없는 review가 있음: " + str(missing_feature_count))

print("[OK] merged work df shape:", work_df.shape)

# ------------------------------------------------------------
# 6. TEXT_SIMILARITY edge 생성
# ------------------------------------------------------------

text_edge_pair_dict = {}
product_summaries = []

prod_ids = work_df["prod_id"].unique()
total_products = len(prod_ids)

print("[START] building TEXT_SIMILARITY edges")
print("[INFO] number of products:", total_products)
print("[INFO] SIM_THRESHOLD:", SIM_THRESHOLD)
print("[INFO] TOP_K:", TOP_K)
print("[INFO] MIN_TEXT_LENGTH:", MIN_TEXT_LENGTH)

processed_product_count = 0

for prod_id, group in work_df.groupby("prod_id"):
    processed_product_count += 1

    if processed_product_count % 200 == 0:
        print("[PROGRESS]", processed_product_count, "/", total_products)

    group = group[group["text_length_for_edge"] >= MIN_TEXT_LENGTH].copy()
    group = group.reset_index(drop=True)

    n = len(group)

    if n <= 1:
        product_summaries.append({
            "prod_id": int(prod_id),
            "candidate_reviews": int(n),
            "created_directed_edges": 0
        })
        continue

    X = group[sbert_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).values

    sim_matrix = cosine_similarity(X)

    created_for_product = 0

    user_values = group["user_id"].values
    node_values = group["node_idx"].values
    review_values = group["review_id"].values

    for i in range(n):
        similarities = sim_matrix[i].copy()

        # 자기 자신 제외
        similarities[i] = -1.0

        # 같은 user 제외
        same_user_mask = user_values == user_values[i]
        similarities[same_user_mask] = -1.0

        # threshold 미만 제외
        valid_indices = np.where(similarities >= SIM_THRESHOLD)[0]

        if len(valid_indices) == 0:
            continue

        # similarity 높은 순서대로 정렬
        valid_scores = similarities[valid_indices]
        sorted_order = np.argsort(-valid_scores)
        sorted_indices = valid_indices[sorted_order]

        selected_indices = sorted_indices[:TOP_K]

        src_node = int(node_values[i])
        src_review_id = review_values[i]

        for j in selected_indices:
            dst_node = int(node_values[j])
            dst_review_id = review_values[j]
            score = float(similarities[j])

            if src_node == dst_node:
                continue

            # 양방향 directed edge 저장
            key1 = (src_node, dst_node)
            key2 = (dst_node, src_node)

            if key1 not in text_edge_pair_dict:
                text_edge_pair_dict[key1] = {
                    "src": src_node,
                    "dst": dst_node,
                    "src_review_id": src_review_id,
                    "dst_review_id": dst_review_id,
                    "src_user_id": int(user_values[i]),
                    "dst_user_id": int(user_values[j]),
                    "prod_id": int(prod_id),
                    "similarity": score,
                    "edge_name": TEXT_EDGE_NAME
                }
                created_for_product += 1

            if key2 not in text_edge_pair_dict:
                text_edge_pair_dict[key2] = {
                    "src": dst_node,
                    "dst": src_node,
                    "src_review_id": dst_review_id,
                    "dst_review_id": src_review_id,
                    "src_user_id": int(user_values[j]),
                    "dst_user_id": int(user_values[i]),
                    "prod_id": int(prod_id),
                    "similarity": score,
                    "edge_name": TEXT_EDGE_NAME
                }
                created_for_product += 1

    product_summaries.append({
        "prod_id": int(prod_id),
        "candidate_reviews": int(n),
        "created_directed_edges": int(created_for_product)
    })

print("[DONE] text edge creation loop completed")

text_edge_records = list(text_edge_pair_dict.values())
text_edge_df = pd.DataFrame(text_edge_records)

if len(text_edge_df) == 0:
    print("[WARNING] TEXT_SIMILARITY edge가 0개 생성됨. threshold를 낮춰야 할 수 있음.")
    text_edge_df = pd.DataFrame(columns=[
        "src", "dst", "src_review_id", "dst_review_id",
        "src_user_id", "dst_user_id", "prod_id",
        "similarity", "edge_name"
    ])

print("[INFO] TEXT_SIMILARITY directed edges:", len(text_edge_df))

text_edge_df.to_csv(TEXT_EDGE_LIST_PATH, index=False, encoding="utf-8-sig")
print("[SAVED] text edge list:", TEXT_EDGE_LIST_PATH)

# ------------------------------------------------------------
# 7. base edge index shape 정리
# ------------------------------------------------------------

if len(base_edge_index.shape) != 2:
    raise ValueError("base_edge_index는 2차원이어야 함.")

if base_edge_index.shape[0] == 2:
    base_edge_index_fixed = base_edge_index
elif base_edge_index.shape[1] == 2:
    base_edge_index_fixed = base_edge_index.T
else:
    raise ValueError("base_edge_index shape를 해석할 수 없음: " + str(base_edge_index.shape))

base_edge_type = base_edge_type.reshape(-1)

if base_edge_index_fixed.shape[1] != len(base_edge_type):
    raise ValueError("base_edge_index edge 수와 base_edge_type 길이가 다름.")

print("[OK] base edge index fixed shape:", base_edge_index_fixed.shape)

# ------------------------------------------------------------
# 8. TEXT_SIMILARITY edge type id 설정
# ------------------------------------------------------------

base_max_edge_type = int(np.max(base_edge_type))
text_edge_type_id = base_max_edge_type + 1

print("[INFO] base max edge type:", base_max_edge_type)
print("[INFO] TEXT_SIMILARITY edge type id:", text_edge_type_id)

if len(text_edge_df) > 0:
    text_edge_index = text_edge_df[["src", "dst"]].values.T.astype(np.int64)
    text_edge_type = np.full(
        shape=(text_edge_index.shape[1],),
        fill_value=text_edge_type_id,
        dtype=np.int64
    )
else:
    text_edge_index = np.empty((2, 0), dtype=np.int64)
    text_edge_type = np.empty((0,), dtype=np.int64)

print("[INFO] text edge index shape:", text_edge_index.shape)
print("[INFO] text edge type shape:", text_edge_type.shape)

# ------------------------------------------------------------
# 9. base edge + text edge 결합
# ------------------------------------------------------------

combined_edge_index = np.concatenate(
    [base_edge_index_fixed.astype(np.int64), text_edge_index],
    axis=1
)

combined_edge_type = np.concatenate(
    [base_edge_type.astype(np.int64), text_edge_type],
    axis=0
)

print("[INFO] combined edge index shape:", combined_edge_index.shape)
print("[INFO] combined edge type shape:", combined_edge_type.shape)

np.save(OUTPUT_EDGE_INDEX_PATH, combined_edge_index)
np.save(OUTPUT_EDGE_TYPE_PATH, combined_edge_type)

print("[SAVED] combined edge index:", OUTPUT_EDGE_INDEX_PATH)
print("[SAVED] combined edge type:", OUTPUT_EDGE_TYPE_PATH)

# ------------------------------------------------------------
# 10. combined edge list 저장
# ------------------------------------------------------------

# base edge list는 컬럼명이 프로젝트마다 다를 수 있으므로,
# 최소한 src, dst, edge_name 형태로 맞춘다.

base_for_save = base_edge_list_df.copy()

if "edge_name" not in base_for_save.columns:
    if "relation" in base_for_save.columns:
        base_for_save["edge_name"] = base_for_save["relation"]
    elif "edge_type_name" in base_for_save.columns:
        base_for_save["edge_name"] = base_for_save["edge_type_name"]
    else:
        base_for_save["edge_name"] = "BASE_EDGE"

if "similarity" not in base_for_save.columns:
    base_for_save["similarity"] = np.nan

text_for_save = text_edge_df.copy()
text_for_save["edge_type"] = text_edge_type_id

combined_edge_list_df = pd.concat(
    [base_for_save, text_for_save],
    axis=0,
    ignore_index=True
)

combined_edge_list_df.to_csv(OUTPUT_EDGE_LIST_PATH, index=False, encoding="utf-8-sig")

print("[SAVED] combined edge list:", OUTPUT_EDGE_LIST_PATH)
print("[INFO] combined edge list shape:", combined_edge_list_df.shape)

# ------------------------------------------------------------
# 11. edge config 저장
# ------------------------------------------------------------

edge_config = {}

if BASE_EDGE_CONFIG_PATH.exists():
    with open(BASE_EDGE_CONFIG_PATH, "r", encoding="utf-8") as f:
        edge_config = json.load(f)

edge_config["model"] = "model3"
edge_config["new_relation"] = {
    "edge_type_id": text_edge_type_id,
    "edge_name": TEXT_EDGE_NAME,
    "description": "Connect reviews within the same product when their SBERT semantic similarity is high.",
    "conditions": {
        "same_prod_id": True,
        "different_user_id": True,
        "min_text_length": MIN_TEXT_LENGTH,
        "cosine_similarity_threshold": SIM_THRESHOLD,
        "top_k_per_review": TOP_K,
        "bidirectional_directed_edges": True,
        "label_used": False
    }
}

edge_config["model3_edge_type_mapping_note"] = {
    "base_edge_type_max": base_max_edge_type,
    "text_similarity_edge_type_id": text_edge_type_id
}

edge_config["model3_edge_counts"] = {
    "base_directed_edges": int(base_edge_index_fixed.shape[1]),
    "text_similarity_directed_edges": int(text_edge_index.shape[1]),
    "combined_directed_edges": int(combined_edge_index.shape[1])
}

with open(OUTPUT_EDGE_CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(edge_config, f, ensure_ascii=False, indent=2)

print("[SAVED] edge config:", OUTPUT_EDGE_CONFIG_PATH)

# ------------------------------------------------------------
# 12. summary 저장
# ------------------------------------------------------------

product_summary_df = pd.DataFrame(product_summaries)

products_with_text_edges = 0
max_edges_product = None

if len(product_summary_df) > 0:
    products_with_text_edges = int((product_summary_df["created_directed_edges"] > 0).sum())

    if product_summary_df["created_directed_edges"].max() > 0:
        max_row = product_summary_df.sort_values(
            "created_directed_edges",
            ascending=False
        ).iloc[0]

        max_edges_product = {
            "prod_id": int(max_row["prod_id"]),
            "candidate_reviews": int(max_row["candidate_reviews"]),
            "created_directed_edges": int(max_row["created_directed_edges"])
        }

summary = {
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "model3",
    "step": "03_edges_text_similarity",
    "edge_design": {
        "edge_name": TEXT_EDGE_NAME,
        "same_prod_id": True,
        "different_user_id": True,
        "min_text_length": MIN_TEXT_LENGTH,
        "similarity_threshold": SIM_THRESHOLD,
        "top_k": TOP_K,
        "bidirectional_directed_edges": True,
        "label_used": False
    },
    "counts": {
        "num_nodes": int(len(sample_df)),
        "num_products": int(total_products),
        "base_directed_edges": int(base_edge_index_fixed.shape[1]),
        "text_similarity_directed_edges": int(text_edge_index.shape[1]),
        "combined_directed_edges": int(combined_edge_index.shape[1]),
        "products_with_text_similarity_edges": products_with_text_edges
    },
    "sbert": {
        "source": "model2 SBERT 384 feature",
        "sbert_columns": int(len(sbert_columns))
    },
    "max_edges_product": max_edges_product,
    "outputs": {
        "text_edge_list": str(TEXT_EDGE_LIST_PATH),
        "combined_edge_list": str(OUTPUT_EDGE_LIST_PATH),
        "combined_edge_index": str(OUTPUT_EDGE_INDEX_PATH),
        "combined_edge_type": str(OUTPUT_EDGE_TYPE_PATH),
        "edge_config": str(OUTPUT_EDGE_CONFIG_PATH)
    }
}

with open(OUTPUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("[SAVED] summary:", OUTPUT_SUMMARY_PATH)

print("\n[DONE] model3 TEXT_SIMILARITY edge generation completed.")
print("다음 단계: 04_prepare_data.py")