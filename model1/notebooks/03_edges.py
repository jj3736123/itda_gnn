import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 0. 기본 설정
# ============================================================

# 현재 파일 위치:
# ITDA/model1/notebooks/03_edges.py
#
# parents[0] = notebooks
# parents[1] = model1
# parents[2] = ITDA

BASE_DIR = Path(__file__).resolve().parents[2]

NODE_FEATURE_PATH = BASE_DIR / "model1" / "data" / "model1_node_features.csv"

MODEL_DIR = BASE_DIR / "model1"
MODEL_DATA_DIR = MODEL_DIR / "data"
MODEL_CONFIG_DIR = MODEL_DIR / "configs"
MODEL_OUTPUT_DIR = MODEL_DIR / "outputs"

MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EDGE_LIST_PATH = MODEL_DATA_DIR / "model1_edge_list.csv"
EDGE_INDEX_PATH = MODEL_DATA_DIR / "model1_edge_index.npy"
EDGE_TYPE_PATH = MODEL_DATA_DIR / "model1_edge_type.npy"
EDGE_CONFIG_PATH = MODEL_CONFIG_DIR / "model1_edge_config.json"
EDGE_SUMMARY_PATH = MODEL_OUTPUT_DIR / "edge_summary.json"

TIME_WINDOW_DAYS = 7

EDGE_TYPE_MAP = {
    "RUR": 0,
    "PRODUCT_TIME_7D": 1,
    "NEW_USER_BURST_7D": 2
}


# ============================================================
# 1. 데이터 불러오기
# ============================================================

df = pd.read_csv(NODE_FEATURE_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])

if "node_id" not in df.columns:
    df = df.reset_index(drop=True)
    df["node_id"] = df.index

df["node_id"] = df["node_id"].astype(int)

print("node feature 데이터:", df.shape)
print("node_id 최소/최대:", df["node_id"].min(), df["node_id"].max())


# ============================================================
# 2. edge 저장용 함수
# ============================================================

edges = []


def add_undirected_edge(src, dst, edge_name):
    """
    GNN에서 보통 message passing이 양방향으로 일어나도록
    src -> dst, dst -> src를 모두 추가한다.
    자기 자신으로 가는 self-loop는 여기서 만들지 않는다.
    """

    if src == dst:
        return

    edge_type = EDGE_TYPE_MAP[edge_name]

    edges.append((src, dst, edge_type, edge_name))
    edges.append((dst, src, edge_type, edge_name))


# ============================================================
# 3. Relation 0: RUR edge
# ============================================================
# 같은 user_id가 작성한 review끼리 연결한다.
# 기본 relation으로 사용한다.

rur_pair_count = 0

for user_id, group in df.groupby("user_id"):
    node_list = group["node_id"].to_list()

    if len(node_list) < 2:
        continue

    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            add_undirected_edge(node_list[i], node_list[j], "RUR")
            rur_pair_count += 1

print("RUR pair 수:", rur_pair_count)
print("RUR directed edge 수:", rur_pair_count * 2)


# ============================================================
# 4. Relation 1: Product-Time 7일 edge
# ============================================================
# 같은 product에 대해 7일 이내 작성된 review끼리 연결한다.

product_time_pair_count = 0
time_window = pd.Timedelta(days=TIME_WINDOW_DAYS)

for prod_id, group in df.groupby("prod_id"):
    group = group.sort_values("date").reset_index(drop=True)

    node_list = group["node_id"].to_list()
    date_list = group["date"].to_list()

    left = 0

    for right in range(len(group)):
        while date_list[right] - date_list[left] > time_window:
            left += 1

        # left ~ right - 1 까지는 right와 7일 이내에 있는 이전 리뷰들
        for k in range(left, right):
            add_undirected_edge(node_list[k], node_list[right], "PRODUCT_TIME_7D")
            product_time_pair_count += 1

print("Product-Time 7d pair 수:", product_time_pair_count)
print("Product-Time 7d directed edge 수:", product_time_pair_count * 2)


# ============================================================
# 5. Relation 2: New User Burst 7일 edge
# ============================================================
# 같은 product + 7일 이내 + 두 리뷰 모두 신규/저활동 user인 경우 연결한다.
#
# 신규/저활동 user 기준:
# user_review_count_global <= 2
# user_age_days_at_review <= 30

required_cols = ["user_review_count_global", "user_age_days_at_review"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(col + " 컬럼이 없다. 02_node_features.py 결과를 다시 확인해야 한다.")

df["is_new_low_activity_user"] = (
    (df["user_review_count_global"] <= 2) &
    (df["user_age_days_at_review"] <= 30)
).astype(int)

new_user_burst_pair_count = 0

df_new = df[df["is_new_low_activity_user"] == 1].copy()

for prod_id, group in df_new.groupby("prod_id"):
    group = group.sort_values("date").reset_index(drop=True)

    node_list = group["node_id"].to_list()
    date_list = group["date"].to_list()

    left = 0

    for right in range(len(group)):
        while date_list[right] - date_list[left] > time_window:
            left += 1

        for k in range(left, right):
            add_undirected_edge(node_list[k], node_list[right], "NEW_USER_BURST_7D")
            new_user_burst_pair_count += 1

print("New User Burst 7d pair 수:", new_user_burst_pair_count)
print("New User Burst 7d directed edge 수:", new_user_burst_pair_count * 2)


# ============================================================
# 6. edge dataframe 생성 및 저장
# ============================================================

edge_df = pd.DataFrame(
    edges,
    columns=["src", "dst", "edge_type", "edge_name"]
)

# 완전히 같은 edge 중복 제거
edge_df = edge_df.drop_duplicates().reset_index(drop=True)

edge_index = edge_df[["src", "dst"]].to_numpy().T
edge_type = edge_df["edge_type"].to_numpy()

np.save(EDGE_INDEX_PATH, edge_index)
np.save(EDGE_TYPE_PATH, edge_type)
edge_df.to_csv(EDGE_LIST_PATH, index=False, encoding="utf-8-sig")


# ============================================================
# 7. edge 설정 및 요약 저장
# ============================================================

edge_config = {
    "edge_type_map": EDGE_TYPE_MAP,
    "time_window_days": TIME_WINDOW_DAYS,
    "new_user_burst_condition": {
        "user_review_count_global": "<= 2",
        "user_age_days_at_review": "<= 30"
    },
    "is_undirected": True,
    "self_loop_included": False
}

with open(EDGE_CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(edge_config, f, ensure_ascii=False, indent=4)


edge_count_by_type = (
    edge_df.groupby(["edge_type", "edge_name"])
    .size()
    .reset_index(name="directed_edge_count")
)

summary = {
    "num_nodes": int(df["node_id"].nunique()),
    "total_directed_edges": int(len(edge_df)),
    "total_undirected_pairs_before_dedup": int(
        rur_pair_count + product_time_pair_count + new_user_burst_pair_count
    ),
    "rur_pair_count": int(rur_pair_count),
    "product_time_7d_pair_count": int(product_time_pair_count),
    "new_user_burst_7d_pair_count": int(new_user_burst_pair_count),
    "new_low_activity_review_count": int(df["is_new_low_activity_user"].sum()),
    "edge_count_by_type": edge_count_by_type.to_dict(orient="records"),
    "saved_edge_list_path": str(EDGE_LIST_PATH),
    "saved_edge_index_path": str(EDGE_INDEX_PATH),
    "saved_edge_type_path": str(EDGE_TYPE_PATH),
    "saved_edge_config_path": str(EDGE_CONFIG_PATH)
}

with open(EDGE_SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=4)


# ============================================================
# 8. 결과 확인
# ============================================================

print("\n==============================")
print("edge 생성 완료")
print("==============================")
print("총 directed edge 수:", len(edge_df))
print("\nedge type별 directed edge 수:")
print(edge_count_by_type)

print("\n신규/저활동 user review 수:", df["is_new_low_activity_user"].sum())

print("\nedge_index shape:", edge_index.shape)
print("edge_type shape:", edge_type.shape)

print("\n저장 완료:")
print("edge list:", EDGE_LIST_PATH)
print("edge index:", EDGE_INDEX_PATH)
print("edge type:", EDGE_TYPE_PATH)
print("edge config:", EDGE_CONFIG_PATH)
print("edge summary:", EDGE_SUMMARY_PATH)