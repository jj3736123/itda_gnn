import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 0. 기본 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "model1"
DATA_DIR = MODEL_DIR / "data"
OUTPUT_DIR = MODEL_DIR / "outputs"
REPORT_DIR = MODEL_DIR / "report_assets"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TENSOR_DATA_PATH = DATA_DIR / "model1_tensor_data.pt"
NODE_FEATURE_PATH = DATA_DIR / "model1_node_features.csv"
EDGE_LIST_PATH = DATA_DIR / "model1_edge_list.csv"
SPLIT_PATH = DATA_DIR / "model1_split.csv"

BEST_MODEL_PATH = DATA_DIR / "model1_rgcn_best.pt"
RGCN_RESULT_PATH = OUTPUT_DIR / "rgcn_result.json"

ALL_PRED_PATH = REPORT_DIR / "final_rgcn_all_predictions.csv"
HIGH_FAKE_PATH = REPORT_DIR / "high_confidence_fake_reviews.csv"
PRODUCT_SUMMARY_PATH = REPORT_DIR / "product_suspicious_summary.csv"
DASHBOARD_NODE_PATH = REPORT_DIR / "dashboard_nodes.csv"
DASHBOARD_EDGE_PATH = REPORT_DIR / "dashboard_edges.csv"
SUMMARY_JSON_PATH = REPORT_DIR / "final_interpretation_summary.json"


# ============================================================
# 1. RGCN 모델 구조
# ============================================================

class RelationalGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super().__init__()

        self.num_relations = num_relations
        self.out_channels = out_channels

        self.self_linear = nn.Linear(in_channels, out_channels)

        self.relation_linears = nn.ModuleList()
        for _ in range(num_relations):
            self.relation_linears.append(
                nn.Linear(in_channels, out_channels, bias=False)
            )

    def forward(self, x, edge_index, edge_type):
        src_all = edge_index[0]
        dst_all = edge_index[1]

        out = self.self_linear(x)

        for rel in range(self.num_relations):
            mask = edge_type == rel

            if mask.sum().item() == 0:
                continue

            src = src_all[mask]
            dst = dst_all[mask]

            message = self.relation_linears[rel](x[src])

            agg = torch.zeros(
                x.size(0),
                self.out_channels,
                device=x.device,
                dtype=message.dtype
            )

            agg.index_add_(0, dst, message)

            degree = torch.zeros(
                x.size(0),
                1,
                device=x.device,
                dtype=message.dtype
            )

            ones = torch.ones(
                dst.size(0),
                1,
                device=x.device,
                dtype=message.dtype
            )

            degree.index_add_(0, dst, ones)
            agg = agg / degree.clamp(min=1.0)

            out = out + agg

        return out


class SimpleRGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_relations, dropout):
        super().__init__()

        self.conv1 = RelationalGraphConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            num_relations=num_relations
        )

        self.conv2 = RelationalGraphConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_relations=num_relations
        )

        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.classifier(x).view(-1)
        return logits


# ============================================================
# 2. 데이터 불러오기
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("사용 device:", device)

try:
    tensor_data = torch.load(TENSOR_DATA_PATH, map_location="cpu", weights_only=False)
except TypeError:
    tensor_data = torch.load(TENSOR_DATA_PATH, map_location="cpu")

try:
    checkpoint = torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(BEST_MODEL_PATH, map_location="cpu")

with open(RGCN_RESULT_PATH, "r", encoding="utf-8") as f:
    rgcn_result = json.load(f)

df_node = pd.read_csv(NODE_FEATURE_PATH)
df_split = pd.read_csv(SPLIT_PATH)
edge_df = pd.read_csv(EDGE_LIST_PATH)

df_node["date"] = pd.to_datetime(df_node["date"], errors="coerce")

x = tensor_data["x"].to(device)
y = tensor_data["y"].to(device)
edge_index = tensor_data["edge_index"].to(device)
edge_type = tensor_data["edge_type"].to(device)

num_nodes = x.size(0)
in_channels = x.size(1)
num_relations = int(edge_type.max().item()) + 1

config = checkpoint["config"]
hidden_channels = config["hidden_channels"]
dropout = config["dropout"]

threshold = rgcn_result["best_validation_threshold"]

print("node 수:", num_nodes)
print("feature 수:", in_channels)
print("edge 수:", edge_index.size(1))
print("relation 수:", num_relations)
print("threshold:", threshold)


# ============================================================
# 3. best RGCN으로 전체 node 예측
# ============================================================

model = SimpleRGCN(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    num_relations=num_relations,
    dropout=dropout
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    logits = model(x, edge_index, edge_type)
    prob_fake = torch.sigmoid(logits).detach().cpu().numpy()

pred_label = (prob_fake >= threshold).astype(int)

df_node = df_node.sort_values("node_id").reset_index(drop=True)
df_split = df_split.sort_values("node_id").reset_index(drop=True)

df_pred = df_node.copy()

df_pred["split"] = df_split["split"]
df_pred["prob_fake"] = prob_fake
df_pred["pred_label"] = pred_label
df_pred["correct"] = (df_pred["label"] == df_pred["pred_label"]).astype(int)

df_pred["is_new_low_activity_user"] = (
    (df_pred["user_review_count_global"] <= 2) &
    (df_pred["user_age_days_at_review"] <= 30)
).astype(int)

df_pred["text_short"] = (
    df_pred["text"]
    .astype(str)
    .str.replace("\n", " ", regex=False)
    .str.replace("\r", " ", regex=False)
    .str.slice(0, 200)
)

df_pred.to_csv(ALL_PRED_PATH, index=False, encoding="utf-8-sig")


# ============================================================
# 4. high confidence fake review 추출
# ============================================================

high_fake = df_pred[df_pred["pred_label"] == 1].copy()
high_fake = high_fake.sort_values("prob_fake", ascending=False)

high_fake_cols = [
    "node_id",
    "review_id",
    "user_id",
    "prod_id",
    "date",
    "rating",
    "label",
    "split",
    "prob_fake",
    "pred_label",
    "correct",
    "is_new_low_activity_user",
    "user_review_count_global",
    "user_age_days_at_review",
    "same_rating_prod_7d_count",
    "text_short",
]

high_fake[high_fake_cols].head(300).to_csv(
    HIGH_FAKE_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 5. product 단위 suspicious summary 생성
# ============================================================

product_summary = (
    df_pred.groupby("prod_id")
    .agg(
        sample_review_count=("node_id", "count"),
        unique_user_count=("user_id", "nunique"),
        avg_prob_fake=("prob_fake", "mean"),
        max_prob_fake=("prob_fake", "max"),
        pred_fake_count=("pred_label", "sum"),
        true_fake_count=("label", "sum"),
        new_low_activity_count=("is_new_low_activity_user", "sum"),
        same_rating_7d_mean=("same_rating_prod_7d_count", "mean"),
        date_min=("date", "min"),
        date_max=("date", "max"),
    )
    .reset_index()
)

product_summary["pred_fake_ratio"] = (
    product_summary["pred_fake_count"] / product_summary["sample_review_count"]
)

product_summary["true_fake_ratio"] = (
    product_summary["true_fake_count"] / product_summary["sample_review_count"]
)

product_summary["new_low_activity_ratio"] = (
    product_summary["new_low_activity_count"] / product_summary["sample_review_count"]
)

# 너무 작은 product가 상위에 뜨는 것을 막기 위해 log 가중치 사용
product_summary["size_weight"] = np.log1p(product_summary["sample_review_count"])

product_summary["suspicious_score"] = (
    product_summary["avg_prob_fake"] * 0.45
    + product_summary["pred_fake_ratio"] * 0.30
    + product_summary["new_low_activity_ratio"] * 0.15
    + (
        product_summary["same_rating_7d_mean"]
        / product_summary["same_rating_7d_mean"].max()
    ).fillna(0) * 0.10
) * product_summary["size_weight"]

product_summary = product_summary.sort_values(
    ["suspicious_score", "sample_review_count"],
    ascending=False
).reset_index(drop=True)

product_summary.to_csv(
    PRODUCT_SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 6. 대시보드 / Gephi용 node 파일 생성
# ============================================================

dashboard_nodes = df_pred[
    [
        "node_id",
        "review_id",
        "user_id",
        "prod_id",
        "date",
        "rating",
        "label",
        "split",
        "prob_fake",
        "pred_label",
        "correct",
        "is_new_low_activity_user",
        "same_rating_prod_7d_count",
        "text_short",
    ]
].copy()

dashboard_nodes["node_label"] = dashboard_nodes["node_id"].astype(str)
dashboard_nodes["node_color_group"] = dashboard_nodes["pred_label"].map({
    0: "pred_normal",
    1: "pred_fake"
})

dashboard_nodes.to_csv(
    DASHBOARD_NODE_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 7. 대시보드 / Gephi용 edge 파일 생성
# ============================================================
# 전체 edge가 너무 많으므로,
# 의심 node끼리 연결된 edge를 우선 추출한다.
# 의심 node 기준:
# pred_label == 1 또는 prob_fake >= 0.70

suspicious_node_set = set(
    df_pred[
        (df_pred["pred_label"] == 1) |
        (df_pred["prob_fake"] >= 0.70)
    ]["node_id"].tolist()
)

dashboard_edges = edge_df[
    edge_df["src"].isin(suspicious_node_set) &
    edge_df["dst"].isin(suspicious_node_set)
].copy()

# Gephi에서 보기 쉽게 컬럼명 추가
dashboard_edges = dashboard_edges.rename(columns={
    "src": "Source",
    "dst": "Target",
    "edge_name": "relation"
})

dashboard_edges["Type"] = "Directed"
dashboard_edges["Weight"] = 1

# 너무 클 경우 상위 50000개만 저장
if len(dashboard_edges) > 50000:
    dashboard_edges = dashboard_edges.sample(
        n=50000,
        random_state=42
    ).reset_index(drop=True)

dashboard_edges.to_csv(
    DASHBOARD_EDGE_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 8. 요약 저장
# ============================================================

top_products = product_summary.head(10)[
    [
        "prod_id",
        "sample_review_count",
        "avg_prob_fake",
        "pred_fake_count",
        "pred_fake_ratio",
        "true_fake_count",
        "true_fake_ratio",
        "new_low_activity_count",
        "new_low_activity_ratio",
        "suspicious_score",
    ]
].copy()

summary = {
    "model": "RGCN_with_New_User_Burst",
    "threshold": float(threshold),
    "num_nodes": int(len(df_pred)),
    "num_edges_original": int(len(edge_df)),
    "num_dashboard_edges": int(len(dashboard_edges)),
    "pred_fake_count_all": int((df_pred["pred_label"] == 1).sum()),
    "pred_fake_ratio_all": float((df_pred["pred_label"] == 1).mean()),
    "true_fake_count_all": int((df_pred["label"] == 1).sum()),
    "true_fake_ratio_all": float((df_pred["label"] == 1).mean()),
    "high_confidence_fake_saved_count": int(min(len(high_fake), 300)),
    "top_suspicious_products": top_products.to_dict(orient="records"),
    "saved_files": {
        "all_predictions": str(ALL_PRED_PATH),
        "high_confidence_fake_reviews": str(HIGH_FAKE_PATH),
        "product_suspicious_summary": str(PRODUCT_SUMMARY_PATH),
        "dashboard_nodes": str(DASHBOARD_NODE_PATH),
        "dashboard_edges": str(DASHBOARD_EDGE_PATH),
    }
}

with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=4, default=str)


# ============================================================
# 9. 출력
# ============================================================

print("\n==============================")
print("RGCN 예측 해석 파일 생성 완료")
print("==============================")

print("전체 node 수:", len(df_pred))
print("전체 pred fake 수:", (df_pred["pred_label"] == 1).sum())
print("전체 pred fake 비율:", round((df_pred["pred_label"] == 1).mean(), 4))

print("\n상위 의심 product 10개:")
print(top_products)

print("\n저장 완료:")
print("all predictions:", ALL_PRED_PATH)
print("high confidence fake reviews:", HIGH_FAKE_PATH)
print("product suspicious summary:", PRODUCT_SUMMARY_PATH)
print("dashboard nodes:", DASHBOARD_NODE_PATH)
print("dashboard edges:", DASHBOARD_EDGE_PATH)
print("summary json:", SUMMARY_JSON_PATH)