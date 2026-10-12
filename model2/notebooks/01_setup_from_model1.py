import json
import shutil
from pathlib import Path

import pandas as pd


# =========================
# 0. Path setting
# =========================

CURRENT_FILE = Path(__file__).resolve()

# ITDA/model2/notebooks/01_setup_from_model1.py 기준
MODEL2_DIR = CURRENT_FILE.parents[1]
ITDA_DIR = CURRENT_FILE.parents[2]
MODEL1_DIR = ITDA_DIR / "model1"

MODEL1_DATA_DIR = MODEL1_DIR / "data"
MODEL1_CONFIG_DIR = MODEL1_DIR / "configs"

MODEL2_DATA_DIR = MODEL2_DIR / "data"
MODEL2_CONFIG_DIR = MODEL2_DIR / "configs"
MODEL2_OUTPUT_DIR = MODEL2_DIR / "outputs"

MODEL2_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL2_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 1. Source and target files
# =========================

copy_pairs = [
    # sample
    (
        MODEL1_DATA_DIR / "model1_sampled_reviews.csv",
        MODEL2_DATA_DIR / "model2_sampled_reviews.csv"
    ),

    # split
    (
        MODEL1_DATA_DIR / "model1_split.csv",
        MODEL2_DATA_DIR / "model2_split.csv"
    ),

    # edge files
    (
        MODEL1_DATA_DIR / "model1_edge_list.csv",
        MODEL2_DATA_DIR / "model2_edge_list.csv"
    ),
    (
        MODEL1_DATA_DIR / "model1_edge_index.npy",
        MODEL2_DATA_DIR / "model2_edge_index.npy"
    ),
    (
        MODEL1_DATA_DIR / "model1_edge_type.npy",
        MODEL2_DATA_DIR / "model2_edge_type.npy"
    ),

    # edge config
    (
        MODEL1_CONFIG_DIR / "model1_edge_config.json",
        MODEL2_CONFIG_DIR / "model2_edge_config.json"
    ),
]


# =========================
# 2. Check source files
# =========================

missing_files = []

for source_path, _ in copy_pairs:
    if not source_path.exists():
        missing_files.append(str(source_path))

if missing_files:
    print("다음 파일이 없음:")
    for path in missing_files:
        print("-", path)
    raise FileNotFoundError("model1에서 복사할 파일이 일부 없음")

print("모든 source file 확인 완료")


# =========================
# 3. Copy files
# =========================

copied_files = []

for source_path, target_path in copy_pairs:
    shutil.copy2(source_path, target_path)
    copied_files.append({
        "source": str(source_path),
        "target": str(target_path)
    })
    print("복사 완료:")
    print(" ", source_path)
    print(" ->", target_path)


# =========================
# 4. Load and check copied files
# =========================

sample_path = MODEL2_DATA_DIR / "model2_sampled_reviews.csv"
split_path = MODEL2_DATA_DIR / "model2_split.csv"
edge_list_path = MODEL2_DATA_DIR / "model2_edge_list.csv"

sample_df = pd.read_csv(sample_path)
split_df = pd.read_csv(split_path)
edge_df = pd.read_csv(edge_list_path)

print("\n==============================")
print("model2 setup check")
print("==============================")

print("\nsample shape:", sample_df.shape)
print("sample columns:", list(sample_df.columns))
print("sample product 수:", sample_df["prod_id"].nunique())
print("sample user 수:", sample_df["user_id"].nunique())

print("\nsample label 분포:")
print(sample_df["label"].value_counts().sort_index())

fake_ratio = sample_df["label"].mean()
print("sample fake 비율:", round(fake_ratio, 4))

print("\nsplit shape:", split_df.shape)
print("split columns:", list(split_df.columns))

print("\nsplit 분포:")
if "split" in split_df.columns:
    print(split_df["split"].value_counts())
elif "mask" in split_df.columns:
    print(split_df["mask"].value_counts())
else:
    print("주의: split 또는 mask 컬럼을 찾지 못함")

print("\nedge list shape:", edge_df.shape)
print("edge list columns:", list(edge_df.columns))

if "edge_type" in edge_df.columns:
    print("\nedge type 분포:")
    print(edge_df["edge_type"].value_counts())
elif "relation" in edge_df.columns:
    print("\nrelation 분포:")
    print(edge_df["relation"].value_counts())
else:
    print("주의: edge_type 또는 relation 컬럼을 찾지 못함")


# =========================
# 5. Save summary
# =========================

summary = {
    "model2_goal": "Reuse model1 sample, split, and edges. Upgrade only text representation from TF-IDF/SVD to SBERT.",
    "copied_files": copied_files,
    "sample_shape": list(sample_df.shape),
    "sample_product_count": int(sample_df["prod_id"].nunique()),
    "sample_user_count": int(sample_df["user_id"].nunique()),
    "sample_label_counts": {
        str(k): int(v)
        for k, v in sample_df["label"].value_counts().sort_index().to_dict().items()
    },
    "sample_fake_ratio": float(fake_ratio),
    "split_shape": list(split_df.shape),
    "edge_list_shape": list(edge_df.shape),
}

summary_path = MODEL2_OUTPUT_DIR / "setup_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\nsetup summary 저장 완료:")
print(summary_path)

print("\n==============================")
print("01_setup_from_model1.py 완료")
print("==============================")