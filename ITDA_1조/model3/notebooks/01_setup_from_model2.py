from pathlib import Path
import shutil
import json
from datetime import datetime

# ============================================================
# 01_setup_from_model2.py
# 목적:
# - model3 폴더 구조 생성
# - model2의 sample, split, edge, SBERT feature 파일을 model3로 복사
# - model1/model2와 동일 sample/split을 유지하여 공정 비교 가능하게 만들기
# ============================================================

# 현재 파일 위치:
# ITDA/model3/notebooks/01_setup_from_model2.py
NOTEBOOK_DIR = Path(__file__).resolve().parent
MODEL3_DIR = NOTEBOOK_DIR.parent
ITDA_DIR = MODEL3_DIR.parent

MODEL2_DIR = ITDA_DIR / "model2"

MODEL3_DATA_DIR = MODEL3_DIR / "data"
MODEL3_CONFIG_DIR = MODEL3_DIR / "configs"
MODEL3_OUTPUT_DIR = MODEL3_DIR / "outputs"
MODEL3_REPORT_DIR = MODEL3_DIR / "report_assets"

# ------------------------------------------------------------
# 1. 폴더 생성
# ------------------------------------------------------------

dirs_to_create = [
    MODEL3_DATA_DIR,
    MODEL3_CONFIG_DIR,
    MODEL3_OUTPUT_DIR,
    MODEL3_REPORT_DIR,
]

for folder in dirs_to_create:
    folder.mkdir(parents=True, exist_ok=True)

print("[OK] model3 folder structure created")

# ------------------------------------------------------------
# 2. model2에서 가져올 파일 목록
# ------------------------------------------------------------
# model3에서는 model2와 같은 sample/split을 유지한다.
# edge는 base edge로 복사한 뒤, 이후 03 단계에서 TEXT_SIMILARITY edge를 추가한다.
# model2_node_features.csv는 SBERT 384차원이 들어 있으므로,
# 이후 02 단계에서 PCA 128 feature를 만들 때 사용한다.

copy_plan = [
    # sample / split
    {
        "src": MODEL2_DIR / "data" / "model2_sampled_reviews.csv",
        "dst": MODEL3_DATA_DIR / "model3_sampled_reviews.csv",
        "required": True,
    },
    {
        "src": MODEL2_DIR / "data" / "model2_split.csv",
        "dst": MODEL3_DATA_DIR / "model3_split.csv",
        "required": True,
    },

    # base edge files
    {
        "src": MODEL2_DIR / "data" / "model2_edge_list.csv",
        "dst": MODEL3_DATA_DIR / "model3_edge_list_base.csv",
        "required": True,
    },
    {
        "src": MODEL2_DIR / "data" / "model2_edge_index.npy",
        "dst": MODEL3_DATA_DIR / "model3_edge_index_base.npy",
        "required": True,
    },
    {
        "src": MODEL2_DIR / "data" / "model2_edge_type.npy",
        "dst": MODEL3_DATA_DIR / "model3_edge_type_base.npy",
        "required": True,
    },

    # SBERT 384 node feature source
    {
        "src": MODEL2_DIR / "data" / "model2_node_features.csv",
        "dst": MODEL3_DATA_DIR / "model3_source_node_features_sbert384.csv",
        "required": True,
    },

    # config
    {
        "src": MODEL2_DIR / "configs" / "model2_edge_config.json",
        "dst": MODEL3_CONFIG_DIR / "model3_edge_config_base.json",
        "required": True,
    },
    {
        "src": MODEL2_DIR / "configs" / "model2_feature_columns.json",
        "dst": MODEL3_CONFIG_DIR / "model3_source_feature_columns_sbert384.json",
        "required": True,
    },

    # model2 best result reference
    {
        "src": MODEL2_DIR / "outputs" / "rgcn_tuned_result.json",
        "dst": MODEL3_OUTPUT_DIR / "model2_reference_rgcn_tuned_result.json",
        "required": False,
    },
    {
        "src": MODEL2_DIR / "outputs" / "model2_comparison.csv",
        "dst": MODEL3_OUTPUT_DIR / "model2_reference_comparison.csv",
        "required": False,
    },
    {
        "src": MODEL2_DIR / "outputs" / "model2_comparison.json",
        "dst": MODEL3_OUTPUT_DIR / "model2_reference_comparison.json",
        "required": False,
    },
]

# ------------------------------------------------------------
# 3. 파일 복사
# ------------------------------------------------------------

copied_files = []
missing_required = []
missing_optional = []

for item in copy_plan:
    src = item["src"]
    dst = item["dst"]
    required = item["required"]

    if src.exists():
        shutil.copy2(src, dst)
        copied_files.append({
            "src": str(src),
            "dst": str(dst),
            "required": required
        })
        print("[COPIED]", src.name, "->", dst.name)
    else:
        if required:
            missing_required.append(str(src))
            print("[MISSING REQUIRED]", src)
        else:
            missing_optional.append(str(src))
            print("[MISSING OPTIONAL]", src)

# ------------------------------------------------------------
# 4. required 파일 누락 확인
# ------------------------------------------------------------

if len(missing_required) > 0:
    print("\n[ERROR] Required files are missing.")
    print("아래 파일들이 model2에 있는지 확인해야 함:")
    for path in missing_required:
        print("-", path)

    raise FileNotFoundError("Required model2 files are missing. Check model2 folder.")

# ------------------------------------------------------------
# 5. setup summary 저장
# ------------------------------------------------------------

summary = {
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "model3",
    "source_model": "model2",
    "purpose": "Initialize model3 using model2 sample, split, edge, and SBERT feature source.",
    "strategy": {
        "sample": "Reuse model2 sample",
        "split": "Reuse model2 split",
        "base_edges": [
            "RUR",
            "PRODUCT_TIME_7D",
            "NEW_USER_BURST_7D"
        ],
        "next_steps": [
            "Create SBERT PCA 128 node features",
            "Create TEXT_SIMILARITY edge",
            "Train RGCN and compare with model2"
        ]
    },
    "copied_files": copied_files,
    "missing_optional_files": missing_optional,
    "missing_required_files": missing_required
}

summary_path = MODEL3_OUTPUT_DIR / "setup_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n[OK] setup summary saved:")
print(summary_path)

print("\n[DONE] model3 setup from model2 completed successfully.")
print("다음 단계: 02_node_features_sbert_pca.py 실행")