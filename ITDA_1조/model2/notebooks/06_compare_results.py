import json
from pathlib import Path

import pandas as pd


# =========================
# 0. Path setting
# =========================

CURRENT_FILE = Path(__file__).resolve()

# ITDA/model2/notebooks/06_compare_results.py 기준
MODEL2_DIR = CURRENT_FILE.parents[1]
ITDA_DIR = CURRENT_FILE.parents[2]
MODEL1_DIR = ITDA_DIR / "model1"

MODEL1_OUTPUT_DIR = MODEL1_DIR / "outputs"
MODEL2_OUTPUT_DIR = MODEL2_DIR / "outputs"

MODEL2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV_PATH = MODEL2_OUTPUT_DIR / "model2_comparison.csv"
OUTPUT_JSON_PATH = MODEL2_OUTPUT_DIR / "model2_comparison.json"
OUTPUT_MARKDOWN_PATH = MODEL2_OUTPUT_DIR / "model2_comparison_markdown.txt"


# =========================
# 1. Helper functions
# =========================

def load_json(path):
    if not path.exists():
        print("파일 없음:", path)
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_test_result(result_json):
    if result_json is None:
        return None

    return result_json.get("test_result", None)


def make_row(model_name, source, result_json, note):
    test_result = get_test_result(result_json)

    if test_result is None:
        return {
            "model_name": model_name,
            "source": source,
            "test_pr_auc": None,
            "test_macro_f1": None,
            "test_accuracy": None,
            "test_precision": None,
            "test_recall": None,
            "confusion_matrix": None,
            "note": note
        }

    return {
        "model_name": model_name,
        "source": source,
        "test_pr_auc": test_result.get("pr_auc"),
        "test_macro_f1": test_result.get("macro_f1"),
        "test_accuracy": test_result.get("accuracy"),
        "test_precision": test_result.get("precision"),
        "test_recall": test_result.get("recall"),
        "confusion_matrix": test_result.get("confusion_matrix"),
        "note": note
    }


# =========================
# 2. Load result files
# =========================

model1_rgcn_path = MODEL1_OUTPUT_DIR / "rgcn_result.json"

model2_rgcn_basic_path = MODEL2_OUTPUT_DIR / "rgcn_result.json"
model2_rgcn_tuned_path = MODEL2_OUTPUT_DIR / "rgcn_tuned_result.json"
model2_rgcn_without_nub_path = MODEL2_OUTPUT_DIR / "rgcn_without_new_user_burst_tuned_result.json"

model1_rgcn = load_json(model1_rgcn_path)

model2_rgcn_basic = load_json(model2_rgcn_basic_path)
model2_rgcn_tuned = load_json(model2_rgcn_tuned_path)
model2_rgcn_without_nub = load_json(model2_rgcn_without_nub_path)


# =========================
# 3. Make comparison table
# =========================

rows = []

rows.append(
    make_row(
        model_name="model1_rgcn_with_new_user_burst",
        source="model1",
        result_json=model1_rgcn,
        note="model1 final model. manual 8 + TF-IDF/SVD 32. Uses RUR, PRODUCT_TIME_7D, NEW_USER_BURST_7D."
    )
)

rows.append(
    make_row(
        model_name="model2_rgcn_basic_with_new_user_burst",
        source="model2",
        result_json=model2_rgcn_basic,
        note="model2 basic RGCN. manual 8 + SBERT 384. Uses all 3 edge types."
    )
)

rows.append(
    make_row(
        model_name="model2_rgcn_tuned_with_new_user_burst",
        source="model2",
        result_json=model2_rgcn_tuned,
        note="model2 tuned RGCN. hidden_dim 32, dropout 0.5, lr 0.002, weight_decay 5e-4. Uses all 3 edge types."
    )
)

rows.append(
    make_row(
        model_name="model2_rgcn_tuned_without_new_user_burst",
        source="model2",
        result_json=model2_rgcn_without_nub,
        note="model2 tuned RGCN ablation. Removes NEW_USER_BURST_7D edge. Uses only RUR and PRODUCT_TIME_7D."
    )
)

comparison_df = pd.DataFrame(rows)

comparison_df = comparison_df.sort_values(
    by="test_pr_auc",
    ascending=False,
    na_position="last"
).reset_index(drop=True)


# =========================
# 4. Add ranking and difference
# =========================

comparison_df.insert(0, "rank_by_pr_auc", range(1, len(comparison_df) + 1))

model1_pr_auc = None

model1_rows = comparison_df[
    comparison_df["model_name"] == "model1_rgcn_with_new_user_burst"
]

if len(model1_rows) == 1:
    model1_pr_auc = model1_rows.iloc[0]["test_pr_auc"]

if model1_pr_auc is not None:
    comparison_df["pr_auc_diff_from_model1"] = comparison_df["test_pr_auc"] - model1_pr_auc
else:
    comparison_df["pr_auc_diff_from_model1"] = None


# =========================
# 5. Save files
# =========================

comparison_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

comparison_records = comparison_df.to_dict(orient="records")

summary = {
    "description": "Comparison between model1 final RGCN and model2 RGCN variants.",
    "ranking_metric": "test_pr_auc",
    "best_model": comparison_df.iloc[0]["model_name"],
    "best_test_pr_auc": comparison_df.iloc[0]["test_pr_auc"],
    "model1_baseline_pr_auc": model1_pr_auc,
    "records": comparison_records
}

with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)


# =========================
# 6. Make markdown table
# =========================

markdown_lines = []

markdown_lines.append("# Model Comparison")
markdown_lines.append("")
markdown_lines.append("| Rank | Model | Test PR-AUC | Macro F1 | Accuracy | Precision | Recall | PR-AUC diff from model1 |")
markdown_lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")

for _, row in comparison_df.iterrows():
    diff = row["pr_auc_diff_from_model1"]

    if pd.isna(diff):
        diff_text = ""
    else:
        diff_text = f"{diff:.4f}"

    markdown_lines.append(
        "| "
        + str(row["rank_by_pr_auc"])
        + " | "
        + str(row["model_name"])
        + " | "
        + f"{row['test_pr_auc']:.4f}"
        + " | "
        + f"{row['test_macro_f1']:.4f}"
        + " | "
        + f"{row['test_accuracy']:.4f}"
        + " | "
        + f"{row['test_precision']:.4f}"
        + " | "
        + f"{row['test_recall']:.4f}"
        + " | "
        + diff_text
        + " |"
    )

markdown_lines.append("")
markdown_lines.append("## Interpretation")
markdown_lines.append("")
markdown_lines.append(
    "The best model is `"
    + str(comparison_df.iloc[0]["model_name"])
    + "`, with Test PR-AUC "
    + f"{comparison_df.iloc[0]['test_pr_auc']:.4f}"
    + "."
)

if model1_pr_auc is not None:
    best_diff = comparison_df.iloc[0]["test_pr_auc"] - model1_pr_auc

    markdown_lines.append(
        "Compared with the model1 final RGCN baseline, the best model2 result changes Test PR-AUC by "
        + f"{best_diff:.4f}"
        + "."
    )

markdown_lines.append("")
markdown_lines.append(
    "The ablation result without `NEW_USER_BURST_7D` shows whether the new-user burst relation remains useful under SBERT-based text features."
)

with open(OUTPUT_MARKDOWN_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(markdown_lines))


# =========================
# 7. Print result
# =========================

print("\n==============================")
print("Model Comparison")
print("==============================")

print(comparison_df)

print("\n저장 완료:")
print("csv:", OUTPUT_CSV_PATH)
print("json:", OUTPUT_JSON_PATH)
print("markdown:", OUTPUT_MARKDOWN_PATH)

print("\nBest model:", comparison_df.iloc[0]["model_name"])
print("Best Test PR-AUC:", round(comparison_df.iloc[0]["test_pr_auc"], 4))

if model1_pr_auc is not None:
    print("model1 baseline PR-AUC:", round(model1_pr_auc, 4))
    print(
        "best - model1 PR-AUC diff:",
        round(comparison_df.iloc[0]["test_pr_auc"] - model1_pr_auc, 4)
    )

print("\n==============================")
print("06_compare_results.py 완료")
print("==============================")