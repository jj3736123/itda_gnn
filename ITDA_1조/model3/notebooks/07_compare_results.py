from pathlib import Path
import json
from datetime import datetime

import pandas as pd


NOTEBOOK_DIR = Path(__file__).resolve().parent
MODEL3_DIR = NOTEBOOK_DIR.parent

OUTPUT_DIR = MODEL3_DIR / "outputs"

MODEL2_REFERENCE_PATH = OUTPUT_DIR / "model2_reference_rgcn_tuned_result.json"
MODEL3_ORIGINAL_PATH = OUTPUT_DIR / "rgcn_pca_original_edges_result.json"
MODEL3_TEXT_PATH = OUTPUT_DIR / "rgcn_pca_text_similarity_result.json"

OUTPUT_CSV_PATH = OUTPUT_DIR / "model3_comparison.csv"
OUTPUT_JSON_PATH = OUTPUT_DIR / "model3_comparison.json"
OUTPUT_MD_PATH = OUTPUT_DIR / "model3_comparison_markdown.txt"


def load_json(path):
    if not path.exists():
        print("[WARNING] file not found:", path)
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_nested_value(data, key_list, default=None):
    current = data

    for key in key_list:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current


def extract_result(data, fallback_name):
    if data is None:
        return None

    model_name = data.get("model_name", fallback_name)

    test_result = data.get("test_result", {})
    valid_result = data.get("valid_result", {})

    row = {
        "model": model_name,
        "test_pr_auc": test_result.get("pr_auc", None),
        "test_macro_f1": test_result.get("macro_f1", None),
        "test_accuracy": test_result.get("accuracy", None),
        "test_precision": test_result.get("precision", None),
        "test_recall": test_result.get("recall", None),
        "threshold": test_result.get("threshold", None),
        "valid_pr_auc": valid_result.get("pr_auc", None),
        "valid_macro_f1": valid_result.get("macro_f1", None),
        "best_epoch": get_nested_value(data, ["best", "best_epoch"], None),
        "edge_count": get_nested_value(data, ["edges", "edge_count"], None),
        "num_relations": get_nested_value(data, ["edges", "num_relations"], None),
        "text_similarity_edge_used": get_nested_value(data, ["edges", "text_similarity_edge_used"], None),
        "confusion_matrix": str(test_result.get("confusion_matrix", None))
    }

    return row


print("[LOAD] model2 reference:", MODEL2_REFERENCE_PATH)
model2_data = load_json(MODEL2_REFERENCE_PATH)

print("[LOAD] model3 original:", MODEL3_ORIGINAL_PATH)
model3_original_data = load_json(MODEL3_ORIGINAL_PATH)

print("[LOAD] model3 text similarity:", MODEL3_TEXT_PATH)
model3_text_data = load_json(MODEL3_TEXT_PATH)

rows = []

row_model2 = extract_result(
    model2_data,
    "model2_rgcn_tuned_with_new_user_burst"
)

row_model3_original = extract_result(
    model3_original_data,
    "model3_rgcn_pca128_original_edges"
)

row_model3_text = extract_result(
    model3_text_data,
    "model3_rgcn_pca128_text_similarity"
)

for row in [row_model2, row_model3_original, row_model3_text]:
    if row is not None:
        rows.append(row)

comparison_df = pd.DataFrame(rows)

# model2 json 구조가 다를 경우 README 기준값으로 보완
if len(comparison_df) > 0:
    for idx in comparison_df.index:
        model_name = str(comparison_df.loc[idx, "model"])

        if "model2" in model_name:
            if pd.isna(comparison_df.loc[idx, "test_pr_auc"]):
                comparison_df.loc[idx, "test_pr_auc"] = 0.5789
            if pd.isna(comparison_df.loc[idx, "test_macro_f1"]):
                comparison_df.loc[idx, "test_macro_f1"] = 0.7075
            if pd.isna(comparison_df.loc[idx, "test_accuracy"]):
                comparison_df.loc[idx, "test_accuracy"] = 0.8088
            if pd.isna(comparison_df.loc[idx, "test_precision"]):
                comparison_df.loc[idx, "test_precision"] = 0.5339
            if pd.isna(comparison_df.loc[idx, "test_recall"]):
                comparison_df.loc[idx, "test_recall"] = 0.5370
            if pd.isna(comparison_df.loc[idx, "threshold"]):
                comparison_df.loc[idx, "threshold"] = None
            comparison_df.loc[idx, "text_similarity_edge_used"] = False

comparison_df = comparison_df.sort_values(
    by="test_pr_auc",
    ascending=False
).reset_index(drop=True)

comparison_df.insert(0, "rank", range(1, len(comparison_df) + 1))

model2_pr_auc = None
model2_macro_f1 = None
model3_original_pr_auc = None
model3_text_pr_auc = None

for idx in comparison_df.index:
    model_name = str(comparison_df.loc[idx, "model"])

    if "model2" in model_name:
        model2_pr_auc = comparison_df.loc[idx, "test_pr_auc"]
        model2_macro_f1 = comparison_df.loc[idx, "test_macro_f1"]

    if "original_edges" in model_name:
        model3_original_pr_auc = comparison_df.loc[idx, "test_pr_auc"]

    if "text_similarity" in model_name:
        model3_text_pr_auc = comparison_df.loc[idx, "test_pr_auc"]

if model2_pr_auc is not None:
    comparison_df["pr_auc_diff_from_model2"] = comparison_df["test_pr_auc"] - model2_pr_auc
else:
    comparison_df["pr_auc_diff_from_model2"] = None

if model2_macro_f1 is not None:
    comparison_df["macro_f1_diff_from_model2"] = comparison_df["test_macro_f1"] - model2_macro_f1
else:
    comparison_df["macro_f1_diff_from_model2"] = None

comparison_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

summary = {
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "model3",
    "comparison": comparison_df.to_dict(orient="records"),
    "main_findings": {
        "best_model": comparison_df.iloc[0]["model"] if len(comparison_df) > 0 else None,
        "best_test_pr_auc": float(comparison_df.iloc[0]["test_pr_auc"]) if len(comparison_df) > 0 else None,
        "best_test_macro_f1": float(comparison_df.iloc[0]["test_macro_f1"]) if len(comparison_df) > 0 else None,
        "model2_pr_auc": float(model2_pr_auc) if model2_pr_auc is not None else None,
        "model3_original_pr_auc": float(model3_original_pr_auc) if model3_original_pr_auc is not None else None,
        "model3_text_pr_auc": float(model3_text_pr_auc) if model3_text_pr_auc is not None else None,
        "pca_effect": float(model3_original_pr_auc - model2_pr_auc) if model2_pr_auc is not None and model3_original_pr_auc is not None else None,
        "text_similarity_edge_effect": float(model3_text_pr_auc - model3_original_pr_auc) if model3_original_pr_auc is not None and model3_text_pr_auc is not None else None,
        "total_improvement_from_model2": float(model3_text_pr_auc - model2_pr_auc) if model2_pr_auc is not None and model3_text_pr_auc is not None else None
    }
}

with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

md_lines = []

md_lines.append("# Model3 Comparison Result")
md_lines.append("")
md_lines.append("## Performance Table")
md_lines.append("")
md_lines.append("| Rank | Model | Test PR-AUC | Macro F1 | Accuracy | Precision | Recall | Threshold |")
md_lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")

for idx in comparison_df.index:
    row = comparison_df.loc[idx]

    threshold_value = "-"
    if row["threshold"] is not None and not pd.isna(row["threshold"]):
        if isinstance(row["threshold"], str):
            threshold_value = row["threshold"]
        else:
            threshold_value = str(round(float(row["threshold"]), 2))

    line = (
        "| "
        + str(row["rank"])
        + " | "
        + str(row["model"])
        + " | "
        + str(round(float(row["test_pr_auc"]), 4))
        + " | "
        + str(round(float(row["test_macro_f1"]), 4))
        + " | "
        + str(round(float(row["test_accuracy"]), 4))
        + " | "
        + str(round(float(row["test_precision"]), 4))
        + " | "
        + str(round(float(row["test_recall"]), 4))
        + " | "
        + threshold_value
        + " |"
    )

    md_lines.append(line)

md_lines.append("")
md_lines.append("## Main Findings")
md_lines.append("")

if model2_pr_auc is not None and model3_original_pr_auc is not None:
    md_lines.append(
        "- PCA 128 original edge model improved Test PR-AUC from "
        + str(round(float(model2_pr_auc), 4))
        + " to "
        + str(round(float(model3_original_pr_auc), 4))
        + "."
    )

if model3_original_pr_auc is not None and model3_text_pr_auc is not None:
    md_lines.append(
        "- Adding TEXT_SIMILARITY edge improved Test PR-AUC from "
        + str(round(float(model3_original_pr_auc), 4))
        + " to "
        + str(round(float(model3_text_pr_auc), 4))
        + "."
    )

if model2_pr_auc is not None and model3_text_pr_auc is not None:
    md_lines.append(
        "- Final model3 improved Test PR-AUC by "
        + str(round(float(model3_text_pr_auc - model2_pr_auc), 4))
        + " compared with model2 best."
    )

md_text = "\n".join(md_lines)

with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
    f.write(md_text)

print("[SAVED] csv:", OUTPUT_CSV_PATH)
print("[SAVED] json:", OUTPUT_JSON_PATH)
print("[SAVED] markdown:", OUTPUT_MD_PATH)

print("\n[COMPARISON]")
print(comparison_df)

print("\n[DONE] model3 comparison completed.")