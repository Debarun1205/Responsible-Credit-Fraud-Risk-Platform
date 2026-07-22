"""
Fairness audit: computes false-positive/false-negative rates per subgroup of
a protected/demographic column, and flags statistically significant
disparities while controlling the false discovery rate (FDR) across all
subgroup comparisons — so a "significant" finding isn't just noise from
testing many groups at once.

Neither sample dataset ships with a genuine protected attribute (see
data/README.md). Point `protected_col` at whatever column you've chosen —
a real attribute, a documented proxy, or a synthetic column for
demonstration — and label it clearly in whatever report you generate.

Usage:
    python fairness/audit.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def subgroup_rates(y_true: pd.Series, y_pred: pd.Series, group: pd.Series) -> pd.DataFrame:
    """One row per subgroup value, with FPR, FNR, and raw counts."""
    rows = []
    for value in group.dropna().unique():
        mask = group == value
        counts = _confusion_counts(y_true[mask], y_pred[mask])
        fpr = counts["fp"] / (counts["fp"] + counts["tn"]) if (counts["fp"] + counts["tn"]) else np.nan
        fnr = counts["fn"] / (counts["fn"] + counts["tp"]) if (counts["fn"] + counts["tp"]) else np.nan
        rows.append({"group": value, "n": int(mask.sum()), "fpr": fpr, "fnr": fnr, **counts})
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def pairwise_fpr_tests(y_true: pd.Series, y_pred: pd.Series, group: pd.Series, fdr_alpha: float = 0.05) -> pd.DataFrame:
    """
    Runs a two-proportion z-test on FPR between every pair of subgroups,
    then applies Benjamini-Hochberg FDR correction across all pairwise
    p-values so multiple comparisons don't inflate false "significant"
    findings.
    """
    values = list(group.dropna().unique())
    pairs, raw_pvalues = [], []

    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            g1_mask = group == values[i]
            g2_mask = group == values[j]

            c1 = _confusion_counts(y_true[g1_mask], y_pred[g1_mask])
            c2 = _confusion_counts(y_true[g2_mask], y_pred[g2_mask])

            n1, n2 = c1["fp"] + c1["tn"], c2["fp"] + c2["tn"]
            if n1 == 0 or n2 == 0:
                continue

            p1, p2 = c1["fp"] / n1, c2["fp"] / n2
            p_pool = (c1["fp"] + c2["fp"]) / (n1 + n2)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
            z = (p1 - p2) / se if se > 0 else 0.0
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))

            pairs.append({"group_a": values[i], "group_b": values[j], "fpr_a": p1, "fpr_b": p2, "p_value": p_value})
            raw_pvalues.append(p_value)

    result = pd.DataFrame(pairs)
    if result.empty:
        return result

    # Benjamini-Hochberg FDR correction.
    m = len(raw_pvalues)
    order = np.argsort(raw_pvalues)
    ranked = np.array(raw_pvalues)[order]
    bh_critical = (np.arange(1, m + 1) / m) * fdr_alpha
    passed = ranked <= bh_critical
    threshold = ranked[passed].max() if passed.any() else 0.0

    result["significant_after_fdr"] = result["p_value"] <= threshold
    return result.sort_values("p_value").reset_index(drop=True)


def generate_report(y_true: pd.Series, y_pred: pd.Series, group: pd.Series, group_label: str) -> str:
    rates = subgroup_rates(y_true, y_pred, group)
    tests = pairwise_fpr_tests(y_true, y_pred, group)

    lines = [f"# Fairness audit: {group_label}\n", "## Subgroup rates\n", rates.to_markdown(index=False), ""]
    if not tests.empty:
        lines += ["## Pairwise FPR comparisons (FDR-controlled)\n", tests.to_markdown(index=False)]
        n_sig = int(tests["significant_after_fdr"].sum())
        lines.append(f"\n**{n_sig} of {len(tests)} pairwise comparisons remained significant after FDR correction.**")
    else:
        lines.append("Not enough subgroup pairs to run pairwise tests.")

    return "\n".join(lines)


if __name__ == "__main__":
    import os

    import joblib

    model_path = os.path.join(os.path.dirname(__file__), "..", "credit_risk", "model.pkl")
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "samples", "credit_risk_sample.csv")

    if not os.path.exists(model_path):
        print("No trained model found — run `python credit_risk/train.py` first.")
    else:
        from credit_risk.features import build_structured_features, build_target
        from credit_risk.llm_features import add_llm_features

        raw = pd.read_csv(data_path)
        y_true = build_target(raw)
        X = pd.concat(
            [build_structured_features(raw).reset_index(drop=True), add_llm_features(raw).reset_index(drop=True)],
            axis=1,
        )
        clf = joblib.load(model_path)
        y_pred = pd.Series(clf.predict(X), index=raw.index)

        # home_ownership used here only as a demonstration proxy grouping —
        # see data/README.md for why neither dataset has a true demographic column.
        report = generate_report(y_true, y_pred, raw["home_ownership"], group_label="home_ownership (demo proxy)")
        print(report)
