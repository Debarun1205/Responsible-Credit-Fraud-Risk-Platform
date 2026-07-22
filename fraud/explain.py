"""
Generates a natural-language explanation for why a transaction was flagged
as fraudulent, given its feature values and the model's top contributing
features. This is a pure explainability layer — it runs after the model
has already made its decision and does not influence it.

If ANTHROPIC_API_KEY isn't set, falls back to a templated explanation so
the pipeline still runs end-to-end without a live key.
"""

from __future__ import annotations

import pandas as pd

from shared import llm_client

SYSTEM_PROMPT = """You are explaining, in one or two plain-English sentences, \
why an automated fraud model flagged a transaction. You are given the \
transaction's Amount and its top contributing feature values (already \
identified by the model — you are not deciding whether it's fraud, only \
explaining the flag). Be concrete and reference the actual values given. \
Do not hedge with "this could indicate" more than once. Respond with plain \
text only, no JSON, no markdown."""


def _fallback_explanation(amount: float, top_features: dict[str, float]) -> str:
    feature_str = ", ".join(f"{k}={v:.2f}" for k, v in top_features.items())
    return (
        f"Flagged based on an unusual combination of feature values ({feature_str}) "
        f"for a transaction of amount {amount:.2f}, which deviates from typical patterns "
        f"the model learned from genuine transactions."
    )


def explain_flag(amount: float, top_features: dict[str, float]) -> str:
    if not llm_client.is_available():
        return _fallback_explanation(amount, top_features)

    feature_str = ", ".join(f"{k} = {v:.3f}" for k, v in top_features.items())
    prompt = f"Transaction amount: {amount:.2f}\nTop contributing feature values: {feature_str}"
    return llm_client.complete(prompt, system=SYSTEM_PROMPT, max_tokens=150)


def top_contributing_features(clf, row: pd.Series, n: int = 4) -> dict[str, float]:
    """
    Picks the n features with the highest (importance * |value|) for this
    specific row — a simple, model-agnostic-enough proxy for "what drove
    this particular prediction." For a more rigorous per-row attribution,
    swap this out for SHAP values.
    """
    importances = pd.Series(clf.feature_importances_, index=row.index)
    contribution = importances * row.abs()
    top = contribution.sort_values(ascending=False).head(n).index
    return {feat: row[feat] for feat in top}


if __name__ == "__main__":
    import os

    import joblib

    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "samples", "fraud_sample.csv")

    clf = joblib.load(model_path)
    df = pd.read_csv(data_path)
    row = df.drop(columns=["Class"]).iloc[0]

    top_feats = top_contributing_features(clf, row)
    print("Top features:", top_feats)
    print("Explanation:", explain_flag(row["Amount"], top_feats))
