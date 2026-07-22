"""
Trains and evaluates the credit risk classifier two ways:
  1. baseline  — structured features only
  2. augmented — structured features + LLM-derived text features

Prints ROC-AUC for both so the LLM contribution can be judged honestly
(report this comparison in the main README's Results section — including
if the lift is small or negative, which is still a real, defensible finding).

Usage:
    python credit_risk/train.py
"""

from __future__ import annotations

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from credit_risk.features import build_structured_features, build_target
from credit_risk.llm_features import add_llm_features

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "samples", "credit_risk_sample.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def train_and_eval(X: pd.DataFrame, y: pd.Series, label: str) -> RandomForestClassifier:
    if y.nunique() < 2:
        raise ValueError(
            f"[{label}] Target column has only one class present ({y.unique().tolist()}). "
            "The model can't learn to distinguish outcomes with only one class in the data — "
            "check that data/samples/credit_risk_sample.csv actually contains both "
            "'Charged Off' and 'Fully Paid' rows in loan_status."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"[{label}] ROC-AUC: {auc:.4f}  (n_train={len(X_train)}, n_test={len(X_test)})")
    return clf


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    y = build_target(raw)

    # 1. Baseline: structured features only.
    X_structured = build_structured_features(raw)
    train_and_eval(X_structured, y, "baseline (structured only)")

    # 2. Augmented: structured + LLM-derived features.
    X_llm = add_llm_features(raw)
    X_combined = pd.concat([X_structured.reset_index(drop=True), X_llm.reset_index(drop=True)], axis=1)
    augmented_clf = train_and_eval(X_combined, y, "augmented (structured + LLM)")

    joblib.dump(augmented_clf, MODEL_PATH)
    print(f"Saved augmented model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
