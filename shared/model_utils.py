"""
Shared model training/evaluation utilities used by both the credit risk and
fraud tabs, so "train 3 models and compare," "show feature importance," and
"tune the classification threshold" aren't implemented twice.

No LLM calls anywhere in this file — pure scikit-learn / XGBoost.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except ImportError:  # pragma: no cover - falls back gracefully if xgboost isn't installed
    _HAS_XGBOOST = False


def get_candidate_models() -> dict:
    """
    Returns {display_name: unfitted_estimator}. XGBoost is included only if
    the package is actually installed — if not, it's silently omitted rather
    than crashing the app, and the comparison table just has 2 rows instead of 3.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=42, class_weight="balanced"
        ),
    }
    if _HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1, eval_metric="logloss", random_state=42
        )
    return models


def train_and_compare(X: pd.DataFrame, y: pd.Series, test_size: float = 0.25, random_state: int = 42) -> tuple[pd.DataFrame, dict]:
    """
    Trains every candidate model on the same train/test split and returns:
      - a comparison dataframe (one row per model: ROC-AUC, precision, recall, F1)
      - a dict of {name: (fitted_model, X_test, y_test, y_proba)} for downstream
        use (feature importance, threshold tuning, download).
    """
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise ValueError(
            f"The target column has only one class present ({class_counts.index.tolist()}). "
            "A classifier needs at least two outcomes to learn anything — check your target "
            "column and positive-value selection."
        )
    min_count = class_counts.min()
    min_needed = max(2, round(1 / test_size))  # roughly what a stratified split needs per class
    if min_count < min_needed:
        raise ValueError(
            f"The smallest class in the target column has only {min_count} row(s) "
            f"({dict(class_counts)}). That's too few to split into train/test sets reliably — "
            "this usually means the wrong column was picked as the target (e.g. an ID or "
            "timestamp column instead of a real label). Double-check your target column and "
            "positive-value selection above."
        )

    # XGBoost rejects column names containing [, ], or < — these show up
    # naturally from one-hot encoding category values like "< 1 year".
    # Sanitize once here so all models see consistent, safe column names.
    X = X.rename(columns=lambda c: re.sub(r"[\[\]<>]", "_", str(c)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    rows = []
    fitted = {}
    for name, model in get_candidate_models().items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        rows.append(
            {
                "model": name,
                "roc_auc": roc_auc_score(y_test, proba),
                "precision": precision_score(y_test, preds, zero_division=0),
                "recall": recall_score(y_test, preds, zero_division=0),
                "f1": f1_score(y_test, preds, zero_division=0),
            }
        )
        fitted[name] = (model, X_test, y_test, proba)

    comparison_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return comparison_df, fitted


def feature_importance_df(model, feature_names: list[str], top_n: int = 15) -> pd.DataFrame | None:
    """
    Returns a sorted dataframe of the top_n most important features, or None
    if the model doesn't expose feature_importances_ (e.g. LogisticRegression
    uses .coef_ instead — handled separately below).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def precision_recall_at_thresholds(y_true: pd.Series, y_proba: np.ndarray, thresholds=None) -> pd.DataFrame:
    """
    Computes precision/recall/F1/flagged-count at each threshold, so a UI
    slider can show the live trade-off instead of only the default 0.5 cutoff.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)

    rows = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        rows.append(
            {
                "threshold": round(float(t), 2),
                "precision": precision_score(y_true, preds, zero_division=0),
                "recall": recall_score(y_true, preds, zero_division=0),
                "f1": f1_score(y_true, preds, zero_division=0),
                "flagged_count": int(preds.sum()),
            }
        )
    return pd.DataFrame(rows)


def metrics_at_threshold(y_true: pd.Series, y_proba: np.ndarray, threshold: float) -> dict:
    preds = (y_proba >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "flagged_count": int(preds.sum()),
    }
