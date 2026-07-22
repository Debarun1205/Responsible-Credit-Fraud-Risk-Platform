"""
Fraud detection model.

Uses a supervised classifier (the sample/full dataset has labels), evaluated
with precision/recall/ROC-AUC rather than raw accuracy, since fraud is
heavily imbalanced (~0.1-0.2% positive class in the real dataset).

Usage:
    python fraud/model.py
"""

from __future__ import annotations

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from shared.schema import FRAUD_TARGET

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "samples", "fraud_sample.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


def load_features(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    y = df[FRAUD_TARGET]
    X = df.drop(columns=[FRAUD_TARGET])
    return X, y


def train_and_eval(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    auc = roc_auc_score(y_test, proba)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)

    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"(n_train={len(X_train)}, n_test={len(X_test)}, positive rate={y.mean():.4%})")

    return clf


def main() -> None:
    X, y = load_features(DATA_PATH)
    clf = train_and_eval(X, y)
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
