"""
Structured (non-LLM) feature engineering for the credit risk model.

Takes the raw loan dataframe and returns a numeric feature matrix + target,
ready for a scikit-learn estimator. This is intentionally kept separate from
llm_features.py so the baseline model (structured-only) and the augmented
model (structured + LLM) can be trained and compared cleanly.
"""

from __future__ import annotations

import pandas as pd

from shared.schema import CREDIT_TARGET, CREDIT_TARGET_POSITIVE

CATEGORICAL_COLS = [
    "term",
    "grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
]

NUMERIC_COLS = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
]


def build_target(df: pd.DataFrame) -> pd.Series:
    """1 = defaulted / charged off, 0 = fully paid."""
    return (df[CREDIT_TARGET] == CREDIT_TARGET_POSITIVE).astype(int)


def build_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categoricals, pass numeric columns through, median-impute
    missing values. Returns a numeric-only dataframe ready for sklearn.
    """
    numeric = df[NUMERIC_COLS].copy()
    numeric = numeric.fillna(numeric.median(numeric_only=True))

    categorical = pd.get_dummies(df[CATEGORICAL_COLS].astype(str), dummy_na=True)

    return pd.concat([numeric.reset_index(drop=True), categorical.reset_index(drop=True)], axis=1)


def load_and_build(csv_path: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Convenience loader: returns (raw_df, target, structured_features)."""
    raw = pd.read_csv(csv_path)
    y = build_target(raw)
    X = build_structured_features(raw)
    return raw, y, X
