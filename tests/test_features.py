import pandas as pd
import pytest

from credit_risk.features import (
    build_features_generic,
    build_structured_features,
    build_target,
    build_target_generic,
)


def test_build_target_is_binary(credit_df):
    y = build_target(credit_df)
    assert set(y.unique()) <= {0, 1}
    assert y.dtype.kind in ("i", "u")


def test_build_target_matches_charged_off_rows(credit_df):
    y = build_target(credit_df)
    expected_positive_count = (credit_df["loan_status"] == "Charged Off").sum()
    assert y.sum() == expected_positive_count


def test_build_structured_features_no_nulls(credit_df):
    X = build_structured_features(credit_df)
    assert X.isna().sum().sum() == 0


def test_build_structured_features_row_count_matches(credit_df):
    X = build_structured_features(credit_df)
    assert len(X) == len(credit_df)


def test_build_structured_features_all_numeric(credit_df):
    X = build_structured_features(credit_df)
    assert all(pd.api.types.is_numeric_dtype(X[col]) or X[col].dtype == bool for col in X.columns)


def test_build_target_generic_matches_build_target(credit_df):
    generic = build_target_generic(credit_df, target_col="loan_status", positive_value="Charged Off")
    original = build_target(credit_df)
    assert (generic == original).all()


def test_build_target_generic_works_on_different_schema(fraud_df):
    y = build_target_generic(fraud_df, target_col="Class", positive_value=1)
    assert set(y.unique()) <= {0, 1}
    assert y.sum() == (fraud_df["Class"] == 1).sum()


def test_build_features_generic_respects_column_selection(credit_df):
    X = build_features_generic(credit_df, categorical_cols=["grade"], numeric_cols=["loan_amnt", "int_rate"])
    # loan_amnt, int_rate pass through as-is; grade gets one-hot encoded into >=1 columns
    assert "loan_amnt" in X.columns
    assert "int_rate" in X.columns
    assert any(col.startswith("grade_") for col in X.columns)


def test_build_features_generic_empty_selection_returns_empty_frame(credit_df):
    X = build_features_generic(credit_df, categorical_cols=[], numeric_cols=[])
    assert X.shape[1] == 0
    assert len(X) == len(credit_df)


def test_build_features_generic_handles_missing_values(credit_df):
    # emp_title has real missing values in the sample data — should not raise.
    assert credit_df["emp_title"].isna().sum() > 0
    X = build_features_generic(credit_df, categorical_cols=["emp_title"], numeric_cols=[])
    assert X.isna().sum().sum() == 0
