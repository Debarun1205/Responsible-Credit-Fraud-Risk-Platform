import numpy as np
import pandas as pd
import pytest

from credit_risk.features import build_features_generic, build_target_generic
from shared.model_utils import (
    feature_importance_df,
    guess_target_column,
    metrics_at_threshold,
    precision_recall_at_thresholds,
    train_and_compare,
)


@pytest.fixture
def credit_xy(credit_df):
    y = build_target_generic(credit_df, "loan_status", "Charged Off")
    X = build_features_generic(
        credit_df,
        categorical_cols=["grade", "home_ownership", "purpose"],
        numeric_cols=["loan_amnt", "int_rate", "dti"],
    )
    return X, y


# --- guess_target_column ---


def test_guess_target_column_uses_preferred_if_present(credit_df):
    assert guess_target_column(credit_df, preferred="loan_status") == "loan_status"


def test_guess_target_column_falls_back_to_common_name(fraud_df):
    # "loan_status" isn't in the fraud data, so it should fall back to "Class"
    # rather than the first column (which is "Time" — a bad target).
    assert guess_target_column(fraud_df, preferred="loan_status") == "Class"


def test_guess_target_column_never_picks_a_high_cardinality_column(fraud_df):
    guessed = guess_target_column(fraud_df, preferred="nonexistent_column")
    assert fraud_df[guessed].nunique() <= fraud_df.shape[0] / 2


# --- train_and_compare ---


def test_train_and_compare_returns_all_expected_models(credit_xy):
    X, y = credit_xy
    comparison_df, fitted = train_and_compare(X, y)
    assert "Logistic Regression" in comparison_df["model"].values
    assert "Random Forest" in comparison_df["model"].values
    assert set(comparison_df.columns) >= {"model", "roc_auc", "precision", "recall", "f1"}


def test_train_and_compare_auc_in_valid_range(credit_xy):
    X, y = credit_xy
    comparison_df, _ = train_and_compare(X, y)
    assert comparison_df["roc_auc"].between(0, 1).all()


def test_train_and_compare_sorted_best_first(credit_xy):
    X, y = credit_xy
    comparison_df, _ = train_and_compare(X, y)
    aucs = comparison_df["roc_auc"].tolist()
    assert aucs == sorted(aucs, reverse=True)


def test_train_and_compare_raises_on_single_class(credit_xy):
    X, _ = credit_xy
    y_constant = pd.Series([0] * len(X))
    with pytest.raises(ValueError, match="only one class"):
        train_and_compare(X, y_constant)


def test_train_and_compare_raises_clear_error_on_extreme_imbalance(credit_xy):
    X, _ = credit_xy
    y_degenerate = pd.Series([0] * (len(X) - 1) + [1])
    with pytest.raises(ValueError, match="too few"):
        train_and_compare(X, y_degenerate)


def test_train_and_compare_sanitizes_bracket_column_names(credit_df):
    # emp_length contains "< 1 year" which XGBoost rejects in raw column names —
    # this should not raise.
    y = build_target_generic(credit_df, "loan_status", "Charged Off")
    X = build_features_generic(credit_df, categorical_cols=["emp_length"], numeric_cols=["loan_amnt"])
    assert any("<" in col for col in X.columns)  # confirm the risky column actually exists pre-fix
    comparison_df, _ = train_and_compare(X, y)
    assert len(comparison_df) > 0


# --- feature_importance_df ---


def test_feature_importance_df_for_tree_model(credit_xy):
    X, y = credit_xy
    _, fitted = train_and_compare(X, y)
    rf_model, X_test, _, _ = fitted["Random Forest"]
    fi = feature_importance_df(rf_model, list(X_test.columns))
    assert fi is not None
    assert list(fi.columns) == ["feature", "importance"]
    assert fi["importance"].is_monotonic_decreasing


def test_feature_importance_df_for_logistic_regression_uses_coef(credit_xy):
    X, y = credit_xy
    _, fitted = train_and_compare(X, y)
    lr_model, X_test, _, _ = fitted["Logistic Regression"]
    fi = feature_importance_df(lr_model, list(X_test.columns))
    assert fi is not None  # should use .coef_ path, not return None


# --- threshold tuning ---


def test_precision_recall_at_thresholds_shape(credit_xy):
    X, y = credit_xy
    _, fitted = train_and_compare(X, y)
    _, _, y_test, y_proba = fitted["Random Forest"]
    pr_df = precision_recall_at_thresholds(y_test, y_proba)
    assert set(pr_df.columns) >= {"threshold", "precision", "recall", "f1", "flagged_count"}
    assert pr_df["precision"].between(0, 1).all()
    assert pr_df["recall"].between(0, 1).all()


def test_metrics_at_threshold_higher_threshold_flags_fewer_or_equal(credit_xy):
    X, y = credit_xy
    _, fitted = train_and_compare(X, y)
    _, _, y_test, y_proba = fitted["Random Forest"]
    low = metrics_at_threshold(y_test, y_proba, 0.1)
    high = metrics_at_threshold(y_test, y_proba, 0.9)
    assert high["flagged_count"] <= low["flagged_count"]
