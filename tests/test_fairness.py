import numpy as np
import pandas as pd
import pytest

from fairness.audit import generate_report, pairwise_fpr_tests, subgroup_rates


@pytest.fixture
def perfect_predictions():
    """y_pred == y_true everywhere, across two groups — a clean baseline case."""
    y_true = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    y_pred = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    group = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])
    return y_true, y_pred, group


@pytest.fixture
def biased_predictions():
    """Group A gets false positives, group B doesn't — a clear disparity to detect."""
    y_true = pd.Series([0] * 20 + [0] * 20)
    y_pred = pd.Series([1] * 10 + [0] * 10 + [0] * 20)  # A: 10 false positives, B: none
    group = pd.Series(["A"] * 20 + ["B"] * 20)
    return y_true, y_pred, group


def test_subgroup_rates_perfect_predictions_have_zero_error(perfect_predictions):
    y_true, y_pred, group = perfect_predictions
    rates = subgroup_rates(y_true, y_pred, group)
    assert (rates["fpr"] == 0).all()
    assert (rates["fnr"] == 0).all()


def test_subgroup_rates_one_row_per_group(perfect_predictions):
    y_true, y_pred, group = perfect_predictions
    rates = subgroup_rates(y_true, y_pred, group)
    assert set(rates["group"]) == {"A", "B"}
    assert len(rates) == 2


def test_subgroup_rates_detects_fpr_disparity(biased_predictions):
    y_true, y_pred, group = biased_predictions
    rates = subgroup_rates(y_true, y_pred, group)
    fpr_a = rates.loc[rates["group"] == "A", "fpr"].iloc[0]
    fpr_b = rates.loc[rates["group"] == "B", "fpr"].iloc[0]
    assert fpr_a > fpr_b


def test_pairwise_fpr_tests_flags_clear_disparity_as_significant(biased_predictions):
    y_true, y_pred, group = biased_predictions
    tests = pairwise_fpr_tests(y_true, y_pred, group)
    assert len(tests) == 1
    assert tests.iloc[0]["significant_after_fdr"]


def test_pairwise_fpr_tests_no_disparity_when_predictions_match(perfect_predictions):
    y_true, y_pred, group = perfect_predictions
    tests = pairwise_fpr_tests(y_true, y_pred, group)
    assert not tests.iloc[0]["significant_after_fdr"]


def test_pairwise_fpr_tests_single_group_returns_empty():
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 1, 1])
    group = pd.Series(["A", "A", "A", "A"])
    tests = pairwise_fpr_tests(y_true, y_pred, group)
    assert tests.empty


def test_generate_report_returns_markdown_string(perfect_predictions):
    y_true, y_pred, group = perfect_predictions
    report = generate_report(y_true, y_pred, group, group_label="test_group")
    assert isinstance(report, str)
    assert "test_group" in report
    assert "Subgroup rates" in report
