"""
These tests only exercise the fallback (no ANTHROPIC_API_KEY) code paths,
since that's what runs in CI / without a live key. They confirm the fallback
is deterministic and shaped correctly — not that the real LLM extraction
produces good features, which needs a live key to check.
"""

import pandas as pd

from credit_risk.llm_features import (
    add_llm_features,
    add_llm_features_generic,
    extract_llm_features,
    extract_llm_features_generic,
)


def test_extract_llm_features_fallback_is_deterministic():
    result_1 = extract_llm_features("Teacher", "medical")
    result_2 = extract_llm_features("Teacher", "medical")
    assert result_1 == result_2


def test_extract_llm_features_fallback_has_expected_keys():
    result = extract_llm_features("Teacher", "medical")
    assert set(result.keys()) == {"income_stability_signal", "purpose_specificity", "purpose_risk_flag"}
    assert result["income_stability_signal"] in ("low", "medium", "high")
    assert result["purpose_specificity"] in ("vague", "specific")
    assert isinstance(result["purpose_risk_flag"], bool)


def test_add_llm_features_row_count_matches(credit_df):
    result = add_llm_features(credit_df.head(10))
    assert len(result) == 10


def test_extract_llm_features_generic_fallback_deterministic():
    fields = {"emp_title": "Nurse", "purpose": "vacation"}
    assert extract_llm_features_generic(fields) == extract_llm_features_generic(fields)


def test_add_llm_features_generic_empty_text_cols_returns_empty(credit_df):
    result = add_llm_features_generic(credit_df.head(5), text_cols=[])
    assert result.shape == (5, 0)


def test_add_llm_features_generic_with_text_cols(credit_df):
    result = add_llm_features_generic(credit_df.head(5), text_cols=["emp_title", "purpose"])
    assert len(result) == 5
    assert "text_risk_flag" in result.columns


def test_add_llm_features_generic_works_on_arbitrary_column_names(fraud_df):
    # Fraud sample has no natural text columns, but the function should still
    # run without error on whatever is passed, treating values as strings.
    fake_text_df = fraud_df.head(3).copy()
    fake_text_df["note"] = ["urgent wire transfer", "routine purchase", "unknown recipient"]
    result = add_llm_features_generic(fake_text_df, text_cols=["note"])
    assert len(result) == 3
