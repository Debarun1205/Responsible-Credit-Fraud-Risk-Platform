from agent.profiler import (
    categorical_summary,
    correlation_matrix,
    missingness_report,
    numeric_summary,
)


def test_missingness_report_detects_known_missing_column(credit_df):
    report = missingness_report(credit_df)
    assert "emp_title" in report
    assert report["emp_title"]["missing_count"] == credit_df["emp_title"].isna().sum()


def test_missingness_report_excludes_complete_columns(credit_df):
    report = missingness_report(credit_df)
    assert "loan_amnt" not in report  # loan_amnt has no missing values in the sample


def test_numeric_summary_includes_known_numeric_columns(credit_df):
    summary = numeric_summary(credit_df)
    assert "loan_amnt" in summary
    assert "int_rate" in summary
    assert "count" in summary["loan_amnt"]


def test_numeric_summary_excludes_text_columns(credit_df):
    summary = numeric_summary(credit_df)
    assert "purpose" not in summary


def test_categorical_summary_includes_known_categorical_columns(credit_df):
    summary = categorical_summary(credit_df)
    assert "purpose" in summary
    assert "grade" in summary


def test_categorical_summary_respects_max_categories(credit_df):
    summary = categorical_summary(credit_df, max_categories=2)
    assert len(summary["purpose"]) <= 2


def test_correlation_matrix_finds_known_relationship(credit_df):
    # loan_amnt and installment are constructed to be strongly correlated
    # in the synthetic sample generator.
    pairs = correlation_matrix(credit_df, threshold=0.5)
    pair_cols = [{p["col_a"], p["col_b"]} for p in pairs]
    assert {"loan_amnt", "installment"} in pair_cols


def test_correlation_matrix_respects_threshold(credit_df):
    pairs_loose = correlation_matrix(credit_df, threshold=0.1)
    pairs_strict = correlation_matrix(credit_df, threshold=0.9)
    assert len(pairs_strict) <= len(pairs_loose)
