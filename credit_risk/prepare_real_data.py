"""
Cleans the raw, full LendingClub "accepted loans" CSV (downloaded via
data/download_data.py) into a manageable file matching the same column
schema as data/samples/credit_risk_sample.csv, so it drops directly into
the existing app/training code with no further changes.

The real file needs this because, unlike the synthetic sample:
  - int_rate and revol_util are stored as strings like "13.56%", not floats
  - term is stored as "36 months" / " 36 months" with inconsistent spacing
  - loan_status has 7+ categories (Current, Late (31-120 days), In Grace
    Period, Does not meet the credit policy..., etc.), not just the two
    we train on
  - it has ~150 columns and 2-3 million rows — far more than needed for a
    portfolio-scale training run, and slow to work with directly in a
    Streamlit app

Usage:
    python credit_risk/prepare_real_data.py \
        --input data/full/credit_risk/accepted_2007_to_2018Q4.csv \
        --output data/full/credit_risk_real_clean.csv \
        --sample-size 50000
"""

from __future__ import annotations

import argparse

import pandas as pd

# The exact columns the rest of the app expects (matches
# data/samples/credit_risk_sample.csv and credit_risk/features.py).
EXPECTED_COLUMNS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "emp_title",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "delinq_2yrs", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "loan_status",
]

# Only these two outcomes are usable as a clean binary target — every other
# loan_status value (Current, Late, In Grace Period, etc.) is a loan that
# hasn't resolved yet, so it can't be labeled "default" or "paid off".
KEEP_STATUSES = ["Fully Paid", "Charged Off"]


def _clean_percent_column(series: pd.Series) -> pd.Series:
    """Handles '13.56%' strings, plain floats, or already-clean numeric columns uniformly."""
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")


def _clean_term_column(series: pd.Series) -> pd.Series:
    """Normalizes whitespace so " 36 months" and "36 months" aren't treated as different categories."""
    return series.astype(str).str.strip()


def clean_lendingclub_csv(
    input_path: str,
    output_path: str,
    sample_size: int | None = 50_000,
    random_state: int = 42,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Reads the raw file in chunks (it's large), keeps only the columns this
    project uses, filters to resolved loans (Fully Paid / Charged Off),
    cleans percent-string columns, and optionally downsamples for a
    manageable training/demo size. Writes the result to output_path and
    also returns it as a dataframe.
    """
    kept_chunks = []
    total_rows_seen = 0

    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        total_rows_seen += len(chunk)

        available_cols = [c for c in EXPECTED_COLUMNS if c in chunk.columns]
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in chunk.columns]
        if missing_cols and not kept_chunks:
            # Only warn once, on the first chunk, so we don't spam the console.
            print(f"Warning: {len(missing_cols)} expected column(s) not found in the source file: {missing_cols}")

        chunk = chunk[available_cols]
        chunk = chunk[chunk["loan_status"].isin(KEEP_STATUSES)]
        if not chunk.empty:
            kept_chunks.append(chunk)

    if not kept_chunks:
        raise ValueError(
            "No rows with loan_status in ['Fully Paid', 'Charged Off'] were found. "
            "Check that the input file is the LendingClub accepted-loans CSV and that "
            "column names match what's expected."
        )

    df = pd.concat(kept_chunks, ignore_index=True)
    print(f"Read {total_rows_seen:,} total rows, kept {len(df):,} resolved (Fully Paid / Charged Off) rows.")

    if "int_rate" in df.columns:
        df["int_rate"] = _clean_percent_column(df["int_rate"])
    if "revol_util" in df.columns:
        df["revol_util"] = _clean_percent_column(df["revol_util"])
    if "term" in df.columns:
        df["term"] = _clean_term_column(df["term"])

    if sample_size is not None and len(df) > sample_size:
        # Stratified sample so the class balance in the output matches the
        # real class balance in the full file, rather than skewing it.
        # Uses groupby().sample() (not .apply()) — .apply() silently drops
        # the grouping column when the returned frame still contains it
        # unchanged, which would delete loan_status from the output.
        frac = sample_size / len(df)
        df = df.groupby("loan_status", group_keys=False).sample(frac=frac, random_state=random_state).reset_index(drop=True)
        print(f"Downsampled to {len(df):,} rows (stratified by loan_status) for a manageable training size.")

    df.to_csv(output_path, index=False)
    print(f"Saved cleaned file to {output_path}")
    print(f"Class balance: {dict(df['loan_status'].value_counts())}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the raw downloaded LendingClub CSV")
    parser.add_argument("--output", default="data/full/credit_risk_real_clean.csv", help="Where to save the cleaned CSV")
    parser.add_argument("--sample-size", type=int, default=50_000, help="Max rows in the output (stratified sample). Use 0 for no limit.")
    args = parser.parse_args()

    sample_size = None if args.sample_size == 0 else args.sample_size
    clean_lendingclub_csv(args.input, args.output, sample_size=sample_size)
