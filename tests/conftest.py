"""Shared fixtures for the test suite."""

import os

import pandas as pd
import pytest

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "samples")


@pytest.fixture
def credit_df() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "credit_risk_sample.csv"))


@pytest.fixture
def fraud_df() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA_DIR, "fraud_sample.csv"))
