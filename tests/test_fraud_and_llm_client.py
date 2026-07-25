import os

import pandas as pd
import pytest

from fraud.model import load_features, train_and_eval
from shared import llm_client


def test_load_features_splits_target_correctly(fraud_df):
    X, y = load_features(os.path.join(os.path.dirname(__file__), "..", "data", "samples", "fraud_sample.csv"))
    assert "Class" not in X.columns
    assert list(y.unique()) != [] and set(y.unique()) <= {0, 1}
    assert len(X) == len(y) == len(fraud_df)


def test_train_and_eval_returns_fitted_classifier(fraud_df):
    y = fraud_df["Class"]
    X = fraud_df.drop(columns=["Class"])
    clf = train_and_eval(X, y)
    assert hasattr(clf, "predict")
    preds = clf.predict(X.head(5))
    assert len(preds) == 5


def test_llm_client_is_available_reflects_env_var(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm_client.is_available() is False

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key-for-testing")
    assert llm_client.is_available() is True


def test_llm_client_complete_json_strips_code_fences(monkeypatch):
    """complete_json should parse JSON even if the model wraps it in ```json fences."""

    class FakeResponse:
        content = []

    def fake_complete(prompt, system=None, max_tokens=1000):
        return '```json\n{"a": 1, "b": "x"}\n```'

    monkeypatch.setattr(llm_client, "complete", fake_complete)
    result = llm_client.complete_json("irrelevant prompt")
    assert result == {"a": 1, "b": "x"}
