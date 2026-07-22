"""
LLM-derived features from unstructured text fields (emp_title, purpose).

This is the "AI" layer for the credit risk model: Claude reads free-text
fields and returns structured signals that a classifier can consume. It is
kept separate from features.py (the structured/no-LLM path) so you can
directly compare a baseline model against an LLM-augmented one.

If ANTHROPIC_API_KEY isn't set, extract_llm_features() falls back to a
deterministic placeholder so the rest of the pipeline (and this file's
__main__ demo) still runs — swap in a real key to get real extractions.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from shared import llm_client

SYSTEM_PROMPT = """You are extracting structured risk signals from loan \
application text fields. Given an employment title and a stated loan \
purpose, respond with ONLY a JSON object (no markdown, no preamble) with \
these exact keys:
- "income_stability_signal": one of "high", "medium", "low" — inferred from \
the employment title alone (e.g. "Self-employed" or missing -> lower; \
"Teacher", "Nurse", government-sounding titles -> higher)
- "purpose_specificity": one of "specific", "vague" — whether the stated \
purpose is a concrete, verifiable need (e.g. "home_improvement") vs a vague \
catch-all (e.g. "other")
- "purpose_risk_flag": true or false — true if the purpose is one commonly \
associated with elevated default risk (e.g. "small_business", "moving")
"""


def _fallback_features(emp_title: str, purpose: str) -> dict:
    """
    Deterministic stand-in used only when no API key is configured, so the
    pipeline is runnable end-to-end without live calls. Not a substitute for
    real extraction — replace by setting ANTHROPIC_API_KEY.
    """
    h = int(hashlib.sha256(f"{emp_title}|{purpose}".encode()).hexdigest(), 16)
    return {
        "income_stability_signal": ["low", "medium", "high"][h % 3],
        "purpose_specificity": ["vague", "specific"][h % 2],
        "purpose_risk_flag": bool(h % 5 == 0),
    }


def extract_llm_features(emp_title: str, purpose: str) -> dict:
    """Extract structured signals from one (emp_title, purpose) pair."""
    if not llm_client.is_available():
        return _fallback_features(emp_title, purpose)

    prompt = f'Employment title: "{emp_title}"\nStated loan purpose: "{purpose}"'
    return llm_client.complete_json(prompt, system=SYSTEM_PROMPT, max_tokens=200)


def add_llm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs extraction for every row and returns a dataframe of one-hot encoded
    LLM-derived features, aligned to df's index, ready to concat onto the
    structured feature matrix from features.py.

    Note: this makes one API call per row. For a large dataset, add caching
    keyed on (emp_title, purpose) since many rows share the same pair.
    """
    records = [
        extract_llm_features(row.get("emp_title", ""), row.get("purpose", ""))
        for _, row in df.iterrows()
    ]
    llm_df = pd.DataFrame(records)
    llm_df["purpose_risk_flag"] = llm_df["purpose_risk_flag"].astype(int)
    return pd.get_dummies(llm_df, columns=["income_stability_signal", "purpose_specificity"])


if __name__ == "__main__":
    import os

    sample = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "samples", "credit_risk_sample.csv"))
    demo = add_llm_features(sample.head(5))
    print("LLM available:", llm_client.is_available())
    print(demo)


# --- Generic version: works on any text column names, not just emp_title/purpose ---

GENERIC_SYSTEM_PROMPT = """You are extracting structured risk signals from \
free-text fields in a loan or credit application. You will be given an \
arbitrary set of named text fields (field names vary by dataset). Respond \
with ONLY a JSON object (no markdown, no preamble) with these exact keys:
- "text_risk_signal": one of "high", "medium", "low" — your best-effort \
overall risk read of the combined text fields
- "text_specificity": one of "specific", "vague" — whether the text fields \
describe something concrete and verifiable vs. vague or generic
- "text_risk_flag": true or false — true if anything in the text raises a \
notable red flag
"""


def _fallback_features_generic(text_fields: dict) -> dict:
    h = int(hashlib.sha256(str(sorted(text_fields.items())).encode()).hexdigest(), 16)
    return {
        "text_risk_signal": ["low", "medium", "high"][h % 3],
        "text_specificity": ["vague", "specific"][h % 2],
        "text_risk_flag": bool(h % 5 == 0),
    }


def extract_llm_features_generic(text_fields: dict) -> dict:
    """text_fields: {column_name: value} for however many text columns the user picked."""
    if not llm_client.is_available():
        return _fallback_features_generic(text_fields)

    field_str = "\n".join(f'{name}: "{value}"' for name, value in text_fields.items())
    return llm_client.complete_json(field_str, system=GENERIC_SYSTEM_PROMPT, max_tokens=200)


def add_llm_features_generic(df: pd.DataFrame, text_cols: list[str]) -> pd.DataFrame:
    """
    Generic counterpart to add_llm_features — takes any list of text column
    names instead of assuming emp_title/purpose. Returns an empty dataframe
    (0 columns) if text_cols is empty, so it's safe to concat unconditionally.
    """
    if not text_cols:
        return pd.DataFrame(index=df.index)

    records = [
        extract_llm_features_generic({col: row.get(col, "") for col in text_cols})
        for _, row in df.iterrows()
    ]
    llm_df = pd.DataFrame(records)
    llm_df["text_risk_flag"] = llm_df["text_risk_flag"].astype(int)
    return pd.get_dummies(llm_df, columns=["text_risk_signal", "text_specificity"])
