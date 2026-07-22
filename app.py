"""
Unified dashboard: upload -> EDA agent -> credit risk / fraud scoring ->
fairness audit, all as tabs of one Streamlit app.

Every tab accepts an uploaded CSV. If none is uploaded, it falls back to the
matching sample file in data/samples/ so the app is always usable out of
the box.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from agent.claude_agent import run as run_agent
from agent.render import render_full_profile
from credit_risk.features import CATEGORICAL_COLS, NUMERIC_COLS, build_structured_features, build_target
from credit_risk.llm_features import add_llm_features
from fairness.audit import generate_report
from fraud.model import load_features as load_fraud_features
from fraud.model import train_and_eval as train_fraud_model
from shared import llm_client
from shared.schema import CREDIT_TARGET, FRAUD_AMOUNT_COL, FRAUD_TARGET

st.set_page_config(page_title="Responsible Credit & Fraud Risk Platform", layout="wide")
st.title("Responsible Credit & Fraud Risk Platform")

if not llm_client.is_available():
    st.warning(
        " "
        
    )


def show_schema_help(
    sample_path: str, required_cols: list[str], target_col: str, context: str, target_note: str = ""
) -> None:
    """
    Renders an expander showing exactly which columns are required, an
    example value for each (pulled from the sample file), and a button to
    download that sample as a starting template.

    `context` must be unique per call site (e.g. "credit", "fairness") since
    two tabs can reference the same sample file and Streamlit widget keys
    must be unique.
    """
    with st.expander("📋 Expected CSV format — click to see required columns"):
        sample_df = pd.read_csv(sample_path)
        example_row = sample_df.iloc[0]

        schema_table = pd.DataFrame(
            {
                "column": required_cols,
                "dtype": [str(sample_df[c].dtype) if c in sample_df.columns else "?" for c in required_cols],
                "example value": [example_row[c] if c in sample_df.columns else "?" for c in required_cols],
                "role": ["target (label)" if c == target_col else "feature" for c in required_cols],
            }
        )
        st.dataframe(schema_table, use_container_width=True, hide_index=True)

        if target_note:
            st.caption(target_note)

        st.caption(
            "Column names must match exactly (case-sensitive). Extra columns beyond "
            "these are fine and will be ignored."
        )

        with open(sample_path, "rb") as f:
            st.download_button(
                "⬇ Download a template CSV",
                f,
                file_name=os.path.basename(sample_path),
                mime="text/csv",
                key=f"template_{context}",
            )


def load_dataset(uploaded_file, default_path: str, required_cols: list[str], schema_note: str) -> pd.DataFrame | None:
    """
    Returns a validated dataframe from the uploaded file, or the default
    sample if nothing was uploaded. Shows a clear error and returns None if
    an uploaded file is missing required columns, instead of letting a
    downstream KeyError crash the whole page.
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source_note = f"Using uploaded file: **{uploaded_file.name}**"
    else:
        df = pd.read_csv(default_path)
        source_note = f"No file uploaded — using default sample: `{default_path}`"

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(
            f"This file is missing required column(s): {missing}. {schema_note}"
        )
        return None

    st.caption(source_note)
    return df


tab_eda, tab_credit, tab_fraud, tab_fairness = st.tabs(
    ["1. EDA agent", "2. Credit risk", "3. Fraud detection", "4. Fairness audit"]
)

with tab_eda:
    st.subheader("Upload a dataset to auto-profile")
    uploaded = st.file_uploader("CSV file", type="csv", key="eda_upload")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button("Run EDA agent"):
            if llm_client.is_available():
                with st.spinner("Agent is planning and running its analysis..."):
                    summary = run_agent(df)
                st.markdown(summary)
            else:
                render_full_profile(df)

with tab_credit:
    st.subheader("Credit risk scoring")
    st.caption(
        "Upload a CSV with the same columns as data/samples/credit_risk_sample.csv, "
        "or leave empty to use the sample."
    )
    show_schema_help(
        sample_path="data/samples/credit_risk_sample.csv",
        required_cols=CATEGORICAL_COLS + NUMERIC_COLS + [CREDIT_TARGET],
        target_col=CREDIT_TARGET,
        context="credit",
        target_note=f"'{CREDIT_TARGET}' must contain both outcome values (e.g. 'Charged Off' and 'Fully Paid') — a model can't learn from a file with only one outcome.",
    )
    credit_upload = st.file_uploader("CSV file", type="csv", key="credit_upload")

    if st.button("Train credit risk models"):
        raw = load_dataset(
            credit_upload,
            default_path="data/samples/credit_risk_sample.csv",
            required_cols=CATEGORICAL_COLS + NUMERIC_COLS + [CREDIT_TARGET],
            schema_note="See data/README.md for the expected schema.",
        )
        if raw is None:
            st.stop()

        y = build_target(raw)

        if y.nunique() < 2:
            st.error(
                f"{CREDIT_TARGET} only has one outcome present in this file ({raw[CREDIT_TARGET].unique().tolist()}). "
                "A classifier needs both outcomes present to learn anything — check your uploaded file "
                "or the default sample."
            )
            st.stop()

        X_structured = build_structured_features(raw)
        with st.spinner("Training baseline model..."):
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X_structured, y, test_size=0.25, random_state=42, stratify=y
            )
            baseline = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
            baseline.fit(X_train, y_train)
            baseline_auc = roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])

        with st.spinner("Extracting LLM features and training augmented model..."):
            X_llm = add_llm_features(raw)
            X_combined = pd.concat([X_structured.reset_index(drop=True), X_llm.reset_index(drop=True)], axis=1)
            X_train2, X_test2, y_train2, y_test2 = train_test_split(
                X_combined, y, test_size=0.25, random_state=42, stratify=y
            )
            augmented = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
            augmented.fit(X_train2, y_train2)
            augmented_auc = roc_auc_score(y_test2, augmented.predict_proba(X_test2)[:, 1])

        col1, col2 = st.columns(2)
        col1.metric("Baseline ROC-AUC", f"{baseline_auc:.4f}")
        col2.metric("LLM-augmented ROC-AUC", f"{augmented_auc:.4f}", delta=f"{augmented_auc - baseline_auc:+.4f}")

with tab_fraud:
    st.subheader("Fraud detection")
    st.caption(
        "Upload a CSV with the same columns as data/samples/fraud_sample.csv, "
        "or leave empty to use the sample."
    )
    _fraud_sample_cols = list(pd.read_csv("data/samples/fraud_sample.csv", nrows=1).columns)
    show_schema_help(
        sample_path="data/samples/fraud_sample.csv",
        required_cols=_fraud_sample_cols,
        target_col=FRAUD_TARGET,
        context="fraud",
        target_note=(
            f"Only '{FRAUD_TARGET}' and '{FRAUD_AMOUNT_COL}' are strictly required by name — "
            "the rest can be any numeric feature columns, they don't need to be called V1-V28 specifically."
        ),
    )
    fraud_upload = st.file_uploader("CSV file", type="csv", key="fraud_upload")

    if st.button("Train fraud model"):
        raw = load_dataset(
            fraud_upload,
            default_path="data/samples/fraud_sample.csv",
            required_cols=[FRAUD_TARGET, FRAUD_AMOUNT_COL],
            schema_note="See data/README.md for the expected schema.",
        )
        if raw is None:
            st.stop()

        y = raw[FRAUD_TARGET]
        X = raw.drop(columns=[FRAUD_TARGET])

        if y.nunique() < 2:
            st.error(
                f"{FRAUD_TARGET} only has one outcome present in this file. "
                "A classifier needs both fraud and non-fraud rows to learn anything."
            )
            st.stop()

        with st.spinner("Training fraud model..."):
            clf = train_fraud_model(X, y)
        st.success("Model trained — see terminal/logs for precision, recall, ROC-AUC.")

        st.write("Sample flag explanation:")
        from fraud.explain import explain_flag, top_contributing_features

        row = X.iloc[0]
        top_feats = top_contributing_features(clf, row)
        st.json(top_feats)
        st.write(explain_flag(row[FRAUD_AMOUNT_COL], top_feats))

with tab_fairness:
    st.subheader("Fairness audit")
    st.caption(
        "Upload the same kind of file as the credit risk tab. Choose which column to audit "
        "as the protected/demographic attribute below — neither sample dataset has a genuine "
        "one, so pick a real attribute if your uploaded file has it, or a documented proxy."
    )
    show_schema_help(
        sample_path="data/samples/credit_risk_sample.csv",
        required_cols=CATEGORICAL_COLS + NUMERIC_COLS + [CREDIT_TARGET],
        target_col=CREDIT_TARGET,
        context="fairness",
        target_note="After uploading, pick any column below (e.g. a real demographic attribute) to audit for subgroup fairness.",
    )
    fairness_upload = st.file_uploader("CSV file", type="csv", key="fairness_upload")

    raw_preview = load_dataset(
        fairness_upload,
        default_path="data/samples/credit_risk_sample.csv",
        required_cols=CATEGORICAL_COLS + NUMERIC_COLS + [CREDIT_TARGET],
        schema_note="See data/README.md for the expected schema.",
    )

    if raw_preview is not None:
        default_group_col = "home_ownership" if "home_ownership" in raw_preview.columns else raw_preview.columns[0]
        group_col = st.selectbox(
            "Column to audit for fairness",
            options=list(raw_preview.columns),
            index=list(raw_preview.columns).index(default_group_col),
        )

        if st.button("Run fairness audit"):
            raw = raw_preview
            y_true = build_target(raw)
            X = pd.concat(
                [
                    build_structured_features(raw).reset_index(drop=True),
                    add_llm_features(raw).reset_index(drop=True),
                ],
                axis=1,
            )
            from sklearn.ensemble import RandomForestClassifier

            clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
            clf.fit(X, y_true)
            y_pred = pd.Series(clf.predict(X), index=raw.index)

            report = generate_report(y_true, y_pred, raw[group_col], group_label=group_col)
            st.markdown(report)
