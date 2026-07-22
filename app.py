"""
Unified dashboard: upload -> EDA agent -> credit risk / fraud scoring ->
fairness audit, all as tabs of one Streamlit app.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from agent.claude_agent import run as run_agent
from agent.render import render_full_profile
from credit_risk.features import build_structured_features, build_target
from credit_risk.llm_features import add_llm_features
from fairness.audit import generate_report
from fraud.model import load_features as load_fraud_features
from fraud.model import train_and_eval as train_fraud_model
from shared import llm_client

st.set_page_config(page_title="Responsible Credit & Fraud Risk Platform", layout="wide")
st.title("Responsible Credit & Fraud Risk Platform")

if not llm_client.is_available():
    st.warning(
        "No ANTHROPIC_API_KEY set — LLM-powered features (agentic EDA, LLM feature "
        "extraction, fraud explanations) will run in fallback/demo mode. "
        "Set the environment variable and restart to see live results."
    )

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
    st.caption("Uses data/samples/credit_risk_sample.csv by default. Trains a baseline and an LLM-augmented model.")
    if st.button("Train credit risk models"):
        raw = pd.read_csv("data/samples/credit_risk_sample.csv")
        y = build_target(raw)

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
    st.caption("Uses data/samples/fraud_sample.csv by default.")
    if st.button("Train fraud model"):
        X, y = load_fraud_features("data/samples/fraud_sample.csv")
        with st.spinner("Training fraud model..."):
            clf = train_fraud_model(X, y)
        st.success("Model trained — see terminal/logs for precision, recall, ROC-AUC.")

        st.write("Sample flag explanation:")
        from fraud.explain import explain_flag, top_contributing_features

        row = X.iloc[0]
        top_feats = top_contributing_features(clf, row)
        st.json(top_feats)
        st.write(explain_flag(row["Amount"], top_feats))

with tab_fairness:
    st.subheader("Fairness audit")
    st.caption(
        "Demonstration only: home_ownership is used as a proxy grouping since neither "
        "sample dataset includes a genuine demographic column. See data/README.md."
    )
    if st.button("Run fairness audit"):
        raw = pd.read_csv("data/samples/credit_risk_sample.csv")
        y_true = build_target(raw)
        X = pd.concat(
            [build_structured_features(raw).reset_index(drop=True), add_llm_features(raw).reset_index(drop=True)],
            axis=1,
        )
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
        clf.fit(X, y_true)
        y_pred = pd.Series(clf.predict(X), index=raw.index)

        report = generate_report(y_true, y_pred, raw["home_ownership"], group_label="home_ownership (demo proxy)")
        st.markdown(report)
