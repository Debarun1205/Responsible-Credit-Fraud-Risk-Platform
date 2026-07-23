"""
Unified dashboard: upload -> EDA agent -> credit risk / fraud scoring ->
fairness audit, all as tabs of one Streamlit app.

Every tab accepts an uploaded CSV of ANY shape. Rather than assuming fixed
column names, the credit risk and fairness tabs let the user pick which
column is the target, which columns are categorical/numeric features, and
which text columns (if any) should go through the LLM feature extractor.
Sensible defaults are pre-selected when the column names match the sample
dataset, but nothing is hardcoded to require them.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import os

import joblib
import pandas as pd
import streamlit as st

from agent.claude_agent import run as run_agent
from agent.dashboard import (
    render_feature_importance,
    render_full_dashboard,
    render_threshold_tradeoff,
)
from agent.render import render_full_profile
from credit_risk.features import build_features_generic, build_target_generic
from credit_risk.llm_features import add_llm_features_generic
from fairness.audit import generate_report
from shared import llm_client
from shared.model_utils import (
    feature_importance_df,
    metrics_at_threshold,
    precision_recall_at_thresholds,
    train_and_compare,
)

st.set_page_config(page_title="Responsible Credit & Fraud Risk Platform", layout="wide")

# --- Sidebar: about the creator ---
with st.sidebar:
    st.markdown("### About this project")
    st.markdown(
        "Built by **Debarun Banerjee**, B.Tech CSE (AI & ML) student "
        "at Narula Institute of Technology.\n\n"
        "[LinkedIn](https://www.linkedin.com/in/debarun-banerjee-b8524a37b) · "
        "[Portfolio](https://debarun.base44.app)"
    )
    st.markdown("---")
    st.caption(
        "This app spans agentic AI (LLM-driven feature extraction and EDA), "
        "machine learning (credit risk and fraud classifiers), and data "
        "science (fairness auditing). See `docs/domain_mapping.md` in the "
        "repo for the full breakdown."
    )

st.title("Responsible Credit & Fraud Risk Platform")

with st.expander("ℹ️ How to use this app (click to expand)"):
    st.markdown(
        """
1. **📊 Dashboard** — upload any CSV to see its shape, target balance, and distributions as charts. No ML background needed here.
2. **1. EDA agent** — an automated data-quality profile (missingness, correlations). Uses Claude if an API key is set, otherwise shows the same info as plain charts/tables.
3. **2. Credit risk** and **3. Fraud detection** — pick your target column, then train and compare three model types at once (Logistic Regression, Random Forest, XGBoost). See which features drove the predictions, and tune the decision threshold to trade off precision vs. recall.
4. **4. Fairness audit** — checks whether the model's error rates differ meaningfully across a demographic column or proxy, with statistically controlled significance testing.

Every tab works on the built-in sample data by default — just click the train/run button — or upload your own CSV with a similar shape.
        """
    )

if "training_history" not in st.session_state:
    st.session_state.training_history = []

if not llm_client.is_available():
    st.warning(
        "No ANTHROPIC_API_KEY set — LLM-powered features (agentic EDA, LLM feature "
        "extraction, fraud explanations) will run in fallback/demo mode. "
        "Set the key in Settings → Secrets to see live results."
    )


# --- Shared helpers ---


def load_dataset(uploaded_file, default_path: str) -> tuple[pd.DataFrame, str]:
    """Returns (dataframe, source_note). No schema assumptions — any CSV works here."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source_note = f"Using uploaded file: **{uploaded_file.name}** ({len(df)} rows, {len(df.columns)} columns)"
    else:
        df = pd.read_csv(default_path)
        source_note = f"No file uploaded — using default sample: `{default_path}` ({len(df)} rows)"
    return df, source_note


def _guess_target_column(df: pd.DataFrame, preferred: str | None) -> str:
    """
    Picks a sensible default target column instead of blindly falling back
    to the first column (which breaks badly if that happens to be an
    ID/timestamp-like column with hundreds of unique values).

    Priority: the caller's preferred name if present -> a common target-like
    name (Class, target, label, y, outcome) if present -> the lowest-
    cardinality column in the dataframe, since real target columns are
    almost always binary/low-cardinality, unlike IDs or continuous fields.
    """
    if preferred and preferred in df.columns:
        return preferred

    for candidate in ["Class", "class", "target", "label", "y", "outcome"]:
        if candidate in df.columns:
            return candidate

    nunique = df.nunique()
    # Prefer a genuinely binary column if one exists — that's what a
    # classification target almost always looks like.
    binary_cols = nunique[nunique == 2]
    if not binary_cols.empty:
        return binary_cols.index[0]

    # Otherwise, the lowest-cardinality column that isn't constant.
    candidates = nunique[nunique >= 2]
    return candidates.idxmin() if not candidates.empty else nunique.idxmin()


def column_picker(df: pd.DataFrame, context: str, default_target: str | None = None) -> dict:
    """
    Renders the target/feature/text-column selection UI for an arbitrary
    dataframe and returns the choices as a dict. `context` must be unique
    per call site so widget keys don't collide across tabs.
    """
    columns = list(df.columns)

    guessed_target = _guess_target_column(df, default_target)
    target_default_idx = columns.index(guessed_target)
    target_col = st.selectbox("Target column (what you're predicting)", columns, index=target_default_idx, key=f"target_{context}")

    if df[target_col].nunique() > 20:
        st.warning(
            f"'{target_col}' has {df[target_col].nunique()} unique values — that's unusual for a "
            "classification target. Double check you've picked the right column; a target should "
            "typically be binary or have just a few categories."
        )

    unique_values = df[target_col].dropna().unique().tolist()
    positive_value = st.selectbox(
        "Positive / 'bad outcome' value to predict",
        unique_values,
        index=0,
        key=f"positive_{context}",
        help="The class the model treats as the outcome of interest, e.g. 'Charged Off' or 1 for fraud.",
    )

    remaining_cols = [c for c in columns if c != target_col]
    default_categorical = [c for c in remaining_cols if not pd.api.types.is_numeric_dtype(df[c])]
    default_numeric = [c for c in remaining_cols if pd.api.types.is_numeric_dtype(df[c])]

    with st.expander("Advanced: adjust feature columns"):
        categorical_cols = st.multiselect(
            "Categorical columns (one-hot encoded)", remaining_cols, default=default_categorical, key=f"cat_{context}"
        )
        numeric_cols = st.multiselect(
            "Numeric columns (used as-is)", remaining_cols, default=default_numeric, key=f"num_{context}"
        )
        text_cols = st.multiselect(
            "Text columns to send through LLM feature extraction (optional)",
            categorical_cols,
            default=[c for c in categorical_cols if c in ("emp_title", "purpose")],
            key=f"text_{context}",
            help="These get passed to Claude for structured signal extraction. Leave empty to skip the LLM step entirely.",
        )

    return {
        "target_col": target_col,
        "positive_value": positive_value,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "text_cols": text_cols,
    }


def show_schema_help(sample_path: str, context: str) -> None:
    """Shows the sample file's columns/dtypes/example values and a download button, for reference."""
    with st.expander("📋 See an example file"):
        sample_df = pd.read_csv(sample_path)
        example_row = sample_df.iloc[0]
        st.dataframe(
            pd.DataFrame(
                {
                    "column": sample_df.columns,
                    "dtype": [str(sample_df[c].dtype) for c in sample_df.columns],
                    "example value": [example_row[c] for c in sample_df.columns],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Any CSV works here — this is just one example. Pick your own target/feature columns below.")
        with open(sample_path, "rb") as f:
            st.download_button("⬇ Download this example CSV", f, file_name=os.path.basename(sample_path), mime="text/csv", key=f"dl_{context}")


# --- Tabs ---

tab_dashboard, tab_eda, tab_credit, tab_fraud, tab_fairness = st.tabs(
    ["📊 Dashboard", "1. EDA agent", "2. Credit risk", "3. Fraud detection", "4. Fairness audit"]
)

with tab_dashboard:
    st.subheader("Visual overview — no ML background needed")
    st.caption(
        "Upload any CSV to see its shape, target balance, distributions, and correlations "
        "as charts. This tab never calls Claude/the LLM — it's pure chart generation, free to run."
    )
    dashboard_upload = st.file_uploader("CSV file", type="csv", key="dashboard_upload")
    dashboard_df, dashboard_source_note = load_dataset(dashboard_upload, "data/samples/credit_risk_sample.csv")
    st.caption(dashboard_source_note)

    dashboard_columns = list(dashboard_df.columns)
    dashboard_target_default = (
        dashboard_columns.index("loan_status")
        if "loan_status" in dashboard_columns
        else (dashboard_columns.index("Class") if "Class" in dashboard_columns else 0)
    )
    dashboard_target_col = st.selectbox(
        "Which column is the outcome you care about? (optional — pick 'None' to skip)",
        ["(none)"] + dashboard_columns,
        index=dashboard_target_default + 1,
        key="dashboard_target",
    )

    render_full_dashboard(
        dashboard_df,
        target_col=None if dashboard_target_col == "(none)" else dashboard_target_col,
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
    st.caption("Works with any CSV — pick your target column and features below.")
    show_schema_help("data/samples/credit_risk_sample.csv", context="credit")

    credit_upload = st.file_uploader("CSV file", type="csv", key="credit_upload")
    raw, source_note = load_dataset(credit_upload, "data/samples/credit_risk_sample.csv")
    st.caption(source_note)

    choices = column_picker(raw, context="credit", default_target="loan_status")

    if st.button("Train credit risk models"):
        y = build_target_generic(raw, choices["target_col"], choices["positive_value"])

        if y.nunique() < 2:
            st.error(
                f"'{choices['target_col']}' only has one value present after filtering. "
                "A classifier needs both outcomes present — check your column and value selections above."
            )
            st.stop()

        X_structured = build_features_generic(raw, choices["categorical_cols"], choices["numeric_cols"])
        if X_structured.shape[1] == 0:
            st.error("No feature columns selected. Pick at least one categorical or numeric column above.")
            st.stop()

        with st.spinner("Extracting LLM features (if any text columns were selected)..."):
            X_llm = add_llm_features_generic(raw, choices["text_cols"])
            X = pd.concat([X_structured.reset_index(drop=True), X_llm.reset_index(drop=True)], axis=1)

        with st.spinner("Training and comparing Logistic Regression, Random Forest, and XGBoost..."):
            try:
                comparison_df, fitted = train_and_compare(X, y)
            except ValueError as e:
                st.error(str(e))
                st.stop()

        st.markdown("#### Model comparison")
        st.dataframe(comparison_df.style.format({"roc_auc": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}"}), use_container_width=True, hide_index=True)

        best_name = comparison_df.iloc[0]["model"]
        best_model, X_test, y_test, y_proba = fitted[best_name]
        st.caption(f"Best model by ROC-AUC: **{best_name}**. Details below are for this model.")

        st.session_state.training_history.append(
            {"tab": "Credit risk", "model": best_name, "roc_auc": comparison_df.iloc[0]["roc_auc"], "rows": len(raw)}
        )

        render_feature_importance(feature_importance_df(best_model, list(X_test.columns)), best_name)

        st.markdown("#### Adjust the decision threshold")
        threshold = st.slider("Classification threshold", 0.05, 0.95, 0.5, 0.05, key="credit_threshold")
        pr_df = precision_recall_at_thresholds(y_test, y_proba)
        render_threshold_tradeoff(pr_df, threshold)
        live_metrics = metrics_at_threshold(y_test, y_proba, threshold)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{live_metrics['precision']:.3f}")
        m2.metric("Recall", f"{live_metrics['recall']:.3f}")
        m3.metric("F1", f"{live_metrics['f1']:.3f}")
        m4.metric("Flagged rows", live_metrics["flagged_count"])

        model_bytes = io.BytesIO()
        joblib.dump(best_model, model_bytes)
        st.download_button(
            f"⬇ Download trained {best_name} model (.pkl)",
            model_bytes.getvalue(),
            file_name="credit_risk_model.pkl",
            key="download_credit_model",
        )

    if st.session_state.training_history:
        with st.expander("📈 Training history this session"):
            st.dataframe(pd.DataFrame(st.session_state.training_history), use_container_width=True, hide_index=True)

with tab_fraud:
    st.subheader("Fraud / anomaly detection")
    st.caption("Works with any CSV — pick your target column below.")
    show_schema_help("data/samples/fraud_sample.csv", context="fraud")

    fraud_upload = st.file_uploader("CSV file", type="csv", key="fraud_upload")
    raw_fraud, source_note_fraud = load_dataset(fraud_upload, "data/samples/fraud_sample.csv")
    st.caption(source_note_fraud)

    fraud_columns = list(raw_fraud.columns)
    fraud_guessed_target = _guess_target_column(raw_fraud, "Class")
    fraud_target_default = fraud_columns.index(fraud_guessed_target)
    fraud_target_col = st.selectbox("Target column (fraud label)", fraud_columns, index=fraud_target_default, key="fraud_target")

    if raw_fraud[fraud_target_col].nunique() > 20:
        st.warning(
            f"'{fraud_target_col}' has {raw_fraud[fraud_target_col].nunique()} unique values — unusual "
            "for a classification target. Double check this is really the label column."
        )

    numeric_feature_cols = [
        c for c in fraud_columns if c != fraud_target_col and pd.api.types.is_numeric_dtype(raw_fraud[c])
    ]
    amount_default = "Amount" if "Amount" in numeric_feature_cols else (numeric_feature_cols[0] if numeric_feature_cols else None)
    amount_col = st.selectbox(
        "Column to reference in flag explanations (e.g. transaction amount)",
        numeric_feature_cols,
        index=numeric_feature_cols.index(amount_default) if amount_default in numeric_feature_cols else 0,
        key="fraud_amount_col",
    ) if numeric_feature_cols else None

    if st.button("Train fraud model"):
        y = raw_fraud[fraud_target_col]
        X = raw_fraud[numeric_feature_cols]

        if y.nunique() < 2:
            st.error(f"'{fraud_target_col}' only has one value present. A classifier needs both outcomes to learn anything.")
            st.stop()
        if X.shape[1] == 0:
            st.error("No numeric feature columns found besides the target.")
            st.stop()

        with st.spinner("Training and comparing Logistic Regression, Random Forest, and XGBoost..."):
            try:
                comparison_df, fitted = train_and_compare(X, y)
            except ValueError as e:
                st.error(str(e))
                st.stop()

        st.markdown("#### Model comparison")
        st.dataframe(comparison_df.style.format({"roc_auc": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}"}), use_container_width=True, hide_index=True)

        best_name = comparison_df.iloc[0]["model"]
        best_model, X_test, y_test, y_proba = fitted[best_name]
        st.caption(f"Best model by ROC-AUC: **{best_name}**. Details below are for this model.")

        st.session_state.training_history.append(
            {"tab": "Fraud detection", "model": best_name, "roc_auc": comparison_df.iloc[0]["roc_auc"], "rows": len(raw_fraud)}
        )

        render_feature_importance(feature_importance_df(best_model, list(X_test.columns)), best_name)

        st.markdown("#### Adjust the flagging threshold")
        threshold = st.slider("Classification threshold", 0.05, 0.95, 0.5, 0.05, key="fraud_threshold")
        pr_df = precision_recall_at_thresholds(y_test, y_proba)
        render_threshold_tradeoff(pr_df, threshold)
        live_metrics = metrics_at_threshold(y_test, y_proba, threshold)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{live_metrics['precision']:.3f}")
        m2.metric("Recall", f"{live_metrics['recall']:.3f}")
        m3.metric("F1", f"{live_metrics['f1']:.3f}")
        m4.metric("Flagged rows", live_metrics["flagged_count"])

        if amount_col:
            st.markdown("#### Sample flag explanation")
            from fraud.explain import explain_flag, top_contributing_features

            row = X.iloc[0]
            top_feats = top_contributing_features(best_model, row) if hasattr(best_model, "feature_importances_") else {}
            if top_feats:
                st.json(top_feats)
                st.write(explain_flag(row[amount_col], top_feats))
            else:
                st.caption(f"{best_name} doesn't expose feature importances, so a per-row explanation isn't available for it.")

        model_bytes = io.BytesIO()
        joblib.dump(best_model, model_bytes)
        st.download_button(
            f"⬇ Download trained {best_name} model (.pkl)",
            model_bytes.getvalue(),
            file_name="fraud_model.pkl",
            key="download_fraud_model",
        )

    if st.session_state.training_history:
        with st.expander("📈 Training history this session"):
            st.dataframe(pd.DataFrame(st.session_state.training_history), use_container_width=True, hide_index=True)

with tab_fairness:
    st.subheader("Fairness audit")
    st.caption(
        "Upload any dataset, pick your target column, and pick a column to audit for "
        "subgroup fairness (a real demographic attribute if you have one, or a documented proxy)."
    )
    show_schema_help("data/samples/credit_risk_sample.csv", context="fairness")

    fairness_upload = st.file_uploader("CSV file", type="csv", key="fairness_upload")
    raw_fair, source_note_fair = load_dataset(fairness_upload, "data/samples/credit_risk_sample.csv")
    st.caption(source_note_fair)

    fair_choices = column_picker(raw_fair, context="fairness", default_target="loan_status")

    fair_columns = list(raw_fair.columns)
    default_group_col = "home_ownership" if "home_ownership" in fair_columns else fair_columns[0]
    group_col = st.selectbox(
        "Column to audit for fairness (protected attribute or proxy)",
        fair_columns,
        index=fair_columns.index(default_group_col),
        key="fairness_group_col",
    )

    if st.button("Run fairness audit"):
        y_true = build_target_generic(raw_fair, fair_choices["target_col"], fair_choices["positive_value"])

        if y_true.nunique() < 2:
            st.error(f"'{fair_choices['target_col']}' only has one value present after filtering.")
            st.stop()

        X_structured = build_features_generic(raw_fair, fair_choices["categorical_cols"], fair_choices["numeric_cols"])
        X_llm = add_llm_features_generic(raw_fair, fair_choices["text_cols"])
        X = pd.concat([X_structured.reset_index(drop=True), X_llm.reset_index(drop=True)], axis=1)

        if X.shape[1] == 0:
            st.error("No feature columns selected. Pick at least one categorical or numeric column above.")
            st.stop()

        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
        clf.fit(X, y_true)
        y_pred = pd.Series(clf.predict(X), index=raw_fair.index)

        report = generate_report(y_true, y_pred, raw_fair[group_col], group_label=group_col)
        st.markdown(report)
        st.download_button(
            "⬇ Download this fairness report (.md)",
            report,
            file_name=f"fairness_report_{group_col}.md",
            key="download_fairness_report",
        )
