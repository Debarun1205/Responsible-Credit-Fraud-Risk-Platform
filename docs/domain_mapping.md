[domain_mapping.md](https://github.com/user-attachments/files/30222339/domain_mapping.md)
# Domain mapping: AI, ML, and DS in this project

This document explains, module by module, which parts of the platform fall under artificial intelligence, machine learning, and data science — and why. The boundary between these fields is not sharp (LLMs are technically machine learning too), but the split below reflects a practical, defensible distinction: **AI** here refers specifically to the LLM/agentic components, **ML** refers to the classical statistical models trained on structured data, and **DS** refers to the analysis, evaluation, and reporting layer that turns model output into a decision-ready result.

## Summary table

| Module | AI | ML | DS |
|---|---|---|---|
| EDA agent | Agent plans and executes its own analysis via tool use | — | Profiling output (missingness, distributions, correlations) is a DS deliverable |
| Credit risk model | Text-to-feature extraction via Claude | Classifier training and evaluation (ROC-AUC) | Feature engineering and result interpretation |
| Fraud detection | Natural-language explanation of flags | Anomaly/classification model training and evaluation | — |
| Fairness audit | — | Subgroup rate computation is a model-evaluation task | Statistical significance testing (FDR control) and bias reporting |

## Module-by-module detail

### EDA agent

**AI component.** The agent is given a raw dataset and a general instruction ("explore this dataset"), and it decides for itself which analyses to run, in what order, using Claude's tool-use capability to call profiling functions. This is agentic behavior — the model is doing planning and tool selection, not just generating text.

**DS component.** The actual outputs of the agent — missingness reports, distribution summaries, correlation matrices — are standard data science deliverables. The agent automates a DS workflow rather than replacing it.

There is no classical ML in this module. No model is trained here; nothing is predicted.

### Credit risk model

**AI component.** Loan purpose text, employment titles, and other free-text fields are passed to Claude, which extracts structured signals (e.g. an inferred risk category, sentiment, or specificity of stated purpose). This is an NLP/LLM application layered in front of the model — it produces features, not predictions.

**ML component.** A classifier (logistic regression, random forest, or gradient boosting) is trained on structured fields plus the LLM-derived features to predict probability of default. Model selection, hyperparameter tuning, and evaluation via ROC-AUC are classical supervised learning.

**DS component.** Deciding which structured features to engineer, interpreting which predictors matter (contract timing, price sensitivity, etc.), and communicating the baseline-vs-augmented comparison honestly is a data science task layered around the ML core.

### Fraud detection model

**AI component.** After the model flags a transaction, Claude generates a natural-language explanation of why it looks suspicious, referencing the specific feature values that triggered the flag. This is a generative AI application consuming the ML model's output — it does not affect the model's decision, only its explainability.

**ML component.** The core detection model (isolation forest, autoencoder, or supervised classifier if labels are available) is trained and evaluated using precision, recall, and ROC-AUC — standard anomaly detection or classification methodology.

The LLM feature extractor from the credit risk module can be reused here for any text fields in the transactions dataset (e.g. merchant category descriptions), which is why `llm_features.py` is shared rather than duplicated.

### Fairness audit

**ML component.** Computing false-positive and false-negative rates for each demographic subgroup, given a model's predictions and true labels, is a model-evaluation task grounded in standard classification metrics.

**DS component.** The audit goes beyond raw rate differences: it applies statistical testing across all non-trivial subgroup combinations while controlling the false discovery rate (FDR), so that a flagged disparity is a real finding rather than noise from testing many subgroups at once. Producing the resulting report — in a form a non-technical reviewer could act on — is a data science communication task.

There is no LLM/AI component in this module by design. Fairness testing benefits from being fully deterministic and reproducible, so it is kept as classical statistics rather than routed through a model that could introduce its own inconsistency.

## Honest caveat

This mapping is a communication tool, not a claim that these are three technically disjoint fields. Large language models are, underneath, a machine learning technique; the "AI" label here is used narrowly to mean the LLM-specific, agentic, and generative components of the system, to make the project's scope legible to someone scanning a resume or README rather than to assert a formal taxonomy.
