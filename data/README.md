[README.md](https://github.com/user-attachments/files/30259779/README.md)
# Data

This project does not commit full datasets to the repository — only small,
runnable samples are checked into git. Full datasets are downloaded locally
via the provided script and are gitignored.

## Why datasets aren't committed here

- **Size:** the full fraud dataset alone is ~150 MB as a single CSV, over
  GitHub's file size limits for a normal commit, and large binary/data files
  bloat repo history permanently even if later deleted.
- **Licensing:** the source datasets are hosted on Kaggle under Kaggle's own
  terms. Downloading and using them for this project is fine; re-hosting the
  raw files in a separate public repo is not, since redistribution rights
  belong to the original dataset owners.
- **Convention:** most ML repositories ship a download script plus a small
  sample, not the raw data, so contributors and reviewers can run the code
  immediately without a large download, and the full data stays sourced from
  its original, licensed location.

## What's here

```
data/
├── samples/
│   ├── credit_risk_sample.csv   # ~300 rows, same schema as the full dataset
│   └── fraud_sample.csv         # ~500 rows, same schema as the full dataset
├── full/                        # created by download_data.py, gitignored
├── download_data.py
└── README.md
```

The sample files are **synthetic** — generated to match the column names,
types, and rough statistical shape of the real datasets, so code can be
developed and tested against them immediately. They are not a substitute for
the real data when it comes to reporting actual model results; train and
evaluate on the full downloaded datasets before writing up any numbers in the
main README.

## Getting the full datasets

### 1. Credit risk: LendingClub loan data

- Source: search "lending club loan data" on [Kaggle](https://www.kaggle.com/datasets)
  (several mirrors exist; pick one with the full accepted-loans file, which
  includes `loan_status`, `loan_amnt`, `int_rate`, `purpose`, `emp_title`, etc.)
- Expected columns used by this project: `loan_amnt`, `term`, `int_rate`,
  `installment`, `grade`, `emp_title`, `emp_length`, `home_ownership`,
  `annual_inc`, `verification_status`, `purpose`, `dti`, `delinq_2yrs`,
  `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`, `loan_status`
- Target column: `loan_status` (binary: default/charged-off vs. fully paid)

### 2. Fraud detection: credit card fraud dataset

- Source: [mlg-ulb/creditcardfraud on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Columns: `Time`, `V1`–`V28` (PCA-transformed features), `Amount`, `Class`
- Target column: `Class` (1 = fraud, 0 = genuine)
- Note: this dataset is highly imbalanced (~0.17% fraud) — account for this
  in evaluation (use precision/recall/AUC, not raw accuracy).

### Download automatically

```bash
pip install kagglehub
python data/download_data.py
```

This authenticates against your Kaggle account (kagglehub will prompt on
first run) and saves both datasets into `data/full/`, which is gitignored.

### Download manually

If you'd rather not use the API, download the CSVs directly from the Kaggle
pages above and place them at:

```
data/full/credit_risk/<file>.csv
data/full/fraud/creditcard.csv
```

## Demographic / fairness audit columns

Neither dataset above includes a clean protected-attribute column out of the
box. For the fairness audit module, either:

- use a column that acts as a reasonable proxy (e.g. `home_ownership` or
  geographic region, if present, for the credit risk data), clearly labeled
  as a proxy rather than a true demographic attribute, or
- join in a synthetic demographic column for demonstration purposes, clearly
  documented as synthetic in the audit report so results aren't mistaken for
  a real bias finding.

Document whichever choice you make directly in `fairness/audit.py` and in
the final results write-up, so reviewers know exactly what was tested.
