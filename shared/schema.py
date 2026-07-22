"""
Shared column-name constants so modules referencing the same datasets don't
drift out of sync with each other.
"""

# credit_risk_sample.csv / full LendingClub-style data
CREDIT_TEXT_COLUMNS = ["emp_title", "purpose"]
CREDIT_TARGET = "loan_status"
CREDIT_TARGET_POSITIVE = "Charged Off"  # the "bad" outcome we're predicting

# fraud_sample.csv / full creditcardfraud-style data
FRAUD_TARGET = "Class"
FRAUD_AMOUNT_COL = "Amount"
