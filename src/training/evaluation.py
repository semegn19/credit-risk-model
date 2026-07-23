import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = predictions

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities)
    }


def run_data_checks(df):

    suspicious = [
        "Recency",
        "Frequency",
        "Monetary",
        "Cluster",
        "FraudResult",
        "num__FraudResult"
    ]

    found = [c for c in suspicious if c in df.columns]

    print("\nSuspicious Columns:", found if found else "None")

    print(f"\nDuplicate Rows: {df.duplicated().sum()}")

    numeric = df.select_dtypes(include=["int64", "float64"]).columns

    corr = (
        df[numeric]
        .corr()["is_high_risk"]
        .sort_values(key=abs, ascending=False)
    )

    print("\nTop Correlations")
    print(corr.head(15))