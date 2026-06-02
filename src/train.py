import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# =====================================================
# HELPER FUNCTIONS
# =====================================================

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


def log_metrics(metrics):
    for name, value in metrics.items():
        mlflow.log_metric(name, value)


# =====================================================
# MAIN TRAINING PIPELINE
# =====================================================

if __name__ == "__main__":

    DATA_PATH = (
        "C:/Users/hp/credit-risk-model/"
        "data/processed/processed_data_with_target.csv"
    )

    df = pd.read_csv(DATA_PATH)

    print("\nDataset Loaded")
    print("Shape:", df.shape)

    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory Usage: {memory_usage:.2f} MB")

    TARGET_COLUMN = "is_high_risk"
    DROP_COLS = [TARGET_COLUMN]

    if "num__FraudResult" in df.columns:
        DROP_COLS.append("num__FraudResult")

    X = df.drop(columns=DROP_COLS)
    y = df[TARGET_COLUMN]

    print("\nFeature Shape:", X.shape)
    print("\nTarget Distribution:")
    print(y.value_counts())

    # =========================
    # LEAKAGE CHECKS
    # =========================

    suspicious_cols = [
        "Recency",
        "Frequency",
        "Monetary",
        "Cluster",
        "FraudResult",
        "num__FraudResult"
    ]

    found_cols = [c for c in suspicious_cols if c in df.columns]

    print("\nSuspicious Columns:", found_cols if found_cols else "None")

    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates}")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    target_corr = (
        df[numeric_cols]
        .corr()["is_high_risk"]
        .sort_values(key=abs, ascending=False)
    )

    print("\nTop Correlations:")
    print(target_corr.head(15))

    print("\nTarget %:")
    print((df["is_high_risk"].value_counts(normalize=True) * 100).round(2))

    # =========================
    # TRAIN TEST SPLIT
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_hashes = pd.util.hash_pandas_object(X_train, index=False)
    test_hashes = pd.util.hash_pandas_object(X_test, index=False)

    overlap = len(set(train_hashes).intersection(set(test_hashes)))

    print(f"\nTrain-Test Overlap: {overlap}")

    # =========================
    # MLFLOW
    # =========================

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit-risk-model")

    best_auc = -1
    best_model_name = None

    # =========================
    # LOGISTIC REGRESSION
    # =========================

    with mlflow.start_run(run_name="LogisticRegression"):

        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )

        grid = GridSearchCV(
            model,
            param_grid={"C": [0.1, 1, 10]},
            scoring="roc_auc",
            cv=3,
            n_jobs=1
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        metrics = evaluate_model(best_model, X_test, y_test)

        mlflow.log_params(grid.best_params_)
        log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name="CreditRiskModel"
        )

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model_name = "LogisticRegression"

    # =========================
    # RANDOM FOREST
    # =========================

    with mlflow.start_run(run_name="RandomForest"):

        model = RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        )

        grid = GridSearchCV(
            model,
            param_grid={"n_estimators": [100], "max_depth": [5, 10]},
            scoring="roc_auc",
            cv=3,
            n_jobs=1
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        metrics = evaluate_model(best_model, X_test, y_test)

        mlflow.log_params(grid.best_params_)
        log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name="CreditRiskModel"
        )

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model_name = "RandomForest"

    # =========================
    # SUMMARY
    # =========================

    print("\nTraining Complete")
    print(f"Best Model: {best_model_name}")
    print(f"Best ROC-AUC: {best_auc:.4f}")
