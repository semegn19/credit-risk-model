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
    """
    Calculate evaluation metrics.
    """

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
    """
    Log metrics to MLflow.
    """
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)


# =====================================================
# MAIN TRAINING PIPELINE
# =====================================================

if __name__ == "__main__":

    # =================================================
    # LOAD DATA
    # =================================================

    DATA_PATH = (
        "C:/Users/hp/credit-risk-model/"
        "data/processed/processed_data_with_target.csv"
    )

    df = pd.read_csv(DATA_PATH)

    print("\nDataset Loaded")
    print("Shape:", df.shape)

    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory Usage: {memory_usage:.2f} MB")

    # =================================================
    # FEATURES AND TARGET
    # =================================================

    TARGET_COLUMN = "is_high_risk"
    DROP_COLS = [TARGET_COLUMN]

    if "num__FraudResult" in df.columns:
        DROP_COLS.append("num__FraudResult")

    X = df.drop(columns=DROP_COLS)
    y = df[TARGET_COLUMN]

    print("\nFeature Matrix Shape:", X.shape)
    print("\nTarget Distribution:")
    print(y.value_counts())

    # =================================================
    # LEAKAGE CHECKS
    # =================================================

    print("\n================================")
    print("LEAKAGE DIAGNOSTICS")
    print("================================")

    suspicious_cols = [
        "Recency",
        "Frequency",
        "Monetary",
        "Cluster",
        "FraudResult",
        "num__FraudResult"
    ]

    found_cols = [col for col in suspicious_cols if col in df.columns]

    print("\nSuspicious Columns Found:")
    print(found_cols if found_cols else "None")

    if "num__FraudResult" in df.columns:
        print("\nFraudResult vs is_high_risk")

        fraud_crosstab = pd.crosstab(
            df["num__FraudResult"],
            df["is_high_risk"],
            normalize="index"
        )

        print(fraud_crosstab)

    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates}")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    target_corr = (
        df[numeric_cols]
        .corr()["is_high_risk"]
        .sort_values(key=abs, ascending=False)
    )

    print("\nTop Correlations with Target:")
    print(target_corr.head(15))

    print("\nTarget Percentage:")
    print(
        (df["is_high_risk"].value_counts(normalize=True) * 100).round(2)
    )

    print("\n================================\n")

    # =================================================
    # TRAIN TEST SPLIT
    # =================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("\n================================")
    print("TRAIN TEST DIAGNOSTICS")
    print("================================")

    print("Train Shape:", X_train.shape)
    print("Test Shape :", X_test.shape)

    print("\nTrain Target Distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTest Target Distribution:")
    print(y_test.value_counts(normalize=True))

    train_hashes = pd.util.hash_pandas_object(X_train, index=False)
    test_hashes = pd.util.hash_pandas_object(X_test, index=False)

    overlap = len(set(train_hashes).intersection(set(test_hashes)))

    print(f"\nIdentical Feature Rows Appearing in Both Train and Test: {overlap}")

    print("\n================================\n")

    # =================================================
    # MLFLOW SETUP
    # =================================================

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit-risk-model")

    best_overall_auc = -1
    best_model_name = None

    # =================================================
    # LOGISTIC REGRESSION
    # =================================================

    with mlflow.start_run(run_name="LogisticRegression"):

        print("\nTraining Logistic Regression...")

        logistic = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )

        grid_search = GridSearchCV(
            logistic,
            param_grid={"C": [0.1, 1, 10]},
            scoring="roc_auc",
            cv=3,
            n_jobs=1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        metrics = evaluate_model(best_model, X_test, y_test)

        mlflow.log_params(grid_search.best_params_)
        log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name="CreditRiskModel"
        )

        print("\nLogistic Regression Results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        if metrics["roc_auc"] > best_overall_auc:
            best_overall_auc = metrics["roc_auc"]
            best_model_name = "LogisticRegression"

    # =================================================
    # RANDOM FOREST
    # =================================================

    with mlflow.start_run(run_name="RandomForest"):

        print("\nTraining Random Forest...")

        rf = RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        )

        grid_search = GridSearchCV(
            rf,
            param_grid={"n_estimators": [100], "max_depth": [5, 10]},
            scoring="roc_auc",
            cv=3,
            n_jobs=1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        metrics = evaluate_model(best_model, X_test, y_test)

        mlflow.log_params(grid_search.best_params_)
        log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name="CreditRiskModel"
        )

        print("\nRandom Forest Results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        if metrics["roc_auc"] > best_overall_auc:
            best_overall_auc = metrics["roc_auc"]
            best_model_name = "RandomForest"

    # =================================================
    # SUMMARY
    # =================================================

    print("\n================================")
    print("TRAINING COMPLETE")
    print("================================")

    print(f"Best Model: {best_model_name}")
    print(f"Best ROC-AUC: {best_overall_auc:.4f}")
    