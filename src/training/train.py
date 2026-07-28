import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split

from src.config.settings import settings


from src.training.models import (
    train_logistic,
    train_random_forest
)

from src.training.mlflow_utils import setup_mlflow

from src.training.evaluation import run_data_checks


def main():

    df = pd.read_csv(settings.PROCESSED_DATA_PATH)

    run_data_checks(df)

    X = df.drop(columns=[settings.TARGET_COLUMN])

    if "num__FraudResult" in X.columns:
        X = X.drop(columns=["num__FraudResult"])

    y = df[settings.TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        random_state=42,
        test_size=0.20
    )

    setup_mlflow()

    best_auc = -1
    best_model = ""

    with mlflow.start_run(run_name="LogisticRegression"):

        best_model, metrics = train_logistic(
            X_train,
            X_test,
            y_train,
            y_test
        )

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model = "Logistic Regression"

    with mlflow.start_run(run_name="RandomForest"):

        best_model, metrics = train_random_forest(
            X_train,
            X_test,
            y_train,
            y_test
        )

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model = "Random Forest"

    print(f"\nBest Model: {best_model}")
    print(f"ROC-AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()