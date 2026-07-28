from pathlib import Path

import mlflow
import pandas as pd
import shap


MODEL_NAME = "CreditRiskModel"
MODEL_STAGE = "Latest"


def load_model():
    return mlflow.pyfunc.load_model(
        f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    )


def load_training_data():
    return pd.read_csv(
        Path("data/processed/processed_data_with_target.csv")
    ).drop(columns=["is_high_risk"])


def get_underlying_model(pyfunc_model):
    """
    Extract sklearn model from MLflow wrapper.
    """
    return pyfunc_model._model_impl.sklearn_model


def compute_shap():

    pyfunc_model = mlflow.pyfunc.load_model(
        f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    )

    model = pyfunc_model._model_impl.sklearn_model

    X = pd.read_csv(
        "data/processed/processed_data_with_target.csv"
    ).drop(columns=["is_high_risk"])

    explainer = shap.Explainer(model, X)

    explanation = explainer(X, check_additivity=False)

    return explanation, X


