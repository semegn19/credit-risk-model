import mlflow.sklearn

from src.config.settings import settings


def load_model():
    return mlflow.sklearn.load_model(
        f"models:/{settings.MODEL_NAME}/{settings.MODEL_STAGE}"
    )