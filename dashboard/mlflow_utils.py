import mlflow
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "credit-risk-model"


def get_latest_metrics():
    """
    Returns metrics from the latest MLflow run.
    """

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        return None

    run = runs[0]

    return {
        "Accuracy": run.data.metrics.get("accuracy"),
        "Precision": run.data.metrics.get("precision"),
        "Recall": run.data.metrics.get("recall"),
        "F1 Score": run.data.metrics.get("f1_score"),
        "ROC-AUC": run.data.metrics.get("roc_auc"),
    }