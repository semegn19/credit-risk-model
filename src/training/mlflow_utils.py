import mlflow


def setup_mlflow():

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit-risk-model")


def log_metrics(metrics):

    for metric, value in metrics.items():
        mlflow.log_metric(metric, value)