import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src.training.evaluation import evaluate_model
from src.training.mlflow_utils import log_metrics


def train_logistic(X_train, X_test, y_train, y_test):

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    grid = GridSearchCV(
        model,
        {"C": [0.1, 1, 10]},
        scoring="roc_auc",
        cv=3
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    metrics = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    mlflow.log_params(grid.best_params_)
    log_metrics(metrics)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="CreditRiskModel"
    )

    return best_model, metrics


def train_random_forest(X_train, X_test, y_train, y_test):

    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    grid = GridSearchCV(
        model,
        {
            "n_estimators": [100],
            "max_depth": [5, 10]
        },
        scoring="roc_auc",
        cv=3
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    metrics = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    mlflow.log_params(grid.best_params_)
    log_metrics(metrics)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="CreditRiskModel"
    )

    return best_model, metrics