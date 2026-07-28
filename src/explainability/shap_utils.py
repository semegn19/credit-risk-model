import shap
import pandas as pd


class CreditRiskExplainer:

    def __init__(self, model):

        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X):

        return self.explainer(X)

    def feature_importance(self, X):

        values = self.explainer(X)

        importance = pd.DataFrame({

            "feature": X.columns,

            "importance":
                abs(values.values).mean(axis=0)

        })

        return importance.sort_values(
            "importance",
            ascending=False
        )

    def single_prediction(self, X):

        values = self.explainer(X)

        return values[0]