import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class ColumnDropper(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors="ignore")


class DataFrameTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def fit(self, X, y=None):
        self.preprocessor.fit(X)
        return self

    def transform(self, X):

        transformed = self.preprocessor.transform(X)

        return pd.DataFrame(
            transformed,
            columns=self.preprocessor.get_feature_names_out(),
            index=X.index
        )