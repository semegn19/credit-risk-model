import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.transformers import DataFrameTransformer


class AggregateFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, customer_col="CustomerId"):
        self.customer_col = customer_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        customer_df = (
            X.groupby(self.customer_col)
            .agg(
                Total_Transaction_Amount=("Amount", "sum"),
                Average_Transaction_Amount=("Amount", "mean"),
                Transaction_Count=("TransactionId", "count"),
                Std_Transaction_Amount=("Amount", "std"),
                Max_Transaction_Amount=("Amount", "max"),
                Min_Transaction_Amount=("Amount", "min"),
                Total_Transaction_Value=("Value", "sum"),
                CountryCode=("CountryCode", "first"),
                CurrencyCode=("CurrencyCode", "first"),
                ProviderId=("ProviderId", "first"),
                ProductCategory=("ProductCategory", "first"),
                ChannelId=("ChannelId", "first"),
                PricingStrategy=("PricingStrategy", "first")
            )
            .reset_index()
        )

        customer_df["Std_Transaction_Amount"] = (
            customer_df["Std_Transaction_Amount"].fillna(0)
        )

        return customer_df


class DateFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, datetime_col="TransactionStartTime"):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df = X.copy()

        df[self.datetime_col] = pd.to_datetime(
            df[self.datetime_col],
            errors="coerce",
            utc=True
        )

        df["Transaction_Hour"] = df[self.datetime_col].dt.hour
        df["Transaction_Day"] = df[self.datetime_col].dt.day
        df["Transaction_Month"] = df[self.datetime_col].dt.month
        df["Transaction_Year"] = df[self.datetime_col].dt.year
        df["Transaction_Weekday"] = df[self.datetime_col].dt.weekday
        df["Transaction_Is_Weekend"] = (
            df["Transaction_Weekday"] >= 5
        ).astype(int)

        return df


def build_pipeline(df):

    temp_df = AggregateFeatures().fit_transform(df)

    categorical = (
        temp_df.select_dtypes(include="object")
        .columns
        .tolist()
    )

    numerical = (
        temp_df.select_dtypes(include=["int64", "float64"])
        .columns
        .tolist()
    )

    if "CustomerId" in categorical:
        categorical.remove("CustomerId")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical),
        ("cat", categorical_pipeline, categorical)
    ])

    return Pipeline([
        ("aggregate_features", AggregateFeatures()),
        ("preprocessor", DataFrameTransformer(preprocessor))
    ])