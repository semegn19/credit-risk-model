import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# AGGREGATE CUSTOMER FEATURES
# ============================================================

class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Create customer-level aggregate transaction features.
    """

    def __init__(self, customer_col="CustomerId"):
        self.customer_col = customer_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        customer_agg = (
            df.groupby(self.customer_col)
            .agg(
                Total_Transaction_Amount=("Amount", "sum"),
                Average_Transaction_Amount=("Amount", "mean"),
                Transaction_Count=("TransactionId", "count"),
                Std_Transaction_Amount=("Amount", "std"),
                Max_Transaction_Amount=("Amount", "max"),
                Min_Transaction_Amount=("Amount", "min"),
                Total_Transaction_Value=("Value", "sum")
            )
            .reset_index()
        )

        customer_agg["Std_Transaction_Amount"] = (
            customer_agg["Std_Transaction_Amount"].fillna(0)
        )

        df = df.merge(customer_agg, on=self.customer_col, how="left")

        return df


# ============================================================
# DATETIME FEATURES
# ============================================================

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract datetime-related features.
    """

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


# ============================================================
# COLUMN DROPPER
# ============================================================

class ColumnDropper(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors="ignore")


# ============================================================
# DATAFRAME OUTPUT WRAPPER
# ============================================================

class DataFrameTransformer(BaseEstimator, TransformerMixin):
    """
    Convert ColumnTransformer output back into DataFrame.
    """

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def fit(self, X, y=None):
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        transformed = self.preprocessor.transform(X)

        feature_names = self.preprocessor.get_feature_names_out()

        return pd.DataFrame(
            transformed,
            columns=feature_names,
            index=X.index
        )


# ============================================================
# BUILD PIPELINE
# ============================================================

def build_pipeline(df):

    temp_df = AggregateFeatures().fit_transform(df)
    temp_df = DateFeatureExtractor().fit_transform(temp_df)

    columns_to_drop = [
        "TransactionId",
        "BatchId",
        "AccountId",
        "SubscriptionId",
        "CustomerId",
        "TransactionStartTime"
    ]

    temp_df = temp_df.drop(columns=columns_to_drop)

    categorical_features = temp_df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_features = temp_df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

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
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    full_pipeline = Pipeline([
        ("aggregate_features", AggregateFeatures()),
        ("date_features", DateFeatureExtractor()),
        (
            "drop_columns",
            ColumnDropper(columns_to_drop)
        ),
        (
            "preprocessor",
            DataFrameTransformer(preprocessor)
        )
    ])

    return full_pipeline


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Load raw transaction data
    df = pd.read_csv("C:/Users/hp/credit-risk-model/data/raw/data.csv")


    # Build pipeline
    pipeline = build_pipeline(df)

    # Fit and transform
    processed_df = pipeline.fit_transform(df)

    print("Pipeline completed successfully.")
    print(f"Processed shape: {processed_df.shape}")

    # Save processed dataset
    processed_df.to_csv(
        "C:/Users/hp/credit-risk-model/data/processed/processed_data.csv",
        index=False
    )

    print("Processed dataset saved.")