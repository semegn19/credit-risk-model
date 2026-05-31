import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans


# ============================================================
# TASK 3 COMPONENTS
# ============================================================

class AggregateFeatures(BaseEstimator, TransformerMixin):

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
            customer_agg["Std_Transaction_Amount"]
            .fillna(0)
        )

        return df.merge(
            customer_agg,
            on=self.customer_col,
            how="left"
        )


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

        columns = self.preprocessor.get_feature_names_out()

        return pd.DataFrame(
            transformed,
            columns=columns,
            index=X.index
        )


# ============================================================
# BUILD TASK 3 PIPELINE
# ============================================================

def build_pipeline(df):

    temp_df = AggregateFeatures().fit_transform(df)
    temp_df = DateFeatureExtractor().fit_transform(temp_df)

    columns_to_drop = [
        "TransactionId",
        "BatchId",
        "AccountId",
        "SubscriptionId",
        "TransactionStartTime"
        # CustomerId intentionally retained
    ]

    temp_df = temp_df.drop(columns=columns_to_drop)

    categorical_features = (
        temp_df.select_dtypes(include=["object"])
        .columns
        .tolist()
    )

    numerical_features = (
        temp_df.select_dtypes(include=["int64", "float64"])
        .columns
        .tolist()
    )

    # Remove CustomerId from encoding
    categorical_features.remove("CustomerId")

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

    pipeline = Pipeline([
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

    return pipeline


# ============================================================
# TASK 4 - RFM CALCULATION
# ============================================================

def calculate_rfm(df):

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"],
        utc=True
    )

    snapshot_date = (
        df["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (
                    snapshot_date - x.max()
                ).days
            ),
            Frequency=("TransactionId", "count"),
            Monetary=("Value", "sum")
        )
        .reset_index()
    )

    return rfm


# ============================================================
# TASK 4 - CLUSTER CUSTOMERS
# ============================================================

def cluster_customers(rfm):

    scaler = StandardScaler()

    rfm_scaled = scaler.fit_transform(
        rfm[["Recency", "Frequency", "Monetary"]]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    return rfm


# ============================================================
# TASK 4 - IDENTIFY HIGH RISK CLUSTER
# ============================================================

def assign_high_risk_label(rfm):

    cluster_summary = (
        rfm.groupby("Cluster")
        .agg({
            "Recency": "mean",
            "Frequency": "mean",
            "Monetary": "mean"
        })
    )

    print("\nCluster Summary")
    print(cluster_summary)

    cluster_summary["risk_score"] = (
    cluster_summary["Recency"].rank(ascending=False)
    + cluster_summary["Frequency"].rank(ascending=True)
    + cluster_summary["Monetary"].rank(ascending=True)
)

    high_risk_cluster = cluster_summary["risk_score"].idxmax()

    rfm["is_high_risk"] = (
        rfm["Cluster"] == high_risk_cluster
    ).astype(int)

    print(
        f"\nHigh Risk Cluster: {high_risk_cluster}"
    )

    return rfm


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    df = pd.read_csv(
        "C:/Users/hp/credit-risk-model/data/raw/data.csv"
    )

    # --------------------------------------------------------
    # Task 3 processing
    # --------------------------------------------------------

    pipeline = build_pipeline(df)

    processed_df = pipeline.fit_transform(df)

    # Keep CustomerId for merge
    processed_df["CustomerId"] = df["CustomerId"].values

    # --------------------------------------------------------
    # Task 4 target engineering
    # --------------------------------------------------------

    rfm = calculate_rfm(df)

    rfm = cluster_customers(rfm)

    rfm = assign_high_risk_label(rfm)

    # --------------------------------------------------------
    # Merge target back
    # --------------------------------------------------------

    processed_df = processed_df.merge(
        rfm[["CustomerId", "is_high_risk"]],
        on="CustomerId",
        how="left"
    )

    # Remove CustomerId before modeling
    processed_df = processed_df.drop(
        columns=["CustomerId"]
    )

    # --------------------------------------------------------
    # Save final dataset
    # --------------------------------------------------------

    processed_df.to_csv(
        "C:/Users/hp/credit-risk-model/data/processed/processed_data_with_target.csv",
        index=False
    )

    print(
        "\nFinal dataset shape:",
        processed_df.shape
    )

    print(
        "\nTarget distribution:"
    )

    print(
        processed_df["is_high_risk"]
        .value_counts(normalize=True)
    )