import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def calculate_rfm(df):

    df = df.copy()

    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"],
        utc=True
    )

    snapshot = (
        df["TransactionStartTime"].max()
        + pd.Timedelta(days=1)
    )

    return (
        df.groupby("CustomerId")
        .agg(
            Recency=(
                "TransactionStartTime",
                lambda x: (snapshot - x.max()).days
            ),
            Frequency=("TransactionId", "count"),
            Monetary=("Value", "sum")
        )
        .reset_index()
    )


def cluster_customers(rfm):

    scaler = StandardScaler()

    scaled = rfm.copy()

    scaled["Frequency"] = np.log1p(scaled["Frequency"])
    scaled["Monetary"] = np.log1p(scaled["Monetary"])

    features = scaler.fit_transform(
        scaled[["Recency", "Frequency", "Monetary"]]
    )

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    rfm["Cluster"] = model.fit_predict(features)

    return rfm


def assign_high_risk_label(rfm):

    summary = (
        rfm.groupby("Cluster")
        .agg(
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean")
        )
    )

    summary["engagement_score"] = (
        summary["Frequency"]
        + summary["Monetary"]
    )

    high_risk = summary["engagement_score"].idxmin()

    rfm["is_high_risk"] = (
        rfm["Cluster"] == high_risk
    ).astype(int)

    return rfm