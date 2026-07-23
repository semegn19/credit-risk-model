import pandas as pd

from src.features.target_engineering import (
    calculate_rfm,
    cluster_customers,
    assign_high_risk_label,
)


def sample_transactions():
    return pd.DataFrame(
        {
            "CustomerId": [
                "C1",
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
            ],
            "TransactionId": [1, 2, 3, 4, 5, 6],
            "TransactionStartTime": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-06",
            ],
            "Value": [100, 200, 300, 400, 500, 600],
        }
    )


def test_calculate_rfm():
    df = sample_transactions()

    rfm = calculate_rfm(df)

    assert len(rfm) == 5
    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns


def test_cluster_customers():
    df = sample_transactions()

    rfm = calculate_rfm(df)
    clustered = cluster_customers(rfm)

    assert "Cluster" in clustered.columns
    assert clustered["Cluster"].nunique() <= 3


def test_assign_high_risk():
    df = sample_transactions()

    rfm = calculate_rfm(df)
    rfm = cluster_customers(rfm)
    rfm = assign_high_risk_label(rfm)

    assert "is_high_risk" in rfm.columns
    assert set(rfm["is_high_risk"].unique()).issubset({0, 1})