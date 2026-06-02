import pandas as pd

from src.data_processing import (
    AggregateFeatures,
    DateFeatureExtractor
)


def test_aggregate_features():

    df = pd.DataFrame({
        "CustomerId": ["A", "A", "B"],
        "Amount": [100, 200, 300],
        "Value": [100, 200, 300],
        "TransactionId": [1, 2, 3],

        # ADD missing expected columns (dummy values)
        "ChannelId": ["C1", "C1", "C2"],
        "CountryCode": [256, 256, 256],
        "CurrencyCode": ["UGX", "UGX", "UGX"],
        "PricingStrategy": [1, 1, 1],
        "ProductCategory": ["airtime", "airtime", "data"],
        "ProviderId": ["P1", "P1", "P2"]
    })

    transformer = AggregateFeatures()

    result = transformer.fit_transform(df)

    assert "Total_Transaction_Amount" in result.columns
    assert "Transaction_Count" in result.columns


def test_date_features():

    df = pd.DataFrame({
        "TransactionStartTime": [
            "2024-01-15 10:30:00"
        ]
    })

    transformer = DateFeatureExtractor()

    result = transformer.fit_transform(df)

    assert "Transaction_Hour" in result.columns
    assert "Transaction_Month" in result.columns
