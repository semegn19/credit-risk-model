
import pandas as pd

from src.features.feature_engineering import (
    AggregateFeatures,
    DateFeatureExtractor,
    build_pipeline,
)


def sample_transactions():
    return pd.DataFrame(
        {
            "CustomerId": ["C1", "C1", "C2"],
            "TransactionId": [1, 2, 3],
            "Amount": [100, 200, 300],
            "Value": [100, 200, 300],
            "CountryCode": [256, 256, 256],
            "CurrencyCode": ["UGX", "UGX", "UGX"],
            "ProviderId": ["ProviderId_1"] * 3,
            "ProductCategory": ["airtime"] * 3,
            "ChannelId": ["ChannelId_1"] * 3,
            "PricingStrategy": [2, 2, 2],
            "TransactionStartTime": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ],
        }
    )


def test_aggregate_features():
    df = sample_transactions()

    result = AggregateFeatures().fit_transform(df)

    assert len(result) == 2
    assert "Total_Transaction_Amount" in result.columns

    c1 = result[result.CustomerId == "C1"].iloc[0]

    assert c1["Transaction_Count"] == 2
    assert c1["Total_Transaction_Amount"] == 300


def test_date_feature_extractor():
    df = sample_transactions()

    result = DateFeatureExtractor().fit_transform(df)

    assert "Transaction_Hour" in result.columns
    assert "Transaction_Month" in result.columns
    assert "Transaction_Is_Weekend" in result.columns


def test_build_pipeline_returns_dataframe():
    df = sample_transactions()

    pipeline = build_pipeline(df)

    transformed = pipeline.fit_transform(df)

    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape[0] == 2