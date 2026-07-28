import pandas as pd


from src.config.settings import settings

from src.features.feature_engineering import (
    AggregateFeatures,
    build_pipeline
)

from src.features.target_engineering import (
    calculate_rfm,
    cluster_customers,
    assign_high_risk_label
)


def main():

    df = pd.read_csv(settings.RAW_DATA_PATH)

    pipeline = build_pipeline(df)

    processed = pipeline.fit_transform(df)

    processed["CustomerId"] = (
        AggregateFeatures()
        .fit_transform(df)["CustomerId"]
        .values
    )

    rfm = calculate_rfm(df)
    rfm = cluster_customers(rfm)
    rfm = assign_high_risk_label(rfm)

    processed = processed.merge(
        rfm[["CustomerId", "is_high_risk"]],
        on="CustomerId",
        how="left"
    )

    processed = processed.drop(columns="CustomerId")

    processed = processed.drop_duplicates()

    processed.to_csv(settings.PROCESSED_DATA_PATH, index=False)

    print(f"Saved processed dataset to {settings.PROCESSED_DATA_PATH}")
    print(processed.shape)


if __name__ == "__main__":
    main()