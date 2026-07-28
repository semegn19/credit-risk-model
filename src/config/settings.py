from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    RAW_DATA_PATH: str = "data/raw/data.csv"
    PROCESSED_DATA_PATH: str = "data/processed/processed_data_with_target.csv"

    TARGET_COLUMN: str = "is_high_risk"

    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.20
    N_CLUSTERS: int = 3

    MLFLOW_EXPERIMENT: str = "credit-risk-model"
    MODEL_NAME: str = "CreditRiskModel"
    MODEL_STAGE: str = "Latest"

    DEFAULT_THRESHOLD: float = 0.50


settings = Config()