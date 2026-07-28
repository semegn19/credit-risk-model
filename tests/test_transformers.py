import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.features.transformers import (
    ColumnDropper,
    DataFrameTransformer,
)


def test_column_dropper():
    df = pd.DataFrame(
        {
            "A": [1, 2],
            "B": [3, 4],
            "C": [5, 6],
        }
    )

    result = ColumnDropper(["B"]).fit_transform(df)

    assert "B" not in result.columns
    assert result.shape[1] == 2


def test_dataframe_transformer():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [4, 5, 6],
        }
    )

    transformer = DataFrameTransformer(StandardScaler())

    result = transformer.fit_transform(df)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape