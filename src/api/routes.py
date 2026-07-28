import pandas as pd

from fastapi import APIRouter

from src.api.dependencies import load_model
from src.api.models import (
    PredictionInput,
    PredictionOutput,
)

router = APIRouter()

model = load_model()


@router.post(
    "/predict",
    response_model=PredictionOutput
)
def predict(input_data: PredictionInput):

    model = load_model()

    input_df = pd.DataFrame(
        [input_data.model_dump()]
    )

    probability = model.predict_proba(input_df)[0][1]

    return PredictionOutput(
        risk_probability=float(probability),
        risk_label=int(probability >= 0.5)
    )