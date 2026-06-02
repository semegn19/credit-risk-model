from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd

from src.api.pydantic_models import PredictionInput, PredictionOutput

# =====================================================
# LOAD MODEL FROM MLFLOW REGISTRY
# =====================================================

MODEL_NAME = "CreditRiskModel"
MODEL_STAGE = "Latest"   # or "Production"

model = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}/{MODEL_STAGE}"
)

# =====================================================
# INIT FASTAPI
# =====================================================

app = FastAPI(
    title="Credit Risk API",
    version="1.0"
)

# =====================================================
# PREDICT ENDPOINT
# =====================================================

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data.dict()])

    # Get probability
    prob = model.predict(input_df)

    # If model returns array, extract value
    risk_prob = float(prob[0])

    # Convert to class (threshold = 0.5)
    risk_label = int(risk_prob >= 0.5)

    return PredictionOutput(
        risk_probability=risk_prob,
        risk_label=risk_label
    )