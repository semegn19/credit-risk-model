from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def valid_payload():
    return {
        "num__Total_Transaction_Amount": 500000,
        "num__Average_Transaction_Amount": 25000,
        "num__Transaction_Count": 20,
        "num__Std_Transaction_Amount": 5000,
        "num__Max_Transaction_Amount": 80000,
        "num__Min_Transaction_Amount": 1000,
        "num__Total_Transaction_Value": 450000,
        "num__CountryCode": 256,
        "num__PricingStrategy": 2,
        "cat__CurrencyCode_UGX": 1,
        "cat__ProviderId_ProviderId_1": 0,
        "cat__ProviderId_ProviderId_2": 1,
        "cat__ProviderId_ProviderId_3": 0,
        "cat__ProviderId_ProviderId_4": 0,
        "cat__ProviderId_ProviderId_5": 0,
        "cat__ProviderId_ProviderId_6": 0,
        "cat__ProductCategory_airtime": 1,
        "cat__ProductCategory_data_bundles": 0,
        "cat__ProductCategory_financial_services": 0,
        "cat__ProductCategory_movies": 0,
        "cat__ProductCategory_other": 0,
        "cat__ProductCategory_ticket": 0,
        "cat__ProductCategory_transport": 0,
        "cat__ProductCategory_tv": 0,
        "cat__ProductCategory_utility_bill": 0,
        "cat__ChannelId_ChannelId_1": 1,
        "cat__ChannelId_ChannelId_2": 0,
        "cat__ChannelId_ChannelId_3": 0,
        "cat__ChannelId_ChannelId_5": 0,
    }


@patch("src.api.routes.load_model")
def test_predict_returns_probability(mock_loader):

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.82]
    mock_loader.return_value = mock_model

    response = client.post(
        "/predict",
        json=valid_payload()
    )

    assert response.status_code == 200

    body = response.json()

    assert body["risk_probability"] == 0.82
    assert body["risk_label"] == 1


@patch("src.api.routes.load_model")
def test_predict_low_risk(mock_loader):

    mock_model = MagicMock()
    mock_model.predict.return_value = [0.23]
    mock_loader.return_value = mock_model

    response = client.post(
        "/predict",
        json=valid_payload()
    )

    assert response.status_code == 200

    body = response.json()

    assert body["risk_probability"] == 0.23
    assert body["risk_label"] == 0


def test_invalid_request_returns_422():

    payload = valid_payload()

    payload.pop("num__Transaction_Count")

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


def test_negative_transaction_amount_returns_422():

    payload = valid_payload()

    payload["num__Total_Transaction_Amount"] = -10

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422