import requests

API_URL = "http://127.0.0.1:8000/predict"


def predict(data: dict) -> dict:
    response = requests.post(API_URL, json=data)
    response.raise_for_status()
    return response.json()