from fastapi.testclient import TestClient
import json
from main import app
from src.ml.training import load_config

configs = load_config()

client = TestClient(app)


data_0 = {
    "age": 18,
    "workclass": "Local_gov",
    "fnlgt": 1.794650e05,
    "education": "10th",
    "education_num": 8,
    "marital_status": "Never_married",
    "occupation": "Farming_fishing",
    "relationship": "Not_in_family",
    "race": "Black",
    "sex": "Female",
    "capital_gain": 100,
    "capital_loss": 50,
    "hours_per_week": 0,
    "native_country": "Mexico",
}

data_1 = {
    "age": 50,
    "workclass": "Private",
    "fnlgt": 1.880050e05,
    "education": "Masters",
    "education_num": 12,
    "marital_status": "Married_civ_spouse",
    "occupation": "Exec_managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 40174,
    "capital_loss": 195,
    "hours_per_week": 50,
    "native_country": "United_States",
}


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Model API!"}


def test_predict_income_valid_0():
    response = client.post("/predict", json=data_0)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "label" in response.json()
    assert response.json()["prediction"] == "0"


def test_predict_income_valid_1():
    response = client.post("/predict", json=data_1)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "label" in response.json()
    assert response.json()["prediction"] == "1"


def test_predict_income_numerical_invalid():
    # Replace this with invalid data for your model
    for key in configs["model"]["num_features"]:
        data = data_1
        data[key] = -1
        response = client.post("/predict", json=data)
        assert response.status_code == 400


def test_predict_income_categorical_invalid():
    # Replace this with invalid data for your model
    for key in configs["model"]["cat_features"]:
        data = data_1
        data[key] = "invalid"
        response = client.post("/predict", json=data)
        assert response.status_code == 400
