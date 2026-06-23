"""
tests/test_main.py

Unit tests for the FastAPI app: one GET test, two POST tests
(verified to exercise both possible prediction classes, not the
same class twice).
"""
from fastapi.testclient import TestClient
from main import app


def test_get_root():
    """
    GET / should return 200 and a welcome message.
    """
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_post_predict_high_income():
    """
    POST /predict on profile that should predict '>50k'.

    Profile verified directly against the trained model artifacts
    (model/model.pkl, encoder.pkl, lb.pkl) before being added here,
    rather than guessed.
    """
    payload = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 178356,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"


def test_post_predict_low_income():
    """
    POST /predict on a profile that should predict '<=50k'.

    Profile verified directly against the trained model artifacts
    before being added here, and delibrately distinct in outcome
    from test_rpost_predict_high_income.
    """
    payload = {
        "age": 19,
        "workclass": "Private",
        "fnlgt": 200000,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Handlers-cleaners",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States",
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"
