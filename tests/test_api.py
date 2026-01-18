"""
Basic API tests
"""
import sys
from pathlib import Path

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_validation():
    # missing required fields should fail
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_invalid_condition():
    response = client.post("/predict", json={
        "location": "CityA",
        "size": 2000,
        "bedrooms": 3,
        "bathrooms": 2,
        "year_built": 2010,
        "condition": "Invalid",
        "property_type": "Single Family"
    })
    assert response.status_code == 422
