from fastapi.testclient import TestClient
from api import app
import pytest


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_predict_positive(client):

    response = client.post("/predict", json={"text": "I loved this movie, the actors were great in it"})

    assert response.status_code == 200
    assert response.json()["sentiment"] == "POSITIF"
    assert response.json()["confidence"] > 0.5

def test_predict_negative(client):

    response = client.post("/predict", json={"text": "Honestly, this movie was kind of mid"})

    assert response.status_code == 200
    assert response.json()["sentiment"] == "NEGATIF"
    assert response.json()["confidence"] > 0.5