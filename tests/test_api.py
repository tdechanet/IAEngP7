from fastapi.testclient import TestClient
from api import app
from pytest import fixture


@fixture
def client():
    with TestClient(app) as c:
        yield c

def test_redirect(client):

    response = client.get("")

    assert response.url == "http://testserver/docs"

def test_predict_positive(client):

    response = client.post("/predict", json={"text": "I loved this movie, the actors were great in it"})

    assert response.status_code == 200
    assert response.json()["sentiment"] == "POSITIF"
    assert response.json()["confiance"] > 0.5

def test_predict_negative(client):

    response = client.post("/predict", json={"text": "Honestly, this movie was kind of mid"})

    assert response.status_code == 200
    assert response.json()["sentiment"] == "NEGATIF"
    assert response.json()["confiance"] > 0.5

def test_feedback_wrong(client):

    response = client.post("/feedback", json={"prediction": "PÖSITIF", "text": "Honestly, this movie was kind of mid"})

    assert response.status_code == 200
    assert response.json()["output"] == "La valeur prédiction doit être égale à 'NEGATIF' ou 'POSITIF'"

def test_feedback_good(client):

    response = client.post("/feedback", json={"prediction": "POSITIF", "text": "Honestly, this movie was kind of mid"})

    assert response.status_code == 200
    assert response.json()["output"] == "Merci pour votre retour"