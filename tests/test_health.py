from fastapi.testclient import TestClient

from app.main import app


def test_home_page_is_served() -> None:
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "NeuroVision AI" in response.text


def test_health_has_model_state() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert "model_ready" in body
        assert body["classes"] == [
            "Glioma",
            "Meningioma",
            "No Tumour",
            "Pituitary Tumour",
        ]
