from fastapi.testclient import TestClient

from service_foundations.app import create_app
from service_foundations.config import Settings


def test_application_metadata() -> None:
    settings = Settings(
        service_name="test-service",
        service_version="9.9.9",
    )

    app = create_app(settings)

    assert app.title == "test-service"
    assert app.version == "9.9.9"


def test_health_endpoints() -> None:
    app = create_app(Settings())

    with TestClient(app) as client:
        live_response = client.get("/health/live")
        ready_response = client.get("/health/ready")

    assert live_response.status_code == 200
    assert live_response.json() == {"status": "ok"}
    assert ready_response.status_code == 200
    assert ready_response.json() == {"status": "ready"}


def test_readiness_fails_before_startup() -> None:
    app = create_app(Settings())
    client = TestClient(app)

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Service is not ready"}
