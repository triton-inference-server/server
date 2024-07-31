import os
import tempfile

from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)


def test_health_success():
    # Context Manager to trigger app lifespan:
    # https://fastapi.tiangolo.com/advanced/testing-events/
    with tempfile.TemporaryDirectory() as model_repository:
        os.environ["TRITON_MODEL_REPOSITORY"] = model_repository
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200


def test_health_fail():
    # Context Manager to trigger app lifespan:
    # https://fastapi.tiangolo.com/advanced/testing-events/
    os.environ["TRITON_MODEL_REPOSITORY"] = "/does/not/exist"
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 400
