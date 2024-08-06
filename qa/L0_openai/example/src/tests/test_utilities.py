import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.api_server import app

TEST_MODEL = "mock_llm"


# TODO: May need to modify fixture scope
@pytest.fixture(scope="function", autouse=True)
def setup_model_repository():
    model_repository = Path(__file__).parent / "test_models"
    os.environ["TRITON_MODEL_REPOSITORY"] = str(model_repository)


def test_not_found():
    with TestClient(app) as client:
        response = client.get("/does-not-exist")
        assert response.status_code == 404


### Startup / Health ###


def test_startup_success():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200


def test_startup_fail():
    os.environ["TRITON_MODEL_REPOSITORY"] = "/does/not/exist"
    with pytest.raises(Exception):
        # Test that FastAPI lifespan startup fails when initializing Triton
        # with unknown model repository.
        with TestClient(app):
            pass


### Metrics ###


def test_startup_metrics():
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        # FIXME: Flesh out more
        # NOTE: response.json() works even on non-json prometheus data?
        assert "nv_cpu_utilization" in response.json()


### Models ###


def test_models_list():
    # TODO: Load multiple models and make sure exactly ALL are returned
    with TestClient(app) as client:
        response = client.get("/v1/models")
        assert response.status_code == 200
        # TODO: Flesh out
        models = response.json()["data"]
        assert len(models) == 1
        assert models[0]["id"] == TEST_MODEL
        assert models[0]["object"] == "model"
        assert models[0]["created"] > 0


def test_models_get():
    # TODO: Load multiple models and make sure exactly 1 is returned
    with TestClient(app) as client:
        response = client.get(f"/v1/models/{TEST_MODEL}")
        assert response.status_code == 200
        # TODO: Flesh out
        model = response.json()
        assert model["id"] == TEST_MODEL
        assert model["object"] == "model"
        assert model["created"] > 0
