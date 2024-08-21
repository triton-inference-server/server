import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.api_server import init_app


# Override conftest.py default model
@pytest.fixture
def model():
    return "mock_llm"


class TestObservability:
    @pytest.fixture(scope="class")
    def client(self):
        model_repository = Path(__file__).parent / "test_models"
        os.environ["TRITON_MODEL_REPOSITORY"] = str(model_repository)
        app = init_app()
        with TestClient(app) as test_client:
            yield test_client

    ### General Error Handling ###
    def test_not_found(self, client):
        response = client.get("/does-not-exist")
        assert response.status_code == 404

    ### Startup / Health ###
    def test_startup_success(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_startup_fail(self):
        os.environ["TRITON_MODEL_REPOSITORY"] = "/does/not/exist"
        with pytest.raises(Exception):
            # Test that FastAPI lifespan startup fails when initializing Triton
            # with unknown model repository.
            app = init_app()
            with TestClient(app):
                pass

    ### Metrics ###
    def test_startup_metrics(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        # FIXME: Flesh out more
        # NOTE: response.json() works even on non-json prometheus data
        assert "nv_cpu_utilization" in response.json()

    ### Models ###
    def test_models_list(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 200
        models = response.json()["data"]
        assert len(models) == 2
        for model in models:
            assert model["id"]
            assert model["object"] == "model"
            assert model["created"] > 0
            assert model["owned_by"] == "Triton Inference Server"

    def test_models_get(self, client, model):
        response = client.get(f"/v1/models/{model}")
        assert response.status_code == 200
        model_resp = response.json()
        assert model_resp["id"] == model
        assert model_resp["object"] == "model"
        assert model_resp["created"] > 0
        assert model_resp["owned_by"] == "Triton Inference Server"
