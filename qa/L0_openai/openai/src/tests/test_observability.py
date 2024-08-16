import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.api_server import init_app

TEST_MODEL = "mock_llm"


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
        # NOTE: response.json() works even on non-json prometheus data?
        assert "nv_cpu_utilization" in response.json()

    ### Models ###
    def test_models_list(self, client):
        # TODO: Load multiple models and make sure exactly ALL are returned
        response = client.get("/v1/models")
        assert response.status_code == 200
        # TODO: Flesh out
        models = response.json()["data"]
        assert len(models) == 1
        assert models[0]["id"] == TEST_MODEL
        assert models[0]["object"] == "model"
        assert models[0]["created"] > 0

    def test_models_get(self, client):
        # TODO: Load multiple models and make sure exactly 1 is returned
        response = client.get(f"/v1/models/{TEST_MODEL}")
        assert response.status_code == 200
        # TODO: Flesh out
        model = response.json()
        assert model["id"] == TEST_MODEL
        assert model["object"] == "model"
        assert model["created"] > 0
