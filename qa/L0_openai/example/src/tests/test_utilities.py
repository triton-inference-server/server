import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from src.api_server import app

# TODO: Use fixture for less verbose model repo prep
# @pytest.fixture(scope="session")
# def setup_model_repository():
#    pass


def test_not_found():
    with TestClient(app) as client:
        response = client.get("/does-not-exist")
        assert response.status_code == 404


def test_startup_metrics():
    with tempfile.TemporaryDirectory() as model_repository:
        os.environ["TRITON_MODEL_REPOSITORY"] = model_repository
        with TestClient(app) as client:
            response = client.get("/metrics")
            assert response.status_code == 200
            # FIXME: Flesh out more
            # NOTE: response.json() works even on non-json prometheus data?
            assert "nv_cpu_utilization" in response.json()
            # No models loaded, no per-model metrics
            assert "nv_inference_count" not in response.json()


def test_startup_success():
    with tempfile.TemporaryDirectory() as model_repository:
        os.environ["TRITON_MODEL_REPOSITORY"] = model_repository
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200


def test_startup_fail():
    os.environ["TRITON_MODEL_REPOSITORY"] = "/does/not/exist"
    with pytest.raises(Exception):
        # Test that FastAPI lifespan startup fails when initializing Triton
        # with unknown model repository.
        with TestClient(app) as client:
            pass
