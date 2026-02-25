# Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import concurrent.futures
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient
from tests.utils import (
    OpenAIServer,
    setup_fastapi_app,
    setup_server,
    setup_server_explicit,
)

TEST_MODEL_REPOSITORY = str(Path(__file__).parent / "test_models")
TEST_MODEL = "mock_llm"
TEST_MODEL_2 = "identity_py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_model_names(client: TestClient) -> list[str]:
    response = client.get("/v1/models")
    assert response.status_code == 200
    return [m["id"] for m in response.json()["data"]]


def _assert_model_fields(model_data: dict):
    assert model_data["id"]
    assert model_data["object"] == "model"
    assert model_data["created"] > 0
    assert model_data["owned_by"] == "Triton Inference Server"


# ---------------------------------------------------------------------------
# Group 1: Mode enforcement – NONE mode rejects load/unload
# ---------------------------------------------------------------------------
class TestModelManagementNoneMode:
    @pytest.fixture(scope="class")
    def client(self):
        server = setup_server(TEST_MODEL_REPOSITORY)
        app = setup_fastapi_app(tokenizer="", server=server, backend=None)
        with TestClient(app) as test_client:
            yield test_client
        server.stop()

    def test_load_rejected_in_none_mode(self, client):
        response = client.post(f"/v1/models/{TEST_MODEL}/load")
        assert response.status_code == 400
        assert "explicit" in response.json()["detail"].lower()

    def test_unload_rejected_in_none_mode(self, client):
        response = client.post(f"/v1/models/{TEST_MODEL}/unload")
        assert response.status_code == 400
        assert "explicit" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Group 2: Core load / unload operations – EXPLICIT mode
# ---------------------------------------------------------------------------
class TestModelManagement:
    @pytest.fixture(scope="class")
    def client(self):
        server = setup_server_explicit(TEST_MODEL_REPOSITORY)
        app = setup_fastapi_app(tokenizer="", server=server, backend=None)
        with TestClient(app) as test_client:
            yield test_client
        server.stop()

    def test_load_model(self, client):
        # 1. Initially no models loaded
        assert _get_model_names(client) == []

        # 2. Load a model – 200, response has correct OpenAI Model fields
        response = client.post(f"/v1/models/{TEST_MODEL}/load")
        assert response.status_code == 200
        _assert_model_fields(response.json())
        assert response.json()["id"] == TEST_MODEL

        # 3. Model now appears in GET /v1/models
        assert TEST_MODEL in _get_model_names(client)

        # 4. GET /v1/models/{model_name} returns correct info
        response = client.get(f"/v1/models/{TEST_MODEL}")
        assert response.status_code == 200
        _assert_model_fields(response.json())

    def test_unload_model(self, client):
        # Ensure model is loaded first
        if TEST_MODEL not in _get_model_names(client):
            response = client.post(f"/v1/models/{TEST_MODEL}/load")
            assert response.status_code == 200

        # 1. Unload – 200 with success status
        response = client.post(f"/v1/models/{TEST_MODEL}/unload")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["model"] == TEST_MODEL

        # 2. Model no longer in model list
        assert TEST_MODEL not in _get_model_names(client)

        # 3. GET /v1/models/{model_name} now 404
        response = client.get(f"/v1/models/{TEST_MODEL}")
        assert response.status_code == 404

    def test_load_already_loaded_model(self, client):
        # Load the model
        client.post(f"/v1/models/{TEST_MODEL}/load")

        # Second load of the same model should fail
        response = client.post(f"/v1/models/{TEST_MODEL}/load")
        assert response.status_code == 400
        assert "already loaded" in response.json()["detail"].lower()

        # Cleanup
        client.post(f"/v1/models/{TEST_MODEL}/unload")

    def test_unload_unknown_model(self, client):
        response = client.post("/v1/models/nonexistent_model/unload")
        assert response.status_code == 400
        assert "unknown model" in response.json()["detail"].lower()

    def test_load_nonexistent_model(self, client):
        response = client.post("/v1/models/model_not_in_repo/load")
        assert response.status_code == 500

    def test_load_unload_reload(self, client):
        # Load
        response = client.post(f"/v1/models/{TEST_MODEL}/load")
        assert response.status_code == 200

        # Unload
        response = client.post(f"/v1/models/{TEST_MODEL}/unload")
        assert response.status_code == 200
        assert TEST_MODEL not in _get_model_names(client)

        # Reload – should succeed
        response = client.post(f"/v1/models/{TEST_MODEL}/load")
        assert response.status_code == 200
        assert TEST_MODEL in _get_model_names(client)

        # Cleanup
        client.post(f"/v1/models/{TEST_MODEL}/unload")

    def test_load_multiple_models(self, client):
        # Load both models
        r1 = client.post(f"/v1/models/{TEST_MODEL}/load")
        r2 = client.post(f"/v1/models/{TEST_MODEL_2}/load")
        assert r1.status_code == 200
        assert r2.status_code == 200

        # Both appear in list
        names = _get_model_names(client)
        assert TEST_MODEL in names
        assert TEST_MODEL_2 in names

        # Unload one – the other remains
        client.post(f"/v1/models/{TEST_MODEL}/unload")
        names = _get_model_names(client)
        assert TEST_MODEL not in names
        assert TEST_MODEL_2 in names

        # Cleanup
        client.post(f"/v1/models/{TEST_MODEL_2}/unload")


# ---------------------------------------------------------------------------
# Group 3: Sequential and concurrent load/unload
# ---------------------------------------------------------------------------
class TestModelManagementConcurrency:
    @pytest.fixture(scope="class")
    def client(self):
        server = setup_server_explicit(TEST_MODEL_REPOSITORY)
        app = setup_fastapi_app(tokenizer="", server=server, backend=None)
        with TestClient(app) as test_client:
            yield test_client
        server.stop()

    def test_rapid_sequential_load_unload(self, client):
        for _ in range(3):
            r = client.post(f"/v1/models/{TEST_MODEL}/load")
            assert r.status_code == 200
            assert TEST_MODEL in _get_model_names(client)

            r = client.post(f"/v1/models/{TEST_MODEL}/unload")
            assert r.status_code == 200
            assert TEST_MODEL not in _get_model_names(client)

    def test_concurrent_load_different_models(self, client):
        def load_model(model_name):
            return client.post(f"/v1/models/{model_name}/load")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(load_model, TEST_MODEL)
            f2 = executor.submit(load_model, TEST_MODEL_2)
            r1 = f1.result()
            r2 = f2.result()

        assert r1.status_code == 200
        assert r2.status_code == 200
        names = _get_model_names(client)
        assert TEST_MODEL in names
        assert TEST_MODEL_2 in names

        # Cleanup
        client.post(f"/v1/models/{TEST_MODEL}/unload")
        client.post(f"/v1/models/{TEST_MODEL_2}/unload")

    def test_concurrent_load_same_model(self, client):
        def load_model():
            return client.post(f"/v1/models/{TEST_MODEL}/load")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(load_model)
            f2 = executor.submit(load_model)
            r1 = f1.result()
            r2 = f2.result()

        statuses = sorted([r1.status_code, r2.status_code])
        # One should succeed (200), the other should fail (400 "already loaded")
        assert statuses == [200, 400]

        # Model is loaded exactly once
        assert TEST_MODEL in _get_model_names(client)

        # Cleanup
        client.post(f"/v1/models/{TEST_MODEL}/unload")

    def test_concurrent_unload_same_model(self, client):
        client.post(f"/v1/models/{TEST_MODEL}/load")

        def unload_model():
            return client.post(f"/v1/models/{TEST_MODEL}/unload")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(unload_model)
            f2 = executor.submit(unload_model)
            r1 = f1.result()
            r2 = f2.result()

        statuses = sorted([r1.status_code, r2.status_code])
        # One succeeds, the other gets "Unknown model"
        assert statuses == [200, 400]

        assert TEST_MODEL not in _get_model_names(client)


# ---------------------------------------------------------------------------
# Group 4: Startup models (subprocess to test CLI args)
# ---------------------------------------------------------------------------
@pytest.mark.openai
class TestModelManagementStartupModels:
    def test_explicit_no_startup_models(self):
        args = [
            "--model-repository",
            TEST_MODEL_REPOSITORY,
            "--model-control-mode",
            "explicit",
        ]
        with OpenAIServer(args) as openai_server:
            response = requests.get(
                openai_server.url_for("v1", "models"), timeout=10
            )
            assert response.status_code == 200
            assert len(response.json()["data"]) == 0

    def test_explicit_with_startup_models(self):
        args = [
            "--model-repository",
            TEST_MODEL_REPOSITORY,
            "--model-control-mode",
            "explicit",
            "--startup-models",
            TEST_MODEL,
        ]
        with OpenAIServer(args) as openai_server:
            response = requests.get(
                openai_server.url_for("v1", "models"), timeout=10
            )
            assert response.status_code == 200
            names = [m["id"] for m in response.json()["data"]]
            assert TEST_MODEL in names
            assert TEST_MODEL_2 not in names

    def test_explicit_load_unload_via_api(self):
        args = [
            "--model-repository",
            TEST_MODEL_REPOSITORY,
            "--model-control-mode",
            "explicit",
        ]
        with OpenAIServer(args) as server:
            base = server.url_root

            assert requests.post(f"{base}/v1/models/{TEST_MODEL}/load", timeout=30).status_code == 200
            assert requests.post(f"{base}/v1/models/{TEST_MODEL_2}/load", timeout=30).status_code == 200

            names = [m["id"] for m in requests.get(f"{base}/v1/models", timeout=10).json()["data"]]
            assert TEST_MODEL in names
            assert TEST_MODEL_2 in names

            assert requests.post(f"{base}/v1/models/{TEST_MODEL}/unload", timeout=30).status_code == 200

            names = [m["id"] for m in requests.get(f"{base}/v1/models", timeout=10).json()["data"]]
            assert TEST_MODEL not in names
            assert TEST_MODEL_2 in names


# ---------------------------------------------------------------------------
# Group 5: Inference with real LLM backend after load/unload
# ---------------------------------------------------------------------------
@pytest.mark.openai
class TestModelManagementInference:
    @pytest.fixture(scope="class")
    def managed_server(
        self,
        model_repository: str,
        tokenizer_model: str,
        backend: str,
    ):
        args = [
            "--model-repository",
            model_repository,
            "--tokenizer",
            tokenizer_model,
            "--backend",
            backend,
            "--model-control-mode",
            "explicit",
        ]
        with OpenAIServer(args) as openai_server:
            yield openai_server

    @staticmethod
    def _completions(base_url, model_name):
        return requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model_name,
                "prompt": "What is machine learning?",
                "max_tokens": 10,
            },
            timeout=30,
        )

    @staticmethod
    def _chat_completions(base_url, model_name):
        return requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "What is machine learning?"}],
                "max_tokens": 10,
            },
            timeout=30,
        )

    def test_reload_inference(self, managed_server, model: str):
        base = managed_server.url_root

        # Load → both endpoints succeed with 200
        assert requests.post(f"{base}/v1/models/{model}/load", timeout=120).status_code == 200
        r = self._completions(base, model)
        assert r.status_code == 200
        assert r.json()["choices"][0]["text"].strip()

        r = self._chat_completions(base, model)
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"].strip()

        # Unload → both fail with "Unknown model"
        assert requests.post(f"{base}/v1/models/{model}/unload", timeout=60).status_code == 200
        r = self._completions(base, model)
        assert r.status_code == 400
        assert "unknown model" in r.json()["detail"].lower()

        r = self._chat_completions(base, model)
        assert r.status_code == 400
        assert "unknown model" in r.json()["detail"].lower()

        # Reload → both succeed again with 200
        assert requests.post(f"{base}/v1/models/{model}/load", timeout=120).status_code == 200
        r = self._completions(base, model)
        assert r.status_code == 200
        assert r.json()["choices"][0]["text"].strip()

        r = self._chat_completions(base, model)
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"].strip()

        # Cleanup
        requests.post(f"{base}/v1/models/{model}/unload", timeout=60)


# ---------------------------------------------------------------------------
# Group 6: API restriction for model-management endpoints
# ---------------------------------------------------------------------------
@pytest.mark.openai
class TestModelManagementRestriction:
    @pytest.fixture(scope="class")
    def restricted_server(self):
        args = [
            "--model-repository",
            TEST_MODEL_REPOSITORY,
            "--model-control-mode",
            "explicit",
            "--startup-models",
            TEST_MODEL,
            "--openai-restricted-api",
            "model-management",
            "mgmt-key",
            "mgmt-secret",
        ]
        with OpenAIServer(args) as openai_server:
            yield openai_server

    def test_load_without_auth_rejected(self, restricted_server):
        r = requests.post(
            f"{restricted_server.url_root}/v1/models/{TEST_MODEL_2}/load",
            timeout=10,
        )
        assert r.status_code == 401

    def test_unload_without_auth_rejected(self, restricted_server):
        r = requests.post(
            f"{restricted_server.url_root}/v1/models/{TEST_MODEL}/unload",
            timeout=10,
        )
        assert r.status_code == 401

    def test_load_with_valid_auth(self, restricted_server):
        headers = {"mgmt-key": "mgmt-secret"}
        r = requests.post(
            f"{restricted_server.url_root}/v1/models/{TEST_MODEL_2}/load",
            headers=headers,
            timeout=30,
        )
        assert r.status_code == 200

        # Cleanup
        requests.post(
            f"{restricted_server.url_root}/v1/models/{TEST_MODEL_2}/unload",
            headers=headers,
            timeout=30,
        )

    def test_model_list_unrestricted(self, restricted_server):
        r = requests.get(
            f"{restricted_server.url_root}/v1/models", timeout=10
        )
        assert r.status_code == 200

    def test_health_unrestricted(self, restricted_server):
        r = requests.get(
            f"{restricted_server.url_root}/health/ready", timeout=10
        )
        assert r.status_code == 200
