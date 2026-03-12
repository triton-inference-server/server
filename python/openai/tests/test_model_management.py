# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
from pathlib import Path

import pytest
import requests
import tritonserver
from fastapi.testclient import TestClient
from tests.utils import OpenAIServer, setup_fastapi_app, setup_server

TEST_MODEL_REPOSITORY = str(Path(__file__).parent / "test_models")
TEST_MODEL = "mock_llm"
TEST_MODEL_2 = "identity_py"
MODEL_MGMT_LOAD_TIMEOUT_S = int(os.getenv("TEST_MODEL_MANAGEMENT_LOAD_TIMEOUT", "300"))
MODEL_MGMT_UNLOAD_TIMEOUT_S = int(
    os.getenv("TEST_MODEL_MANAGEMENT_UNLOAD_TIMEOUT", "120")
)


def _get_model_names(client: TestClient) -> list[str]:
    response = client.get("/v1/models")
    assert response.status_code == 200
    return [m["id"] for m in response.json()["data"]]


def _assert_model_fields(model_data: dict):
    assert model_data["id"]
    assert model_data["object"] == "model"
    assert model_data["created"] > 0
    assert model_data["owned_by"] == "Triton Inference Server"


# Test Mode enforcement – NONE mode rejects load/unload API calls
class TestModelManagementNoneMode:
    @pytest.fixture(scope="class")
    def client(self):
        server = setup_server(TEST_MODEL_REPOSITORY)
        app = setup_fastapi_app(tokenizer="", server=server, backend=None)
        with TestClient(app) as test_client:
            yield test_client
        server.stop()

    def test_load_and_unload_rejected_in_none_mode(self, client):
        for api in ["load", "unload"]:
            response = client.post(f"/v1/models/{TEST_MODEL}/{api}")
            assert response.status_code == 400
            assert (
                "model load/unload requires --model-control-mode=explicit"
                in response.json()["detail"].lower()
            )


# Test load / unload operations – EXPLICIT mode
class TestModelManagement:
    @pytest.fixture(scope="class")
    def client(self):
        server = setup_server(
            TEST_MODEL_REPOSITORY,
            model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
        )
        app = setup_fastapi_app(tokenizer="", server=server, backend=None)
        with TestClient(app) as test_client:
            yield test_client
        server.stop()

    def test_load_model(self, client):
        # Initially no models loaded
        assert _get_model_names(client) == []

        # Load a model – 200, response has correct OpenAI Model fields
        response = client.post(f"/v1/models/{TEST_MODEL}/load")
        assert response.status_code == 200
        _assert_model_fields(response.json())
        assert response.json()["id"] == TEST_MODEL

        assert TEST_MODEL in _get_model_names(client)

        # GET /v1/models/{model_name} returns correct info
        response = client.get(f"/v1/models/{TEST_MODEL}")
        assert response.status_code == 200
        _assert_model_fields(response.json())

    def test_unload_model(self, client):
        if TEST_MODEL not in _get_model_names(client):
            response = client.post(f"/v1/models/{TEST_MODEL}/load")
            assert response.status_code == 200

        response = client.post(f"/v1/models/{TEST_MODEL}/unload")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["model"] == TEST_MODEL

        assert TEST_MODEL not in _get_model_names(client)

        assert client.get(f"/v1/models/{TEST_MODEL}").status_code == 404

    def test_load_already_loaded_model(self, client):
        client.post(f"/v1/models/{TEST_MODEL}/unload")
        assert client.post(f"/v1/models/{TEST_MODEL}/load").status_code == 200

        # Second load of the same model should fail
        response = client.post(f"/v1/models/{TEST_MODEL}/load")
        assert response.status_code == 400
        assert "already loaded" in response.json()["detail"].lower()

        client.post(f"/v1/models/{TEST_MODEL}/unload")

    def test_unload_unknown_model(self, client):
        response = client.post("/v1/models/nonexistent_model/unload")
        assert response.status_code == 400
        assert "unknown model" in response.json()["detail"].lower()

    def test_load_nonexistent_model(self, client):
        response = client.post("/v1/models/model_not_in_repo/load")
        assert response.status_code == 500

    def test_load_unload_reload(self, client):
        assert client.post(f"/v1/models/{TEST_MODEL}/load").status_code == 200
        assert TEST_MODEL in _get_model_names(client)

        assert client.post(f"/v1/models/{TEST_MODEL}/unload").status_code == 200
        assert TEST_MODEL not in _get_model_names(client)

        assert client.post(f"/v1/models/{TEST_MODEL}/load").status_code == 200
        assert TEST_MODEL in _get_model_names(client)

        client.post(f"/v1/models/{TEST_MODEL}/unload")

    def test_load_multiple_models(self, client):
        assert client.post(f"/v1/models/{TEST_MODEL}/load").status_code == 200
        assert client.post(f"/v1/models/{TEST_MODEL_2}/load").status_code == 200

        names = _get_model_names(client)
        assert TEST_MODEL in names
        assert TEST_MODEL_2 in names

        # Unload one; other remains
        client.post(f"/v1/models/{TEST_MODEL}/unload")
        names = _get_model_names(client)
        assert TEST_MODEL not in names
        assert TEST_MODEL_2 in names

        client.post(f"/v1/models/{TEST_MODEL_2}/unload")


# Test Sequential and concurrent load/unload
class TestModelManagementConcurrency:
    @pytest.fixture(scope="class")
    def client(self):
        server = setup_server(
            TEST_MODEL_REPOSITORY,
            model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
        )
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(client.post, f"/v1/models/{TEST_MODEL}/load")
            f2 = pool.submit(client.post, f"/v1/models/{TEST_MODEL_2}/load")

        assert f1.result().status_code == 200
        assert f2.result().status_code == 200
        names = _get_model_names(client)
        assert TEST_MODEL in names
        assert TEST_MODEL_2 in names

        client.post(f"/v1/models/{TEST_MODEL}/unload")
        client.post(f"/v1/models/{TEST_MODEL_2}/unload")

    def test_concurrent_load_same_model(self, client):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(client.post, f"/v1/models/{TEST_MODEL}/load")
            f2 = pool.submit(client.post, f"/v1/models/{TEST_MODEL}/load")

        codes = sorted([f1.result().status_code, f2.result().status_code])
        assert codes == [
            200,
            400,
        ], f"Expected one 200 and one 400 for concurrent loads of the same model, got {codes}"
        assert TEST_MODEL in _get_model_names(client)

        client.post(f"/v1/models/{TEST_MODEL}/unload")

    def test_concurrent_unload_same_model(self, client):
        client.post(f"/v1/models/{TEST_MODEL}/load")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(client.post, f"/v1/models/{TEST_MODEL}/unload")
            f2 = pool.submit(client.post, f"/v1/models/{TEST_MODEL}/unload")

        # One succeeds (200), the other fails (400 "unknown model") — order is non-deterministic
        codes = sorted([f1.result().status_code, f2.result().status_code])
        assert codes == [
            200,
            400,
        ], f"Expected one 200 and one 400 for concurrent unloads of the same model, got {codes}"
        assert TEST_MODEL not in _get_model_names(client)


# Test "--load-model" and "--model-control-mode" CLI options
@pytest.mark.openai
class TestModelManagementCLIOptions:
    def _assert_server_startup_fails(self, args, expected_error: str):
        """Helper: verify server fails to start and stderr contains expected_error."""
        with pytest.raises(Exception) as exc_info:
            with OpenAIServer(args):
                pass
        assert expected_error in str(exc_info.value)

    def test_load_model_without_explicit_mode_is_error(self):
        """--load-model without --model-control-mode=explicit must exit with error.
        Error message matches native tritonserver exactly."""
        self._assert_server_startup_fails(
            args=[
                "--model-repository",
                TEST_MODEL_REPOSITORY,
                "--load-model",
                TEST_MODEL,
            ],
            expected_error="Error: Use of '--load-model' requires setting '--model-control-mode=explicit' as well.",
        )

    def test_explicit_no_load_model_starts_with_no_models(self):
        args = [
            "--model-repository",
            TEST_MODEL_REPOSITORY,
            "--model-control-mode",
            "explicit",
        ]
        with OpenAIServer(args) as openai_server:
            r = requests.get(openai_server.url_for("v1", "models"), timeout=10)
            assert r.status_code == 200
            assert len(r.json()["data"]) == 0

    def test_explicit_single_load_model_at_startup(self):
        args = [
            "--model-repository",
            TEST_MODEL_REPOSITORY,
            "--model-control-mode",
            "explicit",
            "--load-model",
            TEST_MODEL,
        ]
        with OpenAIServer(args) as openai_server:
            r = requests.get(openai_server.url_for("v1", "models"), timeout=10)
            names = [m["id"] for m in r.json()["data"]]
            assert TEST_MODEL in names
            assert TEST_MODEL_2 not in names

    def test_explicit_multiple_load_model_at_startup(self):
        args = [
            "--model-repository",
            TEST_MODEL_REPOSITORY,
            "--model-control-mode",
            "explicit",
            "--load-model",
            TEST_MODEL,
            "--load-model",
            TEST_MODEL_2,
        ]
        with OpenAIServer(args) as openai_server:
            r = requests.get(openai_server.url_for("v1", "models"), timeout=10)
            names = [m["id"] for m in r.json()["data"]]
            assert TEST_MODEL in names
            assert TEST_MODEL_2 in names

    def test_explicit_load_model_wildcard_loads_all(self):
        args = [
            "--model-repository",
            TEST_MODEL_REPOSITORY,
            "--model-control-mode",
            "explicit",
            "--load-model",
            "*",
        ]
        with OpenAIServer(args) as openai_server:
            r = requests.get(openai_server.url_for("v1", "models"), timeout=10)
            names = [m["id"] for m in r.json()["data"]]
            assert TEST_MODEL in names
            assert TEST_MODEL_2 in names

    def test_explicit_load_model_wildcard_with_other_is_error(self):
        self._assert_server_startup_fails(
            args=[
                "--model-repository",
                TEST_MODEL_REPOSITORY,
                "--model-control-mode",
                "explicit",
                "--load-model",
                "*",
                "--load-model",
                TEST_MODEL,
            ],
            expected_error="Wildcard model name '*' must be the ONLY startup model if specified at all.",
        )

    def test_explicit_dynamic_load_unload_via_api(self):
        args = [
            "--model-repository",
            TEST_MODEL_REPOSITORY,
            "--model-control-mode",
            "explicit",
        ]
        with OpenAIServer(args) as server:
            base = server.url_root

            assert (
                requests.post(
                    f"{base}/v1/models/{TEST_MODEL}/load", timeout=30
                ).status_code
                == 200
            )
            assert (
                requests.post(
                    f"{base}/v1/models/{TEST_MODEL_2}/load", timeout=30
                ).status_code
                == 200
            )

            names = [
                m["id"]
                for m in requests.get(f"{base}/v1/models", timeout=10).json()["data"]
            ]
            assert TEST_MODEL in names
            assert TEST_MODEL_2 in names

            assert (
                requests.post(
                    f"{base}/v1/models/{TEST_MODEL}/unload", timeout=30
                ).status_code
                == 200
            )

            names = [
                m["id"]
                for m in requests.get(f"{base}/v1/models", timeout=10).json()["data"]
            ]
            assert TEST_MODEL not in names
            assert TEST_MODEL_2 in names


# Test Inference with real LLM backend (vLLM / TRT-LLM) after load/unload
@pytest.mark.openai
class TestModelManagementInference:
    @pytest.fixture(scope="class")
    def server_with_explicit_mode(
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

    @pytest.fixture(autouse=True)
    def ensure_model_unloaded(self, server_with_explicit_mode, model: str):
        """Guarantee clean state before and after every test: unload silently
        (model may already be unloaded -- that is fine)."""
        requests.post(
            f"{server_with_explicit_mode.url_root}/v1/models/{model}/unload",
            timeout=MODEL_MGMT_UNLOAD_TIMEOUT_S,
        )
        yield
        requests.post(
            f"{server_with_explicit_mode.url_root}/v1/models/{model}/unload",
            timeout=MODEL_MGMT_UNLOAD_TIMEOUT_S,
        )

    @staticmethod
    def _load(base_url, model_name):
        r = requests.post(
            f"{base_url}/v1/models/{model_name}/load",
            timeout=MODEL_MGMT_LOAD_TIMEOUT_S,
        )
        assert r.status_code == 200, f"Load failed: {r.text}"

    @staticmethod
    def _unload(base_url, model_name):
        r = requests.post(
            f"{base_url}/v1/models/{model_name}/unload",
            timeout=MODEL_MGMT_UNLOAD_TIMEOUT_S,
        )
        assert r.status_code == 200, f"Unload failed: {r.text}"

    @staticmethod
    def _assert_unknown_model(response):
        assert response.status_code == 400
        assert "unknown model" in response.json()["detail"].lower()

    @staticmethod
    def _assert_usage(usage):
        assert usage is not None
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )

    @staticmethod
    def _completions(base_url, model_name, **kwargs):
        return requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model_name,
                "prompt": "What is machine learning?",
                "max_tokens": 10,
                **kwargs,
            },
            timeout=30,
        )

    @staticmethod
    def _chat_completions(base_url, model_name, **kwargs):
        return requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "What is machine learning?"}],
                "max_tokens": 10,
                **kwargs,
            },
            timeout=30,
        )

    def test_load_completions(self, server_with_explicit_mode, model: str):
        base = server_with_explicit_mode.url_root
        self._assert_unknown_model(self._completions(base, model))

        self._load(base, model)
        r = self._completions(base, model)
        assert r.status_code == 200
        data = r.json()
        assert data["choices"][0]["text"].strip()
        assert data["choices"][0]["finish_reason"] == "stop"
        self._assert_usage(data["usage"])

        self._unload(base, model)

    def test_load_chat_completions(self, server_with_explicit_mode, model: str):
        base = server_with_explicit_mode.url_root
        self._assert_unknown_model(self._chat_completions(base, model))

        self._load(base, model)
        r = self._chat_completions(base, model)
        assert r.status_code == 200
        data = r.json()
        msg = data["choices"][0]["message"]
        assert msg["content"].strip()
        assert msg["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "stop"
        self._assert_usage(data["usage"])

        self._unload(base, model)

    def test_load_streaming_completions(self, server_with_explicit_mode, model: str):
        base = server_with_explicit_mode.url_root
        self._load(base, model)

        r = requests.post(
            f"{base}/v1/completions",
            json={
                "model": model,
                "prompt": "What is machine learning?",
                "max_tokens": 10,
                "stream": True,
            },
            stream=True,
            timeout=30,
        )
        assert r.status_code == 200
        chunks = [
            line.removeprefix("data: ")
            for line in r.iter_lines(decode_unicode=True)
            if line.startswith("data: ") and line.strip() != "data: [DONE]"
        ]
        assert len(chunks) > 0

        self._unload(base, model)

    def test_load_streaming_chat_completions(
        self, server_with_explicit_mode, model: str
    ):
        base = server_with_explicit_mode.url_root
        self._load(base, model)

        r = requests.post(
            f"{base}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "What is machine learning?"}],
                "max_tokens": 10,
                "stream": True,
            },
            stream=True,
            timeout=30,
        )
        assert r.status_code == 200
        chunks = [
            line.removeprefix("data: ")
            for line in r.iter_lines(decode_unicode=True)
            if line.startswith("data: ") and line.strip() != "data: [DONE]"
        ]
        assert len(chunks) > 0

        self._unload(base, model)

    def test_unload_rejects_inference(self, server_with_explicit_mode, model: str):
        base = server_with_explicit_mode.url_root
        self._load(base, model)

        assert self._completions(base, model).status_code == 200
        assert self._chat_completions(base, model).status_code == 200

        self._unload(base, model)

        self._assert_unknown_model(self._completions(base, model))
        self._assert_unknown_model(self._chat_completions(base, model))

    def test_reload_inference(self, server_with_explicit_mode, model: str):
        base = server_with_explicit_mode.url_root

        self._load(base, model)
        assert self._completions(base, model).status_code == 200
        assert self._chat_completions(base, model).status_code == 200

        self._unload(base, model)
        self._assert_unknown_model(self._completions(base, model))

        self._load(base, model)
        r = self._completions(base, model)
        assert r.status_code == 200
        assert r.json()["choices"][0]["text"].strip()

        r = self._chat_completions(base, model)
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"].strip()

        self._unload(base, model)

    def test_model_list_after_load_unload(self, server_with_explicit_mode, model: str):
        base = server_with_explicit_mode.url_root

        # No models initially
        r = requests.get(f"{base}/v1/models", timeout=10)
        assert r.status_code == 200
        assert len(r.json()["data"]) == 0

        self._load(base, model)
        names = [
            m["id"]
            for m in requests.get(f"{base}/v1/models", timeout=10).json()["data"]
        ]
        assert model in names

        self._unload(base, model)
        names = [
            m["id"]
            for m in requests.get(f"{base}/v1/models", timeout=10).json()["data"]
        ]
        assert model not in names
