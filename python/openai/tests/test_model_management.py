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
import http.client
import os
import random
import time
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests
from tests.utils import OpenAIServer

TEST_MODEL_REPOSITORY = str(Path(__file__).parent / "test_models")
TEST_MODEL = "mock_llm"
TEST_MODEL_2 = "identity_py"


# Test "--load-model" and "--model-control-mode" CLI options
@pytest.mark.openai
class TestModelControlCLIOptions:
    @staticmethod
    def _assert_server_launch_fails(args, expected_error: str):
        """Helper: verify server fails to start and stderr contains expected_error."""
        with pytest.raises(Exception) as exc_info:
            with OpenAIServer(args):
                pass
        assert expected_error in str(exc_info.value)

    def test_non_explicit_mode_load_model_error(self):
        """--load-model without --model-control-mode=explicit must exit with error.
        Error message matches native tritonserver exactly."""
        self._assert_server_launch_fails(
            args=[
                "--model-repository",
                TEST_MODEL_REPOSITORY,
                "--load-model",
                TEST_MODEL,
            ],
            expected_error="Error: Use of '--load-model' requires setting '--model-control-mode=explicit' as well.",
        )

    def test_explicit_mode_load_zero_model(self):
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

    def test_explicit_mode_load_one_model(self):
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

    def test_explicit_mode_load_multiple_models(self):
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

    def test_explicit_mode_load_all_models(self):
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

    def test_explicit_mode_load_all_models_and_specific_model_error(self):
        self._assert_server_launch_fails(
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

    def test_explicit_mode_load_nonexistent_model_error(self):
        self._assert_server_launch_fails(
            args=[
                "--model-repository",
                TEST_MODEL_REPOSITORY,
                "--model-control-mode",
                "explicit",
                "--load-model",
                "nonexistent_model",
            ],
            expected_error="failed to poll model 'nonexistent_model': model not found in any model repository",
        )

    def test_explicit_mode_load_invalid_model_name_error(self):
        invalid_model_names = [
            (
                os.path.relpath("/etc", TEST_MODEL_REPOSITORY),
                "at least one version must be available under the version policy",
            ),
            (
                os.path.relpath("/etc/passwd", TEST_MODEL_REPOSITORY),
                "Poll failed for model directory",
            ),
            ("model/..", "model not found in any model repository"),
            ("..", "at least one version must be available under the version policy"),
            ("/etc/passwd", "model not found in any model repository"),
            ("", "Invalid model name"),
            ("  ", "model not found in any model repository"),
            ("\n\t", "model not found in any model repository"),
        ]
        for model_name, expected_error in invalid_model_names:
            self._assert_server_launch_fails(
                args=[
                    "--model-repository",
                    TEST_MODEL_REPOSITORY,
                    "--model-control-mode",
                    "explicit",
                    "--load-model",
                    model_name,
                ],
                expected_error=expected_error,
            )


class _ModelManagementBase:
    @pytest.fixture(scope="class")
    def base_url(self, server: OpenAIServer):
        return server.url_root

    @pytest.fixture
    def all_models(self) -> list[str]:
        return [TEST_MODEL, TEST_MODEL_2]

    @pytest.fixture(scope="class")
    def server(
        self,
        model_repository: str,
        tokenizer_model: str,
        backend: str,
        model_control_mode: str,
    ):
        args = [
            "--model-repository",
            model_repository,
        ]
        if tokenizer_model:
            args += ["--tokenizer", tokenizer_model]
        if backend:
            args += ["--backend", backend]
        if model_control_mode:
            args += ["--model-control-mode", model_control_mode]

        with OpenAIServer(args) as openai_server:
            yield openai_server

    @pytest.fixture(autouse=True)
    def _cleanup(self, base_url, all_models: list[str]):
        """Ensure clean state before and after every test by unloading the models."""
        for name in all_models:
            requests.post(f"{base_url}/v1/models/{name}/unload")
        yield
        for name in all_models:
            requests.post(f"{base_url}/v1/models/{name}/unload")

    @staticmethod
    def _list_available_models(base_url: str) -> list[str]:
        response = requests.get(f"{base_url}/v1/models")
        assert response.status_code == 200
        return [m["id"] for m in response.json()["data"]]

    @staticmethod
    def _assert_unknown_model(response: requests.Response):
        assert response.status_code == 400
        assert "unknown model" in response.json()["detail"].lower()


class TestModelControlModeNone(_ModelManagementBase):
    """Test NONE mode rejects load/unload API calls."""

    @pytest.fixture(scope="class")
    def model_repository(self):
        return TEST_MODEL_REPOSITORY

    @pytest.fixture(scope="class")
    def model_control_mode(self, request):
        return request.param

    @pytest.mark.parametrize("model_control_mode", [None, "none"], indirect=True)
    def test_load_and_unload_rejected(self, base_url):
        """Test NONE mode rejects load/unload API calls."""

        for api in ["load", "unload"]:
            response = requests.post(f"{base_url}/v1/models/{TEST_MODEL}/{api}")
            assert response.status_code == 400
            assert (
                "model load/unload requires --model-control-mode=explicit"
                in response.json()["detail"].lower()
            )


class TestModelManagement(_ModelManagementBase):
    """Test load/unload operations in EXPLICIT mode."""

    @pytest.fixture(scope="class")
    def model_control_mode(self):
        return "explicit"

    @pytest.fixture(scope="class")
    def model_repository(self):
        return TEST_MODEL_REPOSITORY

    @staticmethod
    def _assert_model_metadata(model_data: dict):
        assert model_data["id"]
        assert model_data["object"] == "model"
        assert model_data["created"] > 0
        assert model_data["owned_by"] == "Triton Inference Server"

    def test_load_model(self, base_url):
        # Server should start with no models loaded
        assert self._list_available_models(base_url) == []

        response = requests.post(f"{base_url}/v1/models/{TEST_MODEL}/load")
        assert response.status_code == 200
        self._assert_model_metadata(response.json())
        assert response.json()["id"] == TEST_MODEL
        assert TEST_MODEL in self._list_available_models(base_url)

        response = requests.get(f"{base_url}/v1/models/{TEST_MODEL}")
        assert response.status_code == 200
        self._assert_model_metadata(response.json())

    def test_unload_model(self, base_url):
        assert (
            requests.post(f"{base_url}/v1/models/{TEST_MODEL}/load").status_code == 200
        )

        response = requests.post(f"{base_url}/v1/models/{TEST_MODEL}/unload")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["model"] == TEST_MODEL

        assert TEST_MODEL not in self._list_available_models(base_url)
        assert requests.get(f"{base_url}/v1/models/{TEST_MODEL}").status_code == 404

    def test_load_rejects_duplicate(self, base_url):
        assert (
            requests.post(f"{base_url}/v1/models/{TEST_MODEL}/load").status_code == 200
        )

        response = requests.post(f"{base_url}/v1/models/{TEST_MODEL}/load")
        assert response.status_code == 400
        assert "already loaded" in response.json()["detail"].lower()

    def test_load_unload_unknown_model(self, base_url):
        response = requests.post(f"{base_url}/v1/models/unknown_model/load")
        assert response.status_code == 500
        assert "failed to poll from model repository" in response.json()["detail"]

        response = requests.post(f"{base_url}/v1/models/unknown_model/unload")
        self._assert_unknown_model(response)

    def test_load_unload_invalid_model_name(self, base_url):
        invalid_model_names = [
            (os.path.relpath("/etc", TEST_MODEL_REPOSITORY), 404),
            (os.path.relpath("/etc/passwd", TEST_MODEL_REPOSITORY), 404),
            ("model/..", 404),
            ("..", 400),
            ("/etc/passwd", 404),
            ("model/subdir", 404),
            ("model/", 404),
            ("", 404),
            ("%20%20", 400),
            ("%0A%09", 400),
        ]
        parsed = urlparse(base_url)
        for model_name, expected_status in invalid_model_names:
            for endpoint in ["load", "unload"]:
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port)
                conn.request("POST", f"/v1/models/{model_name}/{endpoint}")
                response = conn.getresponse()

                assert response.status == expected_status, (
                    f"Expected {expected_status} for model name {model_name!r}, "
                    f"got {response.status} {response.read().decode()}"
                )

    def test_load_unload_reload(self, base_url):
        assert (
            requests.post(f"{base_url}/v1/models/{TEST_MODEL}/load").status_code == 200
        )
        assert TEST_MODEL in self._list_available_models(base_url)

        assert (
            requests.post(f"{base_url}/v1/models/{TEST_MODEL}/unload").status_code
            == 200
        )
        assert TEST_MODEL not in self._list_available_models(base_url)

        assert (
            requests.post(f"{base_url}/v1/models/{TEST_MODEL}/load").status_code == 200
        )
        assert TEST_MODEL in self._list_available_models(base_url)

    def test_load_multiple_models(self, base_url):
        assert (
            requests.post(f"{base_url}/v1/models/{TEST_MODEL}/load").status_code == 200
        )
        assert (
            requests.post(f"{base_url}/v1/models/{TEST_MODEL_2}/load").status_code
            == 200
        )

        names = self._list_available_models(base_url)
        assert TEST_MODEL in names
        assert TEST_MODEL_2 in names

        # Unload the first model
        assert (
            requests.post(f"{base_url}/v1/models/{TEST_MODEL}/unload").status_code
            == 200
        )
        names = self._list_available_models(base_url)
        assert TEST_MODEL not in names
        assert TEST_MODEL_2 in names

        # Unload the second model
        assert (
            requests.post(f"{base_url}/v1/models/{TEST_MODEL_2}/unload").status_code
            == 200
        )
        names = self._list_available_models(base_url)
        assert TEST_MODEL not in names
        assert TEST_MODEL_2 not in names


class TestConcurrentModelManagement(_ModelManagementBase):
    """Test sequential and concurrent load/unload."""

    @pytest.fixture(scope="class")
    def model_repository(self):
        return TEST_MODEL_REPOSITORY

    @pytest.fixture(scope="class")
    def model_control_mode(self):
        return "explicit"

    def test_concurrent_load_model(self, base_url):
        futures = []
        concurrency = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            for _ in range(concurrency):
                futures.append(
                    pool.submit(
                        requests.post, f"{base_url}/v1/models/{TEST_MODEL}/load"
                    )
                )

        codes = sorted([future.result().status_code for future in futures])
        assert codes == [200] + [400] * (
            concurrency - 1
        ), f"Expected one 200 and the rest 400 for concurrent loads, got {codes}"
        assert TEST_MODEL in self._list_available_models(base_url)

    def test_concurrent_unload_model(self, base_url):
        # Load the model
        requests.post(f"{base_url}/v1/models/{TEST_MODEL}/load")
        assert TEST_MODEL in self._list_available_models(base_url)

        futures = []
        concurrency = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            for _ in range(concurrency):
                futures.append(
                    pool.submit(
                        requests.post, f"{base_url}/v1/models/{TEST_MODEL}/unload"
                    )
                )

        codes = sorted([future.result().status_code for future in futures])
        assert codes == [200] + [400] * (
            concurrency - 1
        ), f"Expected one 200 and the rest 400 for concurrent unloads, got {codes}"
        assert TEST_MODEL not in self._list_available_models(base_url)

    def test_concurrent_load_multiple_models(self, base_url):
        concurrency = 10
        tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            for _ in range(concurrency // 2):
                for model_name in [TEST_MODEL, TEST_MODEL_2]:
                    tasks.append(
                        pool.submit(
                            requests.post, f"{base_url}/v1/models/{model_name}/load"
                        )
                    )

        codes = sorted([task.result().status_code for task in tasks])
        assert codes == [200, 200] + [400] * (
            concurrency - 2
        ), f"Expected two 200s and the rest 400 for concurrent loads, got {codes}"
        available_models = self._list_available_models(base_url)
        assert TEST_MODEL in available_models
        assert TEST_MODEL_2 in available_models

    def test_concurrent_unload_multiple_models(self, base_url):
        # Load both models
        for model_name in [TEST_MODEL, TEST_MODEL_2]:
            requests.post(f"{base_url}/v1/models/{model_name}/load")
            assert model_name in self._list_available_models(base_url)

        concurrency = 10
        tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            for _ in range(concurrency // 2):
                for model_name in [TEST_MODEL, TEST_MODEL_2]:
                    tasks.append(
                        pool.submit(
                            requests.post, f"{base_url}/v1/models/{model_name}/unload"
                        )
                    )

        codes = sorted([task.result().status_code for task in tasks])
        assert codes == [200, 200] + [400] * (
            concurrency - 2
        ), f"Expected two 200s and the rest 400 for concurrent unloads, got {codes}"
        available_models = self._list_available_models(base_url)
        for model_name in [TEST_MODEL, TEST_MODEL_2]:
            assert model_name not in available_models

    def test_concurrent_load_unload_stress(self, base_url):
        futures = []
        concurrency = 50
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            for _ in range(concurrency):
                action = random.choice(["load", "unload"])
                model_name = random.choice([TEST_MODEL, TEST_MODEL_2])
                futures.append(
                    pool.submit(
                        requests.post, f"{base_url}/v1/models/{model_name}/{action}"
                    )
                )
        done, _ = concurrent.futures.wait(
            futures, return_when=concurrent.futures.ALL_COMPLETED
        )
        assert (
            len(done) == concurrency
        ), f"Expected {concurrency} requests to be completed, got {len(done)}"
        for future in done:
            response = future.result()
            assert response.status_code in (
                200,
                400,
            ), f"Unexpected status code: {response.status_code}"

        # Wait for server to be ready
        for _ in range(10):
            response = requests.get(f"{base_url}/health/ready")
            if response.status_code == 200:
                break
            time.sleep(1)
        assert response.status_code == 200


# Test Inference with real LLM backend (vLLM / TRT-LLM) after load/unload
@pytest.mark.openai
class TestModelManagementInference(_ModelManagementBase):
    @pytest.fixture(scope="class")
    def model_control_mode(self):
        return "explicit"

    @pytest.fixture
    def all_models(self, model: str) -> list[str]:
        if model == "tensorrt_llm_bls":
            return ["postprocessing", "preprocessing", "tensorrt_llm", model]
        else:
            return [model]

    @staticmethod
    def _assert_load(base_url, model_list: list[str]):
        for model_name in model_list:
            r = requests.post(
                f"{base_url}/v1/models/{model_name}/load",
            )
            assert r.status_code == 200, f"Load {model_name} failed: {r.text}"

    @staticmethod
    def _assert_unload(base_url, model_list: list[str]):
        for model_name in model_list:
            r = requests.post(
                f"{base_url}/v1/models/{model_name}/unload",
            )
            assert r.status_code == 200, f"Unload {model_name} failed: {r.text}"

    @staticmethod
    def _assert_usage(usage):
        assert usage is not None
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )

    @staticmethod
    def _completions(base_url, model_name: str, **kwargs):
        return requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model_name,
                "prompt": "What is machine learning?",
                "max_tokens": 10,
                **kwargs,
            },
        )

    def test_load_completions(self, base_url, all_models: list[str], model: str):
        self._assert_unknown_model(self._completions(base_url, model))

        self._assert_load(base_url, all_models)
        r = self._completions(base_url, model)
        assert r.status_code == 200
        data = r.json()
        assert data["choices"][0]["text"].strip()
        assert data["choices"][0]["finish_reason"] == "stop"
        self._assert_usage(data["usage"])

        self._assert_unload(base_url, all_models)

    def test_unload_rejects_inference(
        self, base_url, all_models: list[str], model: str
    ):
        self._assert_load(base_url, all_models)
        assert self._completions(base_url, model).status_code == 200

        self._assert_unload(base_url, all_models)
        self._assert_unknown_model(self._completions(base_url, model))

    def test_reload_inference(self, base_url, all_models: list[str], model: str):
        self._assert_load(base_url, all_models)
        assert self._completions(base_url, model).status_code == 200

        self._assert_unload(base_url, all_models)
        self._assert_unknown_model(self._completions(base_url, model))

        self._assert_load(base_url, all_models)
        r = self._completions(base_url, model)
        assert r.status_code == 200
        assert r.json()["choices"][0]["text"].strip()
