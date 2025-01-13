# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import copy
import subprocess
from pathlib import Path
from typing import List

import pytest
import tritonserver
from fastapi.testclient import TestClient
from tests.utils import setup_fastapi_app, setup_server


class TestChatCompletions:
    @pytest.fixture(scope="class")
    def client(self, fastapi_client_class_scope):
        yield fastapi_client_class_scope

    def test_chat_completions_defaults(self, client, model: str, messages: List[dict]):
        response = client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": messages},
        )

        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["content"].strip()
        assert message["role"] == "assistant"
        # "usage" currently not supported
        assert not response.json()["usage"]

    def test_chat_completions_system_prompt(self, client, model: str):
        # NOTE: Currently just sanity check that there are no issues when a
        # system role is provided. There is no test logic to measure the quality
        # of the response yet.
        messages = [
            {"role": "system", "content": "You are a Triton Inference Server expert."},
            {"role": "user", "content": "What is machine learning?"},
        ]

        response = client.post(
            "/v1/chat/completions", json={"model": model, "messages": messages}
        )

        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["content"].strip()
        assert message["role"] == "assistant"

    def test_chat_completions_system_prompt_only(self, client, model: str):
        # No user prompt provided
        messages = [
            {"role": "system", "content": "You are a Triton Inference Server expert."}
        ]

        response = client.post(
            "/v1/chat/completions", json={"model": model, "messages": messages}
        )

        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["content"].strip()
        assert message["role"] == "assistant"

    def test_chat_completions_user_prompt_str(self, client, model: str):
        # No system prompt provided
        messages = [{"role": "user", "content": "What is machine learning?"}]

        response = client.post(
            "/v1/chat/completions", json={"model": model, "messages": messages}
        )

        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["content"].strip()
        assert message["role"] == "assistant"

    def test_chat_completions_user_prompt_dict(self, client, model: str):
        # No system prompt provided
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is machine learning?"}],
            }
        ]

        response = client.post(
            "/v1/chat/completions", json={"model": model, "messages": messages}
        )

        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["content"].strip()
        assert message["role"] == "assistant"

    @pytest.mark.parametrize(
        "param_key, param_value",
        [
            ("temperature", 0.7),
            ("max_tokens", 10),
            ("top_p", 0.9),
            ("frequency_penalty", 0.5),
            ("presence_penalty", 0.2),
            ("n", 1),
            # Single stop word as a string
            ("stop", "."),
            # List of stop words
            ("stop", []),
            ("stop", [".", ","]),
            # logprobs is a boolean for chat completions
            ("logprobs", True),
            ("logit_bias", {"0": 0}),
            # NOTE: Extensions to the spec
            ("min_tokens", 16),
            ("ignore_eos", True),
        ],
    )
    def test_chat_completions_sampling_parameters(
        self, client, param_key, param_value, model: str, messages: List[dict]
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                param_key: param_value,
            },
        )

        # FIXME: Add support and remove this check
        unsupported_parameters = ["logprobs", "logit_bias"]
        if param_key in unsupported_parameters:
            assert response.status_code == 400
            assert (
                response.json()["detail"]
                == "logit bias and log probs not currently supported"
            )
            return

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"]
        assert response.json()["choices"][0]["message"]["role"] == "assistant"

    @pytest.mark.parametrize(
        "param_key, param_value",
        [
            ("temperature", 2.1),
            ("temperature", -0.1),
            ("max_tokens", -1),
            ("top_p", 1.1),
            ("frequency_penalty", 3),
            ("frequency_penalty", -3),
            ("presence_penalty", 2.1),
            ("presence_penalty", -2.1),
            # NOTE: Extensions to the spec
            ("min_tokens", -1),
            ("ignore_eos", 123),
        ],
    )
    def test_chat_completions_invalid_sampling_parameters(
        self, client, param_key, param_value, model: str, messages: List[dict]
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                param_key: param_value,
            },
        )
        print("Response:", response.json())

        # Assert schema validation error
        assert response.status_code == 422

    # Simple tests to verify max_tokens roughly behaves as expected
    def test_chat_completions_max_tokens(
        self, client, model: str, messages: List[dict]
    ):
        responses = []
        payload = {"model": model, "messages": messages, "max_tokens": 1}

        # Send two requests with max_tokens = 1 to check their similarity
        payload["max_tokens"] = 1
        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload,
            )
        )
        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload,
            )
        )
        # Send one requests with larger max_tokens to check its dis-similarity
        payload["max_tokens"] = 100
        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload,
            )
        )

        for response in responses:
            print("Response:", response.json())
            assert response.status_code == 200

        response1_text = (
            responses[0].json()["choices"][0]["message"]["content"].strip().split()
        )
        response2_text = (
            responses[1].json()["choices"][0]["message"]["content"].strip().split()
        )
        response3_text = (
            responses[2].json()["choices"][0]["message"]["content"].strip().split()
        )
        # Simplification: One token shouldn't be more than one space-delimited word
        assert len(response1_text) == len(response2_text) == 1
        assert len(response3_text) > len(response1_text)

    @pytest.mark.parametrize(
        "temperature",
        [0.0, 1.0],
    )
    # Simple tests to verify temperature roughly behaves as expected
    def test_chat_completions_temperature_vllm(
        self, client, temperature, backend: str, model: str, messages: List[dict]
    ):
        if backend != "vllm":
            pytest.skip(reason="Only used to test vLLM-specific temperature behavior")

        responses = []
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 256,
            "temperature": temperature,
        }

        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload,
            )
        )
        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload,
            )
        )

        for response in responses:
            print("Response:", response.json())
            assert response.status_code == 200

        response1_text = (
            responses[0].json()["choices"][0]["message"]["content"].strip().split()
        )
        response2_text = (
            responses[1].json()["choices"][0]["message"]["content"].strip().split()
        )

        # Temperature of 0.0 indicates greedy sampling, so check
        # that two equivalent requests produce the same response.
        if temperature == 0.0:
            # NOTE: This check may be ambitious to get an exact match in all
            # cases depending on how other parameter defaults are set, so
            # it can probably be removed if it introduces flakiness.
            assert response1_text == response2_text
        # Temperature of 1.0 indicates maximum randomness, so check
        # that two equivalent requests produce different responses.
        elif temperature == 1.0:
            assert response1_text != response2_text
        # Don't bother checking values other than the extremes
        else:
            raise ValueError(f"Unexpected {temperature=} for this test.")

    # Remove xfail when fix is released and this test returns xpass status
    @pytest.mark.xfail(
        reason="TRT-LLM BLS model will ignore temperature until a later release"
    )
    # Simple tests to verify temperature roughly behaves as expected
    def test_chat_completions_temperature_tensorrtllm(
        self, client, backend: str, model: str, messages: List[dict]
    ):
        if backend != "tensorrtllm":
            pytest.skip(
                reason="Only used to test TRT-LLM-specific temperature behavior"
            )

        responses = []
        payload1 = {
            "model": model,
            "messages": messages,
            # Increase token length to allow more room for variability
            "max_tokens": 200,
            "temperature": 0.0,
            # TRT-LLM requires certain settings of `top_k` / `top_p` to
            # respect changes in `temperature`
            "top_p": 0.5,
        }

        payload2 = copy.deepcopy(payload1)
        payload2["temperature"] = 1.0

        # First 2 responses should be the same in TRT-LLM with identical payload
        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload1,
            )
        )
        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload1,
            )
        )
        # Third response should differ with different temperature in payload
        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload2,
            )
        )

        for response in responses:
            print("Response:", response.json())
            assert response.status_code == 200

        response1_text = (
            responses[0].json()["choices"][0]["message"]["content"].strip().split()
        )
        response2_text = (
            responses[1].json()["choices"][0]["message"]["content"].strip().split()
        )
        response3_text = (
            responses[2].json()["choices"][0]["message"]["content"].strip().split()
        )

        assert response1_text == response2_text
        assert response1_text != response3_text

    # Simple tests to verify random seed roughly behaves as expected
    def test_chat_completions_seed(self, client, model: str, messages: List[dict]):
        responses = []
        payload1 = {
            "model": model,
            "messages": messages,
            # Increase token length to allow more room for variability
            "max_tokens": 200,
            "seed": 1,
        }
        payload2 = copy.deepcopy(payload1)
        payload2["seed"] = 2

        # First 2 responses should be the same in both vLLM and TRT-LLM with identical seed
        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload1,
            )
        )
        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload1,
            )
        )
        # Third response should differ with different seed in payload
        responses.append(
            client.post(
                "/v1/chat/completions",
                json=payload2,
            )
        )

        for response in responses:
            print("Response:", response.json())
            assert response.status_code == 200

        response1_text = (
            responses[0].json()["choices"][0]["message"]["content"].strip().split()
        )
        response2_text = (
            responses[1].json()["choices"][0]["message"]["content"].strip().split()
        )
        response3_text = (
            responses[2].json()["choices"][0]["message"]["content"].strip().split()
        )

        assert response1_text == response2_text
        assert response1_text != response3_text

    def test_chat_completions_no_message(
        self, client, model: str, messages: List[dict]
    ):
        # Message validation requires min_length of 1
        messages = []
        response = client.post(
            "/v1/chat/completions", json={"model": model, "messages": messages}
        )
        assert response.status_code == 422
        assert (
            response.json()["detail"][0]["msg"]
            == "List should have at least 1 item after validation, not 0"
        )

    def test_chat_completions_empty_message(
        self, client, model: str, messages: List[dict]
    ):
        # Message validation requires min_length of 1
        messages = [{}]
        response = client.post(
            "/v1/chat/completions", json={"model": model, "messages": messages}
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["msg"] == "Field required"

    def test_chat_completions_multiple_choices(
        self, client, model: str, messages: List[dict]
    ):
        response = client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": messages, "n": 2},
        )

        assert response.status_code == 400
        assert "only single choice" in response.json()["detail"]

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_chat_completions_streaming(self, client):
        pass

    def test_chat_completions_no_streaming(
        self, client, model: str, messages: List[dict]
    ):
        response = client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": messages, "stream": False},
        )

        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["content"].strip()
        assert message["role"] == "assistant"

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_function_calling(self):
        pass

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_lora(self):
        pass

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_multi_lora(self):
        pass

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_request_n_choices(self):
        pass

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_request_logprobs(self):
        pass

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_request_logit_bias(self):
        pass

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_usage_response(self):
        pass


# For tests that won't use the same pytest fixture for server startup across
# the whole class test suite.
class TestChatCompletionsTokenizers:
    # Re-use a single Triton server for different frontend configurations
    @pytest.fixture(scope="class")
    def server(self, model_repository: str):
        server = setup_server(model_repository)
        yield server
        server.stop()

    # A tokenizer must be known for /chat/completions endpoint in order to
    # apply chat templates, and for simplicity in determination, users should
    # define the tokenizer. So, explicitly raise an error if none is provided.
    def test_chat_completions_no_tokenizer(
        self,
        server: tritonserver.Server,
        backend: str,
        model: str,
        messages: List[dict],
    ):
        app = setup_fastapi_app(tokenizer="", server=server, backend=backend)
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": model, "messages": messages},
            )

        assert response.status_code == 400
        assert response.json()["detail"] == "Unknown tokenizer"

    def test_chat_completions_custom_tokenizer(
        self,
        server: tritonserver.Server,
        backend: str,
        tokenizer_model: str,
        model: str,
        messages: List[dict],
    ):
        # Tokenizers can be provided by a local file path to a directory containing
        # the relevant files such as tokenizer.json and tokenizer_config.json.
        custom_tokenizer_path = str(Path(__file__).parent / "custom_tokenizer")
        download_cmd = f"huggingface-cli download --local-dir {custom_tokenizer_path} {tokenizer_model} --include *.json"
        print(f"Running download command: {download_cmd}")
        subprocess.run(download_cmd.split(), check=True)

        # Compare the downloaded tokenizer response against remote HF equivalent
        # to assert equivalent functionality in responses and chat template.
        app_local = setup_fastapi_app(
            tokenizer=custom_tokenizer_path, server=server, backend=backend
        )
        app_hf = setup_fastapi_app(
            tokenizer=tokenizer_model, server=server, backend=backend
        )

        responses = []
        with TestClient(app_local) as client_local, TestClient(app_hf) as client_hf:
            payload = {"model": model, "messages": messages, "temperature": 0}
            responses.append(client_local.post("/v1/chat/completions", json=payload))
            responses.append(client_hf.post("/v1/chat/completions", json=payload))

        for response in responses:
            assert response.status_code == 200
            message = response.json()["choices"][0]["message"]
            assert message["content"].strip()
            assert message["role"] == "assistant"

        def equal_dicts(d1, d2, ignore_keys):
            d1_filtered = {k: v for k, v in d1.items() if k not in ignore_keys}
            d2_filtered = {k: v for k, v in d2.items() if k not in ignore_keys}
            return d1_filtered == d2_filtered

        ignore_keys = ["id", "created"]
        assert equal_dicts(
            responses[0].json(), responses[1].json(), ignore_keys=ignore_keys
        )

    def test_chat_completions_invalid_chat_tokenizer(
        self,
        server: tritonserver.Server,
        backend: str,
        model: str,
        messages: List[dict],
    ):
        # NOTE: Use of apply_chat_template on a tokenizer that doesn't support it
        # is a warning prior to transformers 4.44, and an error afterwards.
        # NOTE: Can remove after both TRT-LLM and VLLM containers have this version.
        import transformers

        print(f"{transformers.__version__=}")
        if transformers.__version__ < "4.44.0":
            pytest.xfail()

        # Pick a tokenizer with no chat template defined
        invalid_chat_tokenizer = "gpt2"
        try:
            app = setup_fastapi_app(
                tokenizer=invalid_chat_tokenizer, server=server, backend=backend
            )
        except OSError as e:
            expected_msg = f"We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like {invalid_chat_tokenizer} is not the path to a directory containing a file named config.json."
            if expected_msg in str(e):
                pytest.skip("HuggingFace network issues")
            raise e
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": model, "messages": messages},
            )

        assert response.status_code == 400
        # Error may vary based on transformers version
        expected_errors = [
            "cannot use apply_chat_template()",
            "cannot use chat template",
        ]
        assert any(
            error in response.json()["detail"].lower() for error in expected_errors
        )
