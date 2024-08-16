import copy
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.api_server import init_app

### TEST ENVIRONMENT SETUP ###
TEST_BACKEND = ""
TEST_MODEL = ""
TEST_PROMPT = "What is machine learning?"
TEST_MESSAGES = [{"role": "user", "content": TEST_PROMPT}]
TEST_TOKENIZER = "meta-llama/Meta-Llama-3-8B-Instruct"
try:
    import vllm as _

    TEST_BACKEND = "vllm"
    TEST_MODEL = "llama-3-8b-instruct"
except ImportError:
    pass

try:
    import tensorrt_llm as _

    TEST_BACKEND = "tensorrtllm"
    TEST_MODEL = "tensorrt_llm_bls"
except ImportError:
    pass

if not TEST_BACKEND or not TEST_MODEL:
    raise Exception("Unknown test environment")
###


class TestChatCompletions:
    # TODO: Consider module/package scope, or join Completions tests into same file
    # to run server only once for both sets of tests for faster iteration.
    @pytest.fixture(scope="class")
    def client(self):
        model_repository = str(Path(__file__).parent / f"{TEST_BACKEND}_models")
        app = self.setup_app(
            tokenizer=TEST_TOKENIZER, model_repository=model_repository
        )
        with TestClient(app) as test_client:
            yield test_client

    def setup_app(self, tokenizer: str, model_repository: str):
        os.environ["TOKENIZER"] = tokenizer
        os.environ["TRITON_MODEL_REPOSITORY"] = model_repository
        app = init_app()
        return app

    def test_chat_completions_defaults(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={"model": TEST_MODEL, "messages": TEST_MESSAGES},
        )

        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["content"].strip()
        assert message["role"] == "assistant"
        # "usage" currently not supported
        assert response.json()["usage"] == None

    def test_chat_completions_system_prompt(self, client):
        # NOTE: Currently just sanity check that there are no issues when a
        # system role is provided. There is no test logic to measure the quality
        # of the response yet.
        messages = [
            {"role": "system", "content": "You are a Triton Inference Server expert."},
            {"role": "user", "content": TEST_PROMPT},
        ]

        response = client.post(
            "/v1/chat/completions", json={"model": TEST_MODEL, "messages": messages}
        )

        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["content"].strip()
        assert message["role"] == "assistant"

    def test_chat_completions_system_prompt_only(self, client):
        # No user prompt provided
        messages = [
            {"role": "system", "content": "You are a Triton Inference Server expert."}
        ]

        response = client.post(
            "/v1/chat/completions", json={"model": TEST_MODEL, "messages": messages}
        )

        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["content"].strip()
        assert message["role"] == "assistant"

    @pytest.mark.parametrize(
        "sampling_parameter, value",
        [
            ("temperature", 0.7),
            ("max_tokens", 10),
            ("top_p", 0.9),
            ("frequency_penalty", 0.5),
            ("presence_penalty", 0.2),
            # logprobs is a boolean for chat completions
            ("logprobs", True),
            ("logit_bias", {"0": 0}),
        ],
    )
    def test_chat_completions_sampling_parameters(
        self, client, sampling_parameter, value
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": TEST_MODEL,
                "messages": TEST_MESSAGES,
                sampling_parameter: value,
            },
        )

        # TODO: Add support and remove this check
        unsupported_parameters = ["logprobs", "logit_bias"]
        if sampling_parameter in unsupported_parameters:
            assert response.status_code == 400
            assert response.json()["detail"] == "logit bias and log probs not supported"
            return

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"]
        assert response.json()["choices"][0]["message"]["role"] == "assistant"

    @pytest.mark.parametrize(
        "sampling_parameter, value",
        [
            ("temperature", 2.1),
            ("temperature", -0.1),
            ("max_tokens", -1),
            ("top_p", 1.1),
            ("frequency_penalty", 3),
            ("frequency_penalty", -3),
            ("presence_penalty", 2.1),
            ("presence_penalty", -2.1),
        ],
    )
    def test_chat_completions_invalid_sampling_parameters(
        self, client, sampling_parameter, value
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": TEST_MODEL,
                "messages": TEST_MESSAGES,
                sampling_parameter: value,
            },
        )

        print("Response:", response.json())
        assert response.status_code == 422

    # Simple tests to verify max_tokens roughly behaves as expected
    def test_chat_completions_max_tokens(self, client):
        responses = []
        payload = {"model": TEST_MODEL, "messages": TEST_MESSAGES, "max_tokens": 1}

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

    @pytest.mark.skipif(TEST_BACKEND != "vllm", reason="Only used to test vLLM backend")
    @pytest.mark.parametrize(
        "temperature",
        [0.0, 1.0],
    )
    # Simple tests to verify temperature roughly behaves as expected
    def test_chat_completions_temperature_vllm(self, client, temperature):
        responses = []
        payload = {
            "model": TEST_MODEL,
            "messages": TEST_MESSAGES,
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
            # frameworks depending on how other parameter defaults are set, so
            # it can probably be removed if it introduces flakiness.
            print(f"Comparing '{response1_text}' == '{response2_text}'")
            assert response1_text == response2_text
        # Temperature of 1.0 indicates maximum randomness, so check
        # that two equivalent requests produce different responses.
        elif temperature == 1.0:
            print(f"Comparing '{response1_text}' != '{response2_text}'")
            assert response1_text != response2_text
        # Don't bother checking values other than the extremes
        else:
            raise ValueError(f"Unexpected {temperature=} for this test.")

    # Remove xfail when fix is released and this test returns xpass status
    @pytest.mark.xfail(
        reason="TRT-LLM BLS model will ignore temperature until a later release"
    )
    @pytest.mark.skipif(
        TEST_BACKEND != "tensorrtllm", reason="Only used to test TRT-LLM backend"
    )
    # Simple tests to verify temperature roughly behaves as expected
    def test_chat_completions_temperature_tensorrtllm(self, client):
        responses = []
        payload1 = {
            "model": TEST_MODEL,
            "messages": TEST_MESSAGES,
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
    def test_chat_completions_seed(self, client):
        responses = []
        payload1 = {
            "model": TEST_MODEL,
            "messages": TEST_MESSAGES,
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

    def test_chat_completions_no_message(self, client):
        # Message validation requires min_length of 1
        messages = []
        response = client.post(
            "/v1/chat/completions", json={"model": TEST_MODEL, "messages": messages}
        )
        assert response.status_code == 422
        assert (
            response.json()["detail"][0]["msg"]
            == "List should have at least 1 item after validation, not 0"
        )

    def test_chat_completions_empty_message(self, client):
        # Message validation requires min_length of 1
        messages = [{}]
        response = client.post(
            "/v1/chat/completions", json={"model": TEST_MODEL, "messages": messages}
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["msg"] == "Field required"

    def test_chat_completions_multiple_choices(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={"model": TEST_MODEL, "messages": TEST_MESSAGES, "n": 2},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Only single choice is supported"

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_chat_completions_streaming(self, client):
        pass

    def test_chat_completions_no_streaming(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={"model": TEST_MODEL, "messages": TEST_MESSAGES, "stream": False},
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

    # TODO: Do we want to support "usage" field for token counts in response?
    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_usage_response(self):
        pass


# For tests that won't use the same pytest fixture for server startup across
# the whole class test suite.
class TestChatCompletionsCustomFixture:
    def setup_app(self, tokenizer: str, model_repository: str):
        os.environ["TOKENIZER"] = tokenizer
        os.environ["TRITON_MODEL_REPOSITORY"] = model_repository
        app = init_app()
        return app

    # A TOKENIZER must be known for /chat/completions endpoint in order to
    # apply chat templates, and for simplicity in determination, users should
    # define the TOKENIZER. So, explicitly raise an error if none is provided.
    def test_chat_completions_no_tokenizer(self):
        model_repository = str(Path(__file__).parent / f"{TEST_BACKEND}_models")
        app = self.setup_app(tokenizer="", model_repository=model_repository)
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": TEST_MODEL, "messages": TEST_MESSAGES},
            )
            assert response.status_code == 400
            assert response.json()["detail"] == "Unknown tokenizer"
