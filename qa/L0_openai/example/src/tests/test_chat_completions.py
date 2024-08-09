import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.api_server import app

TEST_MODEL = "gpt2"
TEST_PROMPT = "What is the capital of France?"


# TODO: Test TRTLLM too
class TestChatCompletions:
    # TODO: Consider module/package scope, or join Completions tests into same file
    # to run server only once for both sets of tests for faster iteration.
    @pytest.fixture(scope="class", autouse=True)
    def client(self):
        # TODO: Test TRT-LLM models as well
        model_repository = Path(__file__).parent / "vllm_models"
        os.environ["TRITON_MODEL_REPOSITORY"] = str(model_repository)
        with TestClient(app) as test_client:
            yield test_client

    def test_chat_completions_defaults(self, client):
        messages = [{"role": "user", "content": TEST_PROMPT}]

        response = client.post(
            "/v1/chat/completions", json={"model": TEST_MODEL, "messages": messages}
        )

        assert response.status_code == 200
        message = response.json()["choices"][0]["message"]
        assert message["content"].strip()
        assert message["role"] == "assistant"

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
        messages = [{"role": "user", "content": TEST_PROMPT}]

        response = client.post(
            "/v1/chat/completions",
            json={"model": TEST_MODEL, "messages": messages, sampling_parameter: value},
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
        messages = [{"role": "user", "content": TEST_PROMPT}]

        response = client.post(
            "/v1/chat/completions",
            json={"model": TEST_MODEL, "messages": messages, sampling_parameter: value},
        )

        print("Response:", response.json())
        assert response.status_code == 422

    # Simple tests to verify max_tokens roughly behaves as expected
    def test_chat_completions_max_tokens(self, client):
        responses = []
        messages = [{"role": "user", "content": TEST_PROMPT}]
        payload = {"model": TEST_MODEL, "messages": messages, "max_tokens": 1}

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
    def test_chat_completions_temperature(self, client, temperature):
        responses = []
        messages = [{"role": "user", "content": TEST_PROMPT}]
        payload = {
            "model": TEST_MODEL,
            "messages": messages,
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

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_function_calling(self):
        pass

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_lora(self):
        pass

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_multi_lora(self):
        pass

    # TODO: Test for handling invalid messages or payloads
    # TODO: test chat/instruct model? gpt2 logs error about lack of chat template
    # TODO: test roles?
    # TODO: function calling?
    # TODO: lora / multi-lora?
    # TODO: genai-perf test?
