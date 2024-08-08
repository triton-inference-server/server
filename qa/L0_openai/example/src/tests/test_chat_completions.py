import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.api_server import app

TEST_MODEL = "gpt2"


# TODO: Test TRTLLM too
class TestChatCompletion:
    # TODO: Consider module/package scope, or join Completions tests into same file
    # to run server only once for both sets of tests for faster iteration.
    @pytest.fixture(scope="class", autouse=True)
    def client(self):
        model_repository = Path(__file__).parent / "vllm_models"
        os.environ["TRITON_MODEL_REPOSITORY"] = str(model_repository)
        with TestClient(app) as test_client:
            yield test_client

    def test_chat_completion_defaults(self, client):
        messages = [{"role": "user", "content": "Hello"}]

        response = client.post(
            "/v1/chat/completions", json={"model": TEST_MODEL, "messages": messages}
        )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"]
        # TODO: Need to test different roles?
        assert response.json()["choices"][0]["message"]["role"] == "assistant"

    def test_chat_completion_parameters(self, client):
        messages = [{"role": "user", "content": "Hello"}]

        # Iterate through parameters within test to avoid constant server
        # startup/shutdown when using TestClient. This can likely be refactored.
        request_parameters = [
            ("temperature", 0.7),
            ("max_tokens", 10),
            ("top_p", 0.9),
            ("frequency_penalty", 0.5),
            ("presence_penalty", 0.2),
        ]

        for parameter, value in request_parameters:
            response = client.post(
                "/v1/chat/completions",
                json={"model": TEST_MODEL, "messages": messages, parameter: value},
            )

            assert response.status_code == 200
            assert response.json()["choices"][0]["message"]["content"]
            assert response.json()["choices"][0]["message"]["role"] == "assistant"

    def test_chat_completion_no_messages(self, client):
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

    def test_chat_completion_empty_message(self, client):
        # Message validation requires min_length of 1
        messages = [{}]
        response = client.post(
            "/v1/chat/completions", json={"model": TEST_MODEL, "messages": messages}
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["msg"] == "Field required"

    # TODO: Test for handling invalid messages or payloads
    # TODO: test chat/instruct model? gpt2 logs error about lack of chat template
    # TODO: test roles?
    # TODO: function calling?
    # TODO: lora / multi-lora?
