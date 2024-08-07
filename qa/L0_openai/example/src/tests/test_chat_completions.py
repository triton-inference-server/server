import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.api_server import app

TEST_MODEL = "gpt2"


# TODO: May need to modify fixture scope
@pytest.fixture(scope="function", autouse=True)
def setup_model_repository():
    model_repository = Path(__file__).parent / "vllm_models"
    os.environ["TRITON_MODEL_REPOSITORY"] = str(model_repository)


# Test for Chat Completions API
def test_successful_chat_completion():
    messages = [{"role": "user", "content": "Hello"}]

    # TODO: test various parameters
    # TODO: test chat template - gpt2 raises error?
    # TODO: test roles?
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json={"model": TEST_MODEL, "messages": messages}
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"]
    # TODO: Double check expected role
    assert response.json()["choices"][0]["message"]["role"] == "assistant"


# TODO: Test for handling invalid messages or payloads
