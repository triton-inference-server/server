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


def test_completions_sampling_parameters():
    prompt = "Hello"

    # Iterate through parameters within test to avoid constant server
    # startup/shutdown when using TestClient. This can likely be refactored.
    request_parameters = [
        ("temperature", 0.7),
        ("max_tokens", 10),
        ("top_p", 0.9),
        ("frequency_penalty", 0.5),
        ("presence_penalty", 0.2),
    ]

    with TestClient(app) as client:
        for parameter, value in request_parameters:
            response = client.post(
                "/v1/completions",
                json={"model": TEST_MODEL, "prompt": prompt, parameter: value},
            )

            print("Response:", response.json())
            assert response.status_code == 200
            # TODO: Flesh out or use dummy identity model
            assert response.json()["choices"][0]["text"].strip()


# Test for handling invalid prompt
def test_empty_prompt():
    with TestClient(app) as client:
        response = client.post(
            "/v1/completions", json={"model": TEST_MODEL, "prompt": ""}
        )

    # Assert
    assert response.status_code == 400
