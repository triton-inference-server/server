import pytest
from fastapi.testclient import TestClient
from src.api_server import app


# Test for Chat Completions API
@pytest.mark.parametrize(
    "sampling_parameter, value",
    [
        ("temperature", 0.7),
        ("max_tokens", 10),
        ("top_p", 0.9),
        ("frequency_penalty", 0.5),
        ("presence_penalty", 0.2),
    ],
)
def test_chat_completions_sampling_parameters(sampling_parameter, value):
    # Arrange
    messages = [{"role": "user", "content": "Hello"}]
    expected_response = "Hi there"

    # Act
    with TestClient(app) as client:
        response = client.post(
            "/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "messages": messages,
                sampling_parameter: value,
            },
        )

    # Assert
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == expected_response


# Test for handling invalid chat input
def test_invalid_chat_input():
    # Act
    with TestClient(app) as client:
        response = client.post(
            "/chat/completions", json={"model": "gpt-3.5-turbo", "messages": []}
        )

    # Assert
    assert response.status_code == 400
