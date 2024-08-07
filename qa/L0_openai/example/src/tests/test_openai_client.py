from openai import OpenAI


# TODO: assumes already running server, so either refactor tests to work
# this way, or add TestClient to start server
def test_openai_client_completion():
    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    # openai_api_base = "http://localhost:8000/v1"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # TODO
    models = client.models.list()
    print(f"Models: {models}")
    model = models.data[0].id
    print(f"Model: {model}")

    completion = client.completions.create(
        prompt="Hi there",
        model=model,
    )

    assert completion
    print(f"Completion results: {completion}")


# TODO: assumes already running server, so either refactor tests to work
# this way, or add TestClient to start server
def test_openai_client_chat_completion():
    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    # openai_api_base = "http://localhost:8000/v1"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # TODO
    models = client.models.list()
    print(f"Models: {models}")
    model = models.data[0].id
    print(f"Model: {model}")

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {
                "role": "assistant",
                "content": "The Los Angeles Dodgers won the World Series in 2020.",
            },
            {"role": "user", "content": "Where was it played?"},
        ],
        model=model,
    )

    assert chat_completion
    assert chat_completion.choices
    assert chat_completion.choices[0]
    assert chat_completion.choices[0].finish_reason == "stop"
    print(f"Chat completion results: {chat_completion}")
