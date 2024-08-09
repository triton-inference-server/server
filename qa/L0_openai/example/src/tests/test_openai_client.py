import os
from pathlib import Path

import pytest
from openai import OpenAI


class TestOpenAIClient:
    # Start server, then with scope="class", pass execution back to each test
    # until all tests in the class have been run, then clean up.
    # TODO: OpenAI client requires server is already running
    @pytest.fixture(scope="class", autouse=True)
    def start_server(self):
        model_repository = Path(__file__).parent / "vllm_models"
        os.environ["TRITON_MODEL_REPOSITORY"] = str(model_repository)

        # TODO: Start server in background
        # ex: https://github.com/vllm-project/vllm/blob/main/tests/utils.py
        # proc = subprocess.run(...)
        yield
        # proc.terminate()
        # proc.wait()
        # proc.kill()

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_openai_client_completion(self):
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

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

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_openai_client_chat_completion(self):
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

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

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_openai_client_function_calling(self):
        pass
