from pathlib import Path

import openai
import pytest
from src.tests.utils import OpenAIServer

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


# NOTE: OpenAI client requires actual server running, and won't work
# with the FastAPI TestClient. Run the server at module scope to run
# only once for all the tests below.
@pytest.fixture(scope="module")
def server():
    model_repository = Path(__file__).parent / f"{TEST_BACKEND}_models"
    tokenizer = "meta-llama/Meta-Llama-3-8B-Instruct"
    args = ["--model-repository", model_repository, "--tokenizer", tokenizer]

    with OpenAIServer(args) as openai_server:
        yield openai_server


class TestOpenAIClient:
    @pytest.fixture(scope="class")
    def client(self, server):
        return server.get_client()

    def test_openai_client_models(self, client: openai.OpenAI):
        models = list(client.models.list())
        print(f"Models: {models}")
        if TEST_BACKEND == "tensorrtllm":
            # ensemble or tensorrt_llm_bls
            # preprocess -> tensorrt_llm -> postprocess
            assert len(models) == 5
        elif TEST_BACKEND == "vllm":
            assert len(models) == 1
        else:
            raise Exception(f"Unexpected backend {TEST_BACKEND=}")

    def test_openai_client_completion(self, client: openai.OpenAI):
        completion = client.completions.create(
            prompt=TEST_PROMPT,
            model=TEST_MODEL,
        )

        print(f"Completion results: {completion}")
        assert completion.choices[0].text
        assert completion.choices[0].finish_reason == "stop"

    def test_openai_client_chat_completion(self, client: openai.OpenAI):
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
            model=TEST_MODEL,
        )

        print(f"Chat completion results: {chat_completion}")
        assert chat_completion.choices[0].message.content
        assert chat_completion.choices[0].finish_reason == "stop"

    @pytest.mark.skipif(
        TEST_BACKEND == "tensorrtllm",
        reason="TRT-LLM backend currently only supports setting this parameter at model load time",
    )
    @pytest.mark.parametrize("echo", [False, True])
    def test_openai_client_completion_echo(self, client: openai.OpenAI, echo: bool):
        prompt = "What is the capital of France?"
        completion = client.completions.create(
            prompt=prompt, model=TEST_MODEL, echo=echo
        )

        print(f"Completion results: {completion}")
        response = completion.choices[0].text
        if echo:
            assert prompt in response
        else:
            assert prompt not in response

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_openai_client_function_calling(self):
        pass


class TestAsyncOpenAIClient:
    @pytest.fixture(scope="class")
    def client(self, server):
        return server.get_async_client()

    @pytest.mark.asyncio
    async def test_openai_client_models(self, client: openai.AsyncOpenAI):
        async_models = await client.models.list()
        models = [model async for model in async_models]
        print(f"Models: {models}")
        if TEST_BACKEND == "tensorrtllm":
            # ensemble or tensorrt_llm_bls
            # preprocess -> tensorrt_llm -> postprocess
            assert len(models) == 5
        elif TEST_BACKEND == "vllm":
            assert len(models) == 1
        else:
            raise Exception(f"Unexpected backend {TEST_BACKEND=}")

    @pytest.mark.asyncio
    async def test_openai_client_completion(self, client: openai.AsyncOpenAI):
        completion = await client.completions.create(
            prompt=TEST_PROMPT,
            model=TEST_MODEL,
        )

        print(f"Completion results: {completion}")
        assert completion.choices[0].text
        assert completion.choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_openai_client_chat_completion(self, client: openai.AsyncOpenAI):
        chat_completion = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {
                    "role": "assistant",
                    "content": "The Los Angeles Dodgers won the World Series in 2020.",
                },
                {"role": "user", "content": "Where was it played?"},
            ],
            model=TEST_MODEL,
        )

        assert chat_completion.choices[0].message.content
        assert chat_completion.choices[0].finish_reason == "stop"
        print(f"Chat completion results: {chat_completion}")

    # TODO: Add this test
    @pytest.mark.skip(reason="Not Implemented Yet")
    @pytest.mark.asyncio
    async def test_completion_streaming(self, client: openai.AsyncOpenAI):
        pass

    @pytest.mark.asyncio
    async def test_chat_streaming(self, client: openai.AsyncOpenAI):
        messages = [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "what is 1+1?"},
        ]

        # test single completion
        chat_completion = await client.chat.completions.create(
            model=TEST_MODEL,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
            stream=False,
        )
        output = chat_completion.choices[0].message.content
        stop_reason = chat_completion.choices[0].finish_reason

        # test streaming
        stream = await client.chat.completions.create(
            model=TEST_MODEL,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
            stream=True,
        )
        chunks = []
        finish_reason_count = 0
        async for chunk in stream:
            delta = chunk.choices[0].delta
            print("[DEBUG] DELTA:", delta)
            if delta.role:
                assert delta.role == "assistant"
            if delta.content:
                chunks.append(delta.content)
            if chunk.choices[0].finish_reason is not None:
                finish_reason_count += 1
        # finish reason should only return in last block
        assert finish_reason_count == 1
        assert chunk.choices[0].finish_reason == stop_reason
        assert "".join(chunks) == output

    @pytest.mark.skip(reason="Not Implemented Yet")
    @pytest.mark.asyncio
    async def test_openai_client_function_calling(self):
        pass
