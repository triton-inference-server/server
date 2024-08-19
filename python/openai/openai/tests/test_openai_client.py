from typing import List

import openai
import pytest


class TestOpenAIClient:
    @pytest.fixture(scope="class")
    def client(self, server):
        return server.get_client()

    def test_openai_client_models(self, client: openai.OpenAI, backend: str):
        models = list(client.models.list())
        print(f"Models: {models}")
        if backend == "tensorrtllm":
            # ensemble or tensorrt_llm_bls
            # preprocess -> tensorrt_llm -> postprocess
            assert len(models) == 5
        elif backend == "vllm":
            assert len(models) == 1
        else:
            raise Exception(f"Unexpected backend {backend=}")

    def test_openai_client_completion(
        self, client: openai.OpenAI, model: str, prompt: str
    ):
        completion = client.completions.create(
            prompt=prompt,
            model=model,
        )

        print(f"Completion results: {completion}")
        assert completion.choices[0].text
        assert completion.choices[0].finish_reason == "stop"

    def test_openai_client_chat_completion(
        self, client: openai.OpenAI, model: str, messages: List[dict]
    ):
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
        )

        print(f"Chat completion results: {chat_completion}")
        assert chat_completion.choices[0].message.content
        assert chat_completion.choices[0].finish_reason == "stop"

    @pytest.mark.parametrize("echo", [False, True])
    def test_openai_client_completion_echo(
        self, client: openai.OpenAI, echo: bool, backend: str, model: str, prompt: str
    ):
        if backend == "tensorrtllm":
            pytest.skip(
                reason="TRT-LLM backend currently only supports setting this parameter at model load time",
            )

        completion = client.completions.create(prompt=prompt, model=model, echo=echo)

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
    async def test_openai_client_models(self, client: openai.AsyncOpenAI, backend: str):
        async_models = await client.models.list()
        models = [model async for model in async_models]
        print(f"Models: {models}")
        if backend == "tensorrtllm":
            # ensemble or tensorrt_llm_bls
            # preprocess -> tensorrt_llm -> postprocess
            assert len(models) == 5
        elif backend == "vllm":
            assert len(models) == 1
        else:
            raise Exception(f"Unexpected backend {backend=}")

    @pytest.mark.asyncio
    async def test_openai_client_completion(
        self, client: openai.AsyncOpenAI, model: str, prompt: str
    ):
        completion = await client.completions.create(
            prompt=prompt,
            model=model,
        )

        print(f"Completion results: {completion}")
        assert completion.choices[0].text
        assert completion.choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_openai_client_chat_completion(
        self, client: openai.AsyncOpenAI, model: str, messages: List[dict]
    ):
        chat_completion = await client.chat.completions.create(
            messages=messages,
            model=model,
        )

        assert chat_completion.choices[0].message.content
        assert chat_completion.choices[0].finish_reason == "stop"
        print(f"Chat completion results: {chat_completion}")

    # TODO: Add this test
    @pytest.mark.skip(reason="Not Implemented Yet")
    @pytest.mark.asyncio
    async def test_completion_streaming(self):
        pass

    @pytest.mark.asyncio
    async def test_chat_streaming(
        self, client: openai.AsyncOpenAI, model: str, messages: List[dict]
    ):
        # test single completion
        chat_completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
            stream=False,
        )
        output = chat_completion.choices[0].message.content
        stop_reason = chat_completion.choices[0].finish_reason

        # test streaming
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
            stream=True,
        )
        chunks = []
        finish_reason_count = 0
        async for chunk in stream:
            delta = chunk.choices[0].delta
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
