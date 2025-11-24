# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import List

import openai
import pytest


@pytest.mark.openai
class TestOpenAIClient:
    @pytest.fixture(scope="class")
    def client(self, server):
        return server.get_client()

    def test_openai_client_models(self, client: openai.OpenAI, backend: str):
        models = list(client.models.list())
        print(f"Models: {models}")
        if backend == "tensorrtllm":
            # tensorrt_llm_bls +
            # preprocess -> tensorrt_llm -> postprocess
            assert len(models) == 4
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

        usage = completion.usage
        assert usage is not None
        assert isinstance(usage.prompt_tokens, int)
        assert isinstance(usage.completion_tokens, int)
        assert isinstance(usage.total_tokens, int)
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

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

        usage = chat_completion.usage
        assert usage is not None
        assert isinstance(usage.prompt_tokens, int)
        assert isinstance(usage.completion_tokens, int)
        assert isinstance(usage.total_tokens, int)
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    @pytest.mark.parametrize("echo", [False, True])
    def test_openai_client_completion_echo(
        self, client: openai.OpenAI, echo: bool, model: str, prompt: str
    ):
        completion = client.completions.create(prompt=prompt, model=model, echo=echo)

        response = completion.choices[0].text
        if echo:
            assert response.startswith(prompt)
        else:
            # TODO: Consider using a different prompt. In TRT-LLM model, the second response may contain the prompt in the middle of the response even if echo is False, e.g. " Briefly explained.\nWhat is machine learning? She learns from data\nmachine learning".
            assert prompt not in response

    @pytest.mark.skip(reason="Not Implemented Yet")
    def test_openai_client_function_calling(self):
        pass


@pytest.mark.openai
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
            # tensorrt_llm_bls +
            # preprocess -> tensorrt_llm -> postprocess
            assert len(models) == 4
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

        usage = completion.usage
        assert usage is not None
        assert isinstance(usage.prompt_tokens, int)
        assert isinstance(usage.completion_tokens, int)
        assert isinstance(usage.total_tokens, int)
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

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

        usage = chat_completion.usage
        assert usage is not None
        assert isinstance(usage.prompt_tokens, int)
        assert isinstance(usage.completion_tokens, int)
        assert isinstance(usage.total_tokens, int)
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

        print(f"Chat completion results: {chat_completion}")

    @pytest.mark.asyncio
    async def test_completion_streaming(
        self, client: openai.AsyncOpenAI, model: str, prompt: str
    ):
        # Test single completion for comparison
        chat_completion = await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=10,
            temperature=0.0,
            stream=False,
            seed=0,
        )
        output = chat_completion.choices[0].text
        stop_reason = chat_completion.choices[0].finish_reason

        # Test streaming
        stream = await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=10,
            temperature=0.0,
            stream=True,
            seed=0,
        )
        chunks = []
        finish_reason_count = 0
        async for chunk in stream:
            delta = chunk.choices[0]
            if delta.text:
                chunks.append(delta.text)
            if delta.finish_reason is not None:
                finish_reason_count += 1

        # finish reason should only return in last block
        assert finish_reason_count == 1
        assert chunk.choices[0].finish_reason == stop_reason
        assert "".join(chunks) == output

    @pytest.mark.parametrize(
        "sampling_parameter_dict",
        [
            {},
            # Verify that stop words work with streaming outputs
            {"stop": "is"},
            {"stop": ["is"]},
            {"stop": ["is", ".", ","]},
        ],
    )
    @pytest.mark.asyncio
    async def test_chat_streaming(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        messages: List[dict],
        sampling_parameter_dict: dict,
    ):
        # Fixed seed and temperature for comparing reproducible responses
        seed = 0
        temperature = 0.0
        # Generate enough tokens to easily identify stop words are working.
        max_completion_tokens = 64

        # Test single chat completion for comparison
        chat_completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            seed=seed,
            stream=False,
            **sampling_parameter_dict,
        )
        output = chat_completion.choices[0].message.content
        stop_reason = chat_completion.choices[0].finish_reason

        # Test streaming
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            seed=seed,
            stream=True,
            **sampling_parameter_dict,
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
            assert chunk.usage is None

        # finish reason should only return in last block
        assert finish_reason_count == 1
        assert chunk.choices[0].finish_reason == stop_reason

        # Assert that streaming actually returned multiple responses
        # and that it is equivalent to the non-streamed output
        assert len(chunks) > 1
        streamed_output = "".join(chunks)
        assert streamed_output == output

    @pytest.mark.asyncio
    async def test_chat_streaming_usage_option(
        self, client: openai.AsyncOpenAI, model: str, messages: List[dict]
    ):
        seed = 0
        temperature = 0.0
        max_tokens = 16

        # Get usage and content from a non-streaming call
        stream_false = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            stream=False,
        )
        usage_stream_false = stream_false.usage
        stream_false_output = stream_false.choices[0].message.content
        assert usage_stream_false is not None
        assert stream_false_output is not None

        # First, run with include_usage=False.
        stream_options_false = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            stream=True,
            stream_options={"include_usage": False},
        )
        chunks_false = [chunk async for chunk in stream_options_false]
        for chunk in chunks_false:
            assert chunk.usage is None, "Usage should be null when include_usage=False"
        stream_options_false_output = "".join(
            c.choices[0].delta.content
            for c in chunks_false
            if c.choices and c.choices[0].delta.content
        )

        # Now, run with include_usage=True.
        stream_options_true = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks_true = [chunk async for chunk in stream_options_true]
        content_chunks = [c for c in chunks_true if c.usage is None]

        # Verify that we received exactly one extra chunk.
        assert len(chunks_true) == len(chunks_false) + 1

        # Verify content is consistent
        stream_options_true_output = "".join(
            c.choices[0].delta.content
            for c in content_chunks
            if c.choices and c.choices[0].delta.content
        )
        assert stream_options_true_output == stream_false_output
        assert stream_options_true_output == stream_options_false_output

        # Verify the final chunk has usage data and empty choices.
        final_chunk = chunks_true[-1]
        assert final_chunk.usage is not None
        assert len(final_chunk.choices) == 0
        usage_stream_options_true = final_chunk.usage
        assert (
            isinstance(usage_stream_options_true.prompt_tokens, int)
            and usage_stream_options_true.prompt_tokens > 0
        )
        assert (
            isinstance(usage_stream_options_true.completion_tokens, int)
            and usage_stream_options_true.completion_tokens > 0
        )
        assert (
            usage_stream_options_true.total_tokens
            == usage_stream_options_true.prompt_tokens
            + usage_stream_options_true.completion_tokens
        )

        # Verify other chunks have no usage data.
        for chunk in chunks_true[:-1]:
            assert chunk.usage is None

        # Assert usage is consistent between streaming and non-streaming calls
        assert usage_stream_false.model_dump() == usage_stream_options_true.model_dump()

    @pytest.mark.asyncio
    async def test_completion_streaming_usage_option(
        self, client: openai.AsyncOpenAI, model: str, prompt: str
    ):
        seed = 0
        temperature = 0.0
        max_tokens = 16

        # Get usage and content from a non-streaming call
        stream_false = await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            seed=seed,
        )
        usage_stream_false = stream_false.usage
        stream_false_output = stream_false.choices[0].text
        assert usage_stream_false is not None
        assert stream_false_output is not None

        # First, run with include_usage=False.
        stream_options_false = await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            stream=True,
            stream_options={"include_usage": False},
        )
        chunks_false = [chunk async for chunk in stream_options_false]
        for chunk in chunks_false:
            assert chunk.usage is None
        stream_options_false_output = "".join(
            c.choices[0].text for c in chunks_false if c.choices and c.choices[0].text
        )

        # Now, run with include_usage=True.
        stream_options_true = await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            seed=seed,
            stream_options={"include_usage": True},
        )
        chunks_true = [chunk async for chunk in stream_options_true]
        content_chunks = [c for c in chunks_true if c.usage is None]

        # Verify that we received exactly one extra chunk.
        assert len(chunks_true) == len(chunks_false) + 1

        # Verify content is consistent
        stream_options_true_output = "".join(
            c.choices[0].text for c in content_chunks if c.choices and c.choices[0].text
        )
        assert stream_options_true_output == stream_false_output
        assert stream_options_true_output == stream_options_false_output

        # Verify the final chunk has usage data and empty choices.
        final_chunk = chunks_true[-1]
        assert final_chunk.usage is not None
        assert len(final_chunk.choices) == 0
        usage_stream_options_true = final_chunk.usage
        assert (
            isinstance(usage_stream_options_true.prompt_tokens, int)
            and usage_stream_options_true.prompt_tokens > 0
        )
        assert (
            isinstance(usage_stream_options_true.completion_tokens, int)
            and usage_stream_options_true.completion_tokens > 0
        )
        assert (
            usage_stream_options_true.total_tokens
            == usage_stream_options_true.prompt_tokens
            + usage_stream_options_true.completion_tokens
        )

        # Verify other chunks have no usage data.
        for chunk in chunks_true[:-1]:
            assert chunk.usage is None

        # Assert usage is consistent between streaming and non-streaming calls
        assert usage_stream_false.model_dump() == usage_stream_options_true.model_dump()

    @pytest.mark.asyncio
    async def test_stream_options_without_streaming(
        self, client: openai.AsyncOpenAI, model: str, prompt: str
    ):
        with pytest.raises(openai.BadRequestError) as e:
            await client.completions.create(
                model=model,
                prompt=prompt,
                stream=False,
                stream_options={"include_usage": True},
            )
        assert "`stream_options` can only be used when `stream` is True" in str(e.value)

        with pytest.raises(openai.BadRequestError) as e:
            await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                stream_options={"include_usage": True},
            )
        assert "`stream_options` can only be used when `stream` is True" in str(e.value)

    @pytest.mark.asyncio
    async def test_chat_completion_logprobs(
        self, client: openai.AsyncOpenAI, backend: str, model: str, messages: List[dict]
    ):
        """Test logprobs for chat completions."""
        # Non-vLLM backends should raise an error
        if backend != "vllm":
            with pytest.raises(openai.BadRequestError) as exc_info:
                await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    logprobs=True,
                    top_logprobs=2,
                    max_tokens=10,
                )
            assert "logprobs are currently available only for the vLLM backend" in str(
                exc_info.value
            )
            return

        chat_completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            logprobs=True,
            top_logprobs=2,
            max_tokens=10,
        )

        assert chat_completion.choices[0].message.content
        assert chat_completion.choices[0].logprobs is not None

        logprobs = chat_completion.choices[0].logprobs
        assert logprobs.content is not None
        assert len(logprobs.content) > 0

        # Validate each token logprob
        for token_logprob in logprobs.content:
            assert token_logprob.token
            assert isinstance(token_logprob.logprob, float)
            assert isinstance(token_logprob.bytes, list)
            assert token_logprob.top_logprobs is not None
            assert len(token_logprob.top_logprobs) > 0

    @pytest.mark.asyncio
    async def test_completion_logprobs(
        self, client: openai.AsyncOpenAI, backend: str, model: str, prompt: str
    ):
        """Test logprobs for completions."""
        # Non-vLLM backends should raise an error
        if backend != "vllm":
            with pytest.raises(openai.BadRequestError) as exc_info:
                await client.completions.create(
                    model=model,
                    prompt=prompt,
                    logprobs=3,
                    max_tokens=10,
                )
            assert "logprobs are currently available only for the vLLM backend" in str(
                exc_info.value
            )
            return

        completion = await client.completions.create(
            model=model,
            prompt=prompt,
            logprobs=3,
            max_tokens=10,
        )

        assert completion.choices[0].text
        assert completion.choices[0].logprobs is not None

        logprobs = completion.choices[0].logprobs
        assert logprobs.tokens is not None
        assert logprobs.token_logprobs is not None
        assert logprobs.text_offset is not None
        assert logprobs.top_logprobs is not None

        num_tokens = len(logprobs.tokens)
        assert len(logprobs.token_logprobs) == num_tokens
        assert len(logprobs.text_offset) == num_tokens
        assert len(logprobs.top_logprobs) == num_tokens

    @pytest.mark.asyncio
    async def test_chat_streaming_with_logprobs(
        self, client: openai.AsyncOpenAI, backend: str, model: str, messages: List[dict]
    ):
        """Test streaming chat completions with logprobs."""
        # Non-vLLM backends should raise an error
        if backend != "vllm":
            with pytest.raises(openai.BadRequestError) as exc_info:
                stream = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=10,
                    stream=True,
                    logprobs=True,
                    top_logprobs=2,
                )
                # Try to consume the stream (error should happen before streaming starts)
                async for _ in stream:
                    pass
            assert "logprobs are currently available only for the vLLM backend" in str(
                exc_info.value
            )
            return

        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=10,
            stream=True,
            logprobs=True,
            top_logprobs=2,
        )

        chunks_with_logprobs = 0
        async for chunk in stream:
            if chunk.choices[0].logprobs is not None:
                chunks_with_logprobs += 1
                logprobs = chunk.choices[0].logprobs
                assert logprobs.content is not None
                assert len(logprobs.content) > 0

        # At least some chunks should have logprobs
        assert chunks_with_logprobs > 0

    @pytest.mark.asyncio
    async def test_completion_streaming_with_logprobs(
        self, client: openai.AsyncOpenAI, backend: str, model: str, prompt: str
    ):
        """Test streaming completions with logprobs."""
        # Non-vLLM backends should raise an error
        if backend != "vllm":
            with pytest.raises(openai.BadRequestError) as exc_info:
                stream = await client.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=10,
                    stream=True,
                    logprobs=3,
                )
                # Try to consume the stream (error should happen before streaming starts)
                async for _ in stream:
                    pass
            assert "logprobs are currently available only for the vLLM backend" in str(
                exc_info.value
            )
            return

        stream = await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=10,
            stream=True,
            logprobs=3,
        )

        chunks_with_logprobs = 0
        async for chunk in stream:
            if chunk.choices[0].logprobs is not None:
                chunks_with_logprobs += 1
                logprobs = chunk.choices[0].logprobs
                assert logprobs.tokens is not None
                assert logprobs.token_logprobs is not None
                assert logprobs.text_offset is not None
                assert logprobs.top_logprobs is not None

        # At least some chunks should have logprobs
        assert chunks_with_logprobs > 0

    @pytest.mark.asyncio
    async def test_top_logprobs_requires_logprobs(
        self, client: openai.AsyncOpenAI, model: str, messages: List[dict]
    ):
        """
        Test that top_logprobs without logprobs raises an error
        """
        with pytest.raises(openai.BadRequestError) as exc_info:
            await client.chat.completions.create(
                model=model,
                messages=messages,
                top_logprobs=2,  # Without logprobs=True
                max_tokens=5,
            )
        assert "`top_logprobs` can only be used when `logprobs` is True" in str(
            exc_info.value
        )

    @pytest.mark.asyncio
    async def test_chat_top_logprobs_exceeds_max(
        self, client: openai.AsyncOpenAI, model: str, messages: List[dict]
    ):
        """Test that top_logprobs > 20 raises schema validation error."""
        with pytest.raises(openai.BadRequestError) as exc_info:
            await client.chat.completions.create(
                model=model,
                messages=messages,
                logprobs=True,
                top_logprobs=25,  # Exceeds maximum of 20
                max_tokens=5,
            )
        # Pydantic validation error
        assert "less than or equal to 20" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_completion_logprobs_exceeds_max(
        self, client: openai.AsyncOpenAI, model: str, prompt: str
    ):
        """Test that logprobs > 5 raises schema validation error."""
        with pytest.raises(openai.BadRequestError) as exc_info:
            await client.completions.create(
                model=model,
                prompt=prompt,
                logprobs=7,  # Exceeds maximum of 5
                max_tokens=5,
            )
        # Pydantic validation error
        assert "less than or equal to 5" in str(exc_info.value).lower()
