# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterable, AsyncIterator, Callable, Dict, List, Optional

import tritonserver
from engine.engine import LLMEngine
from engine.utils.tokenizer import get_tokenizer
from engine.utils.triton import (
    _create_trtllm_inference_request,
    _create_vllm_inference_request,
    _get_output,
    _validate_triton_responses_non_streaming,
)
from schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionFinishReason,
    ChatCompletionResponseMessage,
    ChatCompletionStreamingResponseChoice,
    ChatCompletionStreamResponseDelta,
    Choice,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    CreateCompletionRequest,
    CreateCompletionResponse,
    FinishReason,
    Model,
    ObjectType,
)


# TODO: Improve type hints
@dataclass
class TritonModelMetadata:
    # Name used in Triton model repository
    name: str
    # Name of backend used by Triton
    backend: str
    # Triton model object handle
    model: tritonserver.Model
    # Tokenizers used for chat templates
    tokenizer: Optional[Any]
    # Time that model was loaded by Triton
    create_time: int
    # Conversion format between OpenAI and Triton requests
    request_converter: Callable


class TritonLLMEngine(LLMEngine):
    def __init__(
        self, server: tritonserver.Server, tokenizer: str, backend: Optional[str] = None
    ):
        # Assume an already configured and started server
        self.server = server
        self.tokenizer = self._get_tokenizer(tokenizer)
        # TODO: Reconsider name of "backend" vs. something like "request_format"
        self.backend = backend

        # NOTE: Creation time and model metadata will be static at startup for
        # now, and won't account for dynamically loading/unloading models.
        self.create_time = int(time.time())
        self.model_metadata = self._get_model_metadata()

    def ready(self) -> bool:
        return self.server.ready()

    def metrics(self) -> str:
        return self.server.metrics()

    def models(self) -> List[Model]:
        models = []
        for metadata in self.model_metadata.values():
            models.append(
                Model(
                    id=metadata.name,
                    created=metadata.create_time,
                    object=ObjectType.model,
                    owned_by="Triton Inference Server",
                ),
            )

        return models

    async def chat(
        self, request: CreateChatCompletionRequest
    ) -> CreateChatCompletionResponse | AsyncIterator[str]:
        metadata = self.model_metadata.get(request.model)
        self._validate_chat_request(request, metadata)

        conversation = [
            message.model_dump(exclude_none=True) for message in request.messages
        ]
        add_generation_prompt = True

        prompt = metadata.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

        # Convert to Triton request format and perform inference
        responses = metadata.model.async_infer(
            metadata.request_converter(metadata.model, prompt, request)
        )

        # Prepare and send responses back to client in OpenAI format
        request_id = f"cmpl-{uuid.uuid1()}"
        created = int(time.time())
        default_role = "assistant"
        role = self._get_first_response_role(
            conversation, add_generation_prompt, default_role
        )

        if request.stream:
            return self._streaming_chat_iterator(
                request_id, created, request.model, role, responses
            )

        # Response validation with decoupled models in mind
        responses = [response async for response in responses]
        _validate_triton_responses_non_streaming(responses)
        response = responses[0]
        text = _get_output(response)

        return CreateChatCompletionResponse(
            id=request_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionResponseMessage(
                        content=text, role=role, function_call=None
                    ),
                    logprobs=None,
                    finish_reason=ChatCompletionFinishReason.stop,
                )
            ],
            created=created,
            model=request.model,
            system_fingerprint=None,
            object=ObjectType.chat_completion,
        )

    async def completion(
        self, request: CreateCompletionRequest
    ) -> CreateCompletionResponse | AsyncIterator[str]:
        # Validate request and convert to Triton format
        metadata = self.model_metadata.get(request.model)
        self._validate_completion_request(request, metadata)

        # Convert to Triton request format and perform inference
        responses = metadata.model.async_infer(
            metadata.request_converter(metadata.model, request.prompt, request)
        )

        # Prepare and send responses back to client in OpenAI format
        request_id = f"cmpl-{uuid.uuid1()}"
        created = int(time.time())
        if request.stream:
            return self._streaming_completion_iterator(
                request_id, created, metadata.name, responses
            )

        # Response validation with decoupled models in mind
        responses = [response async for response in responses]
        _validate_triton_responses_non_streaming(responses)
        response = responses[0]
        text = _get_output(response)

        choice = Choice(
            finish_reason=FinishReason.stop,
            index=0,
            logprobs=None,
            text=text,
        )
        return CreateCompletionResponse(
            id=request_id,
            choices=[choice],
            system_fingerprint=None,
            object=ObjectType.text_completion,
            created=created,
            model=metadata.name,
        )

    # TODO: This behavior should be tested further
    def _get_first_response_role(
        self, conversation: List[Dict], add_generation_prompt: bool, default_role: str
    ) -> str:
        if add_generation_prompt:
            return default_role

        return conversation[-1]["role"]

    # TODO: Expose explicit flag to catch edge cases
    def _determine_request_converter(self, backend: str):
        # Allow manual override of backend request format if provided by user
        if self.backend:
            backend = self.backend

        # Request conversion from OpenAI format to backend-specific format
        if backend == "vllm":
            return _create_vllm_inference_request

        # Use TRT-LLM format as default for everything else. This could be
        # an ensemble, a python or BLS model, a TRT-LLM backend model, etc.
        return _create_trtllm_inference_request

    def _get_tokenizer(self, tokenizer_name: str):
        tokenizer = None
        if tokenizer_name:
            tokenizer = get_tokenizer(tokenizer_name)

        return tokenizer

    def _get_model_metadata(self) -> Dict[str, TritonModelMetadata]:
        # One tokenizer and creation time shared for all loaded models for now.
        model_metadata = {}

        # Read all triton models and store the necessary metadata for each
        for name, _ in self.server.models().keys():
            model = self.server.model(name)
            backend = model.config()["backend"]
            # Explicitly handle ensembles to avoid any runtime validation errors
            if not backend and model.config()["platform"] == "ensemble":
                backend = "ensemble"
            print(f"Found model: {name=}, {backend=}")

            metadata = TritonModelMetadata(
                name=name,
                backend=backend,
                model=model,
                tokenizer=self.tokenizer,
                create_time=self.create_time,
                request_converter=self._determine_request_converter(backend),
            )
            model_metadata[name] = metadata

        return model_metadata

    def _get_streaming_chat_response_chunk(
        self,
        choice: ChatCompletionStreamingResponseChoice,
        request_id: str,
        created: int,
        model: str,
    ) -> CreateChatCompletionStreamResponse:
        return CreateChatCompletionStreamResponse(
            id=request_id,
            choices=[choice],
            created=created,
            model=model,
            system_fingerprint=None,
            object=ObjectType.chat_completion_chunk,
        )

    def _get_first_streaming_chat_response(
        self, request_id: str, created: int, model: str, role: str
    ) -> CreateChatCompletionStreamResponse:
        # First chunk has no content and sets the role
        choice = ChatCompletionStreamingResponseChoice(
            index=0,
            delta=ChatCompletionStreamResponseDelta(
                role=role, content="", function_call=None
            ),
            logprobs=None,
            finish_reason=None,
        )
        chunk = self._get_streaming_chat_response_chunk(
            choice, request_id, created, model
        )
        return chunk

    def _get_nth_streaming_chat_response(
        self,
        request_id: str,
        created: int,
        model: str,
        response: tritonserver.InferenceResponse,
    ) -> CreateChatCompletionStreamResponse:
        text = _get_output(response)
        choice = ChatCompletionStreamingResponseChoice(
            index=0,
            delta=ChatCompletionStreamResponseDelta(
                role=None, content=text, function_call=None
            ),
            logprobs=None,
            finish_reason=ChatCompletionFinishReason.stop if response.final else None,
        )

        chunk = self._get_streaming_chat_response_chunk(
            choice, request_id, created, model
        )
        return chunk

    async def _streaming_chat_iterator(
        self,
        request_id: str,
        created: int,
        model: str,
        role: str,
        responses: AsyncIterable,
    ) -> AsyncIterator[str]:
        chunk = self._get_first_streaming_chat_response(
            request_id, created, model, role
        )
        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

        async for response in responses:
            chunk = self._get_nth_streaming_chat_response(
                request_id, created, model, response
            )
            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

        yield "data: [DONE]\n\n"

    def _validate_chat_request(
        self, request: CreateChatCompletionRequest, metadata: TritonModelMetadata
    ):
        """
        Validates a chat request to align with currently supported features.
        """

        # Reject missing internal information needed to do inference
        if not metadata:
            raise Exception(f"Unknown model: {request.model}")

        if not metadata.tokenizer:
            raise Exception("Unknown tokenizer")

        if not metadata.backend:
            raise Exception("Unknown backend")

        if not metadata.request_converter:
            raise Exception(f"Unknown request format for model: {request.model}")

        # Reject unsupported features if requested
        if request.n and request.n > 1:
            raise Exception(
                f"Received n={request.n}, but only single choice (n=1) is currently supported"
            )

        if request.logit_bias is not None or request.logprobs:
            raise Exception("logit bias and log probs not currently supported")

    async def _streaming_completion_iterator(
        self, request_id: str, created: int, model: str, responses: AsyncIterable
    ) -> AsyncIterator[str]:
        async for response in responses:
            text = _get_output(response)
            choice = Choice(
                finish_reason=FinishReason.stop if response.final else None,
                index=0,
                logprobs=None,
                text=text,
            )
            chunk = CreateCompletionResponse(
                id=request_id,
                choices=[choice],
                system_fingerprint=None,
                object=ObjectType.text_completion,
                created=created,
                model=model,
            )

            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

        yield "data: [DONE]\n\n"

    def _validate_completion_request(
        self, request: CreateCompletionRequest, metadata: TritonModelMetadata
    ):
        """
        Validates a completions request to align with currently supported features.
        """
        # Reject missing internal information needed to do inference
        if not metadata:
            raise Exception(f"Unknown model: {request.model}")

        if not metadata.backend:
            raise Exception("Unknown backend")

        if not metadata.request_converter:
            raise Exception(f"Unknown request format for model: {request.model}")

        # Reject unsupported features if requested
        if request.suffix is not None:
            raise Exception("suffix is not currently supported")

        if not request.prompt:
            raise Exception("prompt must be non-empty")

        # Currently only support single string as input
        if not isinstance(request.prompt, str):
            raise Exception("only single string input is supported")

        if request.n and request.n > 1:
            raise Exception(
                f"Received n={request.n}, but only single choice (n=1) is currently supported"
            )

        if request.best_of and request.best_of > 1:
            raise Exception(
                f"Received best_of={request.best_of}, but only single choice (best_of=1) is currently supported"
            )

        if request.logit_bias is not None or request.logprobs is not None:
            raise Exception("logit bias and log probs not supported")
