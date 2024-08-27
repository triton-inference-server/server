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

import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import tritonserver
from engine.engine import OpenAIEngine
from schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionFinishReason,
    ChatCompletionResponseMessage,
    ChatCompletionStreamingResponseChoice,
    ChatCompletionStreamResponseDelta,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    Model,
    ObjectType,
)
from utils.tokenizer import get_tokenizer
from utils.triton import create_trtllm_inference_request, create_vllm_inference_request


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


class TritonOpenAIEngine(OpenAIEngine):
    def __init__(self, server: tritonserver.Server):
        # Assume an already configured and started server
        self.server = server

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

    def chat(
        self, request: CreateChatCompletionRequest
    ) -> CreateChatCompletionResponse | Iterator[str]:
        metadata = self.model_metadata.get(request.model)
        self._validate_chat_request(request, metadata)

        conversation = [
            {"role": str(message.role), "content": str(message.content)}
            for message in request.messages
        ]
        add_generation_prompt = True

        prompt = metadata.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

        # Convert to Triton request format and perform inference
        responses = metadata.model.infer(
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
        responses = list(responses)
        self._validate_triton_responses_non_streaming(responses)
        response = responses[0]
        text = self._get_output(response)

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

    # TODO: This behavior should be tested further
    def _get_first_response_role(
        self, conversation: List[Dict], add_generation_prompt: bool, default_role: str
    ) -> str:
        if add_generation_prompt:
            return default_role

        return conversation[-1]["role"]

    # TODO: Expose explicit flag to catch edge cases
    def _determine_request_converter(self, backend):
        # Request conversion from OpenAI format to backend-specific format
        if backend == "vllm":
            return create_vllm_inference_request

        # Use TRT-LLM format as default for everything else. This could be
        # an ensemble, a python or BLS model, a TRT-LLM backend model, etc.
        return create_trtllm_inference_request

    def _get_output(self, response):
        if "text_output" in response.outputs:
            try:
                return response.outputs["text_output"].to_string_array()[0]
            except Exception:
                return str(response.outputs["text_output"].to_bytes_array()[0])
        return ""

    def _validate_triton_responses_non_streaming(self, responses):
        num_responses = len(responses)
        if num_responses == 1 and responses[0].final != True:
            raise Exception("Unexpected internal error with incorrect response flags")
        if num_responses == 2 and responses[-1].final != True:
            raise Exception("Unexpected internal error with incorrect response flags")
        if num_responses > 2:
            raise Exception(
                f"Unexpected number of responses: {num_responses}, expected 1."
            )

    def _get_tokenizer(self):
        # TODO: Consider support for custom tokenizers
        tokenizer = None
        tokenizer_name = os.environ.get("TOKENIZER")
        if tokenizer_name:
            print(
                f"Using env var TOKENIZER={tokenizer_name} to determine the tokenizer"
            )
            tokenizer = get_tokenizer(tokenizer_name)

        return tokenizer

    def _get_model_metadata(self) -> Dict[str, TritonModelMetadata]:
        tokenizer = self._get_tokenizer()

        # One tokenizer, convert function, and creation time for all loaded models.
        # NOTE: This doesn't currently support having both a vLLM and TRT-LLM
        # model loaded at the same time.
        model_metadata = {}

        # Read all triton models and gather the respective backends of each
        for name, _ in self.server.models().keys():
            model = self.server.model(name)
            backend = model.config()["backend"]
            print(f"Found model: {name=}, {backend=}")

            metadata = TritonModelMetadata(
                name=name,
                backend=backend,
                model=model,
                tokenizer=tokenizer,
                create_time=self.create_time,
                request_converter=self._determine_request_converter(backend),
            )
            model_metadata[name] = metadata

        return model_metadata

    def _get_streaming_response_chunk(
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

    def _get_first_streaming_response(
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
        chunk = self._get_streaming_response_chunk(choice, request_id, created, model)
        return chunk

    def _get_nth_streaming_response(
        self,
        request_id: str,
        created: int,
        model: str,
        response: tritonserver.InferenceResponse,
    ) -> CreateChatCompletionStreamResponse:
        text = self._get_output(response)
        choice = ChatCompletionStreamingResponseChoice(
            index=0,
            delta=ChatCompletionStreamResponseDelta(
                role=None, content=text, function_call=None
            ),
            logprobs=None,
            finish_reason=ChatCompletionFinishReason.stop if response.final else None,
        )

        chunk = self._get_streaming_response_chunk(choice, request_id, created, model)
        return chunk

    def _streaming_chat_iterator(
        self, request_id: str, created: int, model: str, role: str, responses: List
    ) -> Iterator[str]:
        chunk = self._get_first_streaming_response(request_id, created, model, role)
        yield f"data: {chunk.json(exclude_unset=True)}\n\n"

        for response in responses:
            chunk = self._get_nth_streaming_response(
                request_id, created, model, response
            )
            yield f"data: {chunk.json(exclude_unset=True)}\n\n"

        yield "data: [DONE]\n\n"

    def _validate_chat_request(
        self, request: CreateChatCompletionRequest, metadata: TritonModelMetadata
    ):
        """
        Validates a chat request to align with currently supported features.
        """

        # Missing internal information needed to do inference
        if not metadata:
            raise Exception(f"Unknown model: {request.model}")

        if not metadata.tokenizer:
            raise Exception("Unknown tokenizer")

        if not metadata.backend:
            raise Exception("Unknown backend")

        if not metadata.request_converter:
            raise Exception(f"Unknown request format for model: {request.model}")

        # Currently unsupported features being requested
        if request.n and request.n > 1:
            raise Exception("Only single choice is supported")

        if request.logit_bias is not None or request.logprobs:
            raise Exception("logit bias and log probs not currently supported")
