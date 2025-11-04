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


from __future__ import annotations

from typing import Iterator, List, Protocol

from schemas.openai import (
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateCompletionRequest,
    CreateCompletionResponse,
    CreateEmbeddingRequest,
    CreateEmbeddingResponse,
    Model,
)


class LLMEngine(Protocol):
    """
    Interface for an OpenAI-aware inference engine to be attached to an
    OpenAI-compatible frontend.

    NOTE: This interface is subject to change, and may land on something more
          generic rather than the current 1:1 with OpenAI endpoints over time.
    """

    def ready(self) -> bool:
        """
        Returns True if the engine is ready to accept inference requests, or False otherwise.
        """
        pass

    def metrics(self) -> str:
        """
        Returns the engine's metrics in a Prometheus-compatible string format.
        """
        pass

    def models(self) -> List[Model]:
        """
        Returns a List of OpenAI Model objects.
        """
        pass

    def chat(
        self, request: CreateChatCompletionRequest
    ) -> CreateChatCompletionResponse | Iterator[str]:
        """
        If request.stream is True, this returns an Iterator (or Generator) that
        produces server-sent-event (SSE) strings in the following form:
            'data: {CreateChatCompletionStreamResponse}\n\n'
            ...
            'data: [DONE]\n\n'

        If request.stream is False, this returns a CreateChatCompletionResponse.
        """
        pass

    def completion(
        self, request: CreateCompletionRequest
    ) -> CreateCompletionResponse | Iterator[str]:
        """
        If request.stream is True, this returns an Iterator (or Generator) that
        produces server-sent-event (SSE) strings in the following form:
            'data: {CreateCompletionResponse}\n\n'
            ...
            'data: [DONE]\n\n'

        If request.stream is False, this returns a CreateCompletionResponse.
        """
        pass

    def embedding(self, request: CreateEmbeddingRequest) -> CreateEmbeddingResponse:
        """
        Returns a CreateEmbeddingResponse.
        """
        pass
