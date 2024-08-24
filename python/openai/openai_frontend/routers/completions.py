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

import time
import uuid
from typing import List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from schemas.openai import (
    Choice,
    CreateCompletionRequest,
    CreateCompletionResponse,
    FinishReason,
    ObjectType,
)
from utils.triton import TritonModelMetadata, get_output, validate_triton_responses

router = APIRouter()


def _streaming_completion_response(
    request_id: str, created: int, model: str, responses: List
) -> str:
    for response in responses:
        text = get_output(response)

        choice = Choice(
            finish_reason=FinishReason.stop if response.final else None,
            index=0,
            logprobs=None,
            text=text,
        )
        response = CreateCompletionResponse(
            id=request_id,
            choices=[choice],
            system_fingerprint=None,
            object=ObjectType.text_completion,
            created=created,
            model=model,
        )

        yield f"data: {response.json(exclude_unset=True)}\n\n"
    yield "data: [DONE]\n\n"


def _validate_completions_request(
    request: CreateCompletionRequest, metadata: TritonModelMetadata
):
    """
    Validates a completions request to align with currently supported features.
    """
    if not metadata:
        raise HTTPException(
            status_code=400, detail=f"Unknown model metadata for model: {request.model}"
        )

    if not metadata.request_converter:
        raise HTTPException(
            status_code=400, detail=f"Unknown request format for model: {request.model}"
        )

    if request.suffix is not None:
        raise HTTPException(status_code=400, detail="suffix is not currently supported")

    if request.model != metadata.name:
        raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}")

    if not request.prompt:
        raise HTTPException(status_code=400, detail="prompt must be non-empty")

    # Currently only support single string as input
    if not isinstance(request.prompt, str):
        raise HTTPException(
            status_code=400, detail="only single string input is supported"
        )

    if request.n and request.n > 1:
        raise HTTPException(status_code=400, detail="Only single choice is supported")

    if request.logit_bias is not None or request.logprobs is not None:
        raise HTTPException(
            status_code=400, detail="logit bias and log probs not supported"
        )


@router.post(
    "/v1/completions", response_model=CreateCompletionResponse, tags=["Completions"]
)
def create_completion(
    request: CreateCompletionRequest, raw_request: Request
) -> CreateCompletionResponse | StreamingResponse:
    """
    Creates a completion for the provided prompt and parameters.
    """

    # Validate request and convert to Triton format
    metadata = raw_request.app.models.get(request.model)
    _validate_completions_request(request, metadata)

    # Convert to Triton request format and perform inference
    triton_model = raw_request.app.server.model(request.model)
    responses = triton_model.infer(
        metadata.request_converter(triton_model, request.prompt, request)
    )

    # Prepare and send responses back to client in OpenAI format
    request_id = f"cmpl-{uuid.uuid1()}"
    created = int(time.time())
    if request.stream:
        return StreamingResponse(
            _streaming_completion_response(
                request_id, created, metadata.name, responses
            ),
            media_type="text/event-stream",
        )

    # Response validation with decoupled models in mind
    responses = list(responses)
    validate_triton_responses(responses)
    response = responses[0]
    text = get_output(response)

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
