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

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionFinishReason,
    ChatCompletionResponseMessage,
    ChatCompletionStreamingResponseChoice,
    ChatCompletionStreamResponseDelta,
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    ObjectType,
)
from utils.triton import get_output, validate_triton_responses

router = APIRouter()


def get_first_response_role(conversation, add_generation_prompt, default_role):
    if add_generation_prompt:
        return default_role

    return conversation[-1]["role"]


def streaming_chat_completion_response(request_id, created, model, role, responses):
    # first chunk
    choice = ChatCompletionStreamingResponseChoice(
        index=0,
        delta=ChatCompletionStreamResponseDelta(
            role=role, content="", function_call=None
        ),
        logprobs=None,
        finish_reason=None,
    )
    chunk = CreateChatCompletionStreamResponse(
        id=request_id,
        choices=[choice],
        created=created,
        model=model,
        system_fingerprint=None,
        object=ObjectType.chat_completion_chunk,
    )
    yield f"data: {chunk.json(exclude_unset=True)}\n\n"

    for response in responses:
        text = get_output(response)

        choice = ChatCompletionStreamingResponseChoice(
            index=0,
            delta=ChatCompletionStreamResponseDelta(
                role=None, content=text, function_call=None
            ),
            logprobs=None,
            finish_reason=ChatCompletionFinishReason.stop if response.final else None,
        )

        chunk = CreateChatCompletionStreamResponse(
            id=request_id,
            choices=[choice],
            created=created,
            model=model,
            system_fingerprint=None,
            object=ObjectType.chat_completion_chunk,
        )

        yield f"data: {chunk.json(exclude_unset=True)}\n\n"

    yield "data: [DONE]\n\n"


@router.post(
    "/v1/chat/completions", response_model=CreateChatCompletionResponse, tags=["Chat"]
)
def create_chat_completion(
    request: CreateChatCompletionRequest,
    raw_request: Request,
) -> CreateChatCompletionResponse | StreamingResponse:
    """
    Creates a model response for the given chat conversation.
    """

    model_metadatas = raw_request.app.models
    if not model_metadatas:
        raise HTTPException(status_code=400, detail="No known models")

    metadata = model_metadatas.get(request.model)
    if not metadata:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")

    if not metadata.request_convert_fn:
        raise HTTPException(
            status_code=400, detail=f"Unknown request format for model: {request.model}"
        )

    if not metadata.tokenizer:
        raise HTTPException(status_code=400, detail="Unknown tokenizer")

    if not metadata.backend:
        raise HTTPException(status_code=400, detail="Unknown backend")

    # TODO: Cleanup
    triton_model = metadata.model
    if request.n and request.n > 1:
        raise HTTPException(status_code=400, detail="Only single choice is supported")

    if request.logit_bias is not None or request.logprobs:
        raise HTTPException(
            status_code=400, detail="logit bias and log probs not supported"
        )

    conversation = [
        {"role": str(message.role), "content": str(message.content)}
        for message in request.messages
    ]

    # NOTE: This behavior should be tested further
    # TODO: Do these need to be exposed to the user?
    add_generation_prompt = True
    default_role = "assistant"
    role = get_first_response_role(conversation, add_generation_prompt, default_role)

    prompt = metadata.tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

    request_id = f"cmpl-{uuid.uuid1()}"
    created = int(time.time())

    responses = triton_model.infer(
        metadata.request_convert_fn(triton_model, prompt, request)
    )

    if request.stream:
        return StreamingResponse(
            streaming_chat_completion_response(
                request_id, created, request.model, role, responses
            ),
            media_type="text/event-stream",
        )

    # Response validation with decoupled models in mind
    responses = list(responses)
    validate_triton_responses(responses)
    response = responses[0]
    text = get_output(response)

    return CreateChatCompletionResponse(
        id=request_id,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionResponseMessage(
                    content=text, role=default_role, function_call=None
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
