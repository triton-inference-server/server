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
from typing import Dict, List

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

router = APIRouter()


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
    try:
        response = raw_request.app.engine.chat(request)
        if request.stream:
            return StreamingResponse(response, media_type="text/event-stream")
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Chat Completion failed: {e}")


def placeholder():
    metadata = raw_request.app.models.get(request.model)
    _validate_chat_request(request, metadata)

    # TODO: Move conversation/role bits into helper

    # Prepare prompt with chat template
    # TODO: Does this need to be exposed to the user?
    add_generation_prompt = True
    conversation = [
        {"role": str(message.role), "content": str(message.content)}
        for message in request.messages
    ]

    prompt = metadata.tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

    # Convert to Triton request format and perform inference
    triton_model = metadata.model
    responses = triton_model.infer(
        metadata.request_converter(triton_model, prompt, request)
    )

    # Prepare and send responses back to client in OpenAI format
    request_id = f"cmpl-{uuid.uuid1()}"
    created = int(time.time())
    default_role = "assistant"
    role = _get_first_response_role(conversation, add_generation_prompt, default_role)

    if request.stream:
        return StreamingResponse(
            _streaming_chat_completion_response(
                request_id, created, request.model, role, responses
            ),
            media_type="text/event-stream",
        )

    # Response validation with decoupled models in mind
    responses = list(responses)
    validate_triton_responses_non_streaming(responses)
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
