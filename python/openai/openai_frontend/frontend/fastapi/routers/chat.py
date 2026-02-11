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

import traceback

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from schemas.openai import CreateChatCompletionRequest, CreateChatCompletionResponse
from utils.utils import ClientError, ServerError, StatusCode

router = APIRouter()


@router.post(
    "/v1/chat/completions", response_model=CreateChatCompletionResponse, tags=["Chat"]
)
async def create_chat_completion(
    request: CreateChatCompletionRequest,
    raw_request: Request,
) -> CreateChatCompletionResponse | StreamingResponse:
    """
    Creates a chat completion for the provided messages and parameters.
    """
    if not raw_request.app.engine:
        raise HTTPException(
            status_code=StatusCode.SERVER_ERROR, detail="No attached inference engine"
        )

    try:
        response = await raw_request.app.engine.chat(request)
        if request.stream:
            return StreamingResponse(response, media_type="text/event-stream")
        return response
    except ClientError as e:
        raise HTTPException(status_code=StatusCode.CLIENT_ERROR, detail=f"{e}")
    except ServerError as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=StatusCode.SERVER_ERROR, detail=f"{e}")
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=StatusCode.SERVER_ERROR, detail=f"{e}")
