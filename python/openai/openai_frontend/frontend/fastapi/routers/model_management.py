# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from schemas.openai import Model
from utils.utils import ClientError, ServerError, StatusCode

router = APIRouter()


@router.post(
    "/v1/models/{model_name}/load",
    response_model=Model,
    tags=["Model Management"],
)
async def load_model(model_name: str, raw_request: Request) -> Model:
    """
    Loads a model by name. Only available in EXPLICIT model control mode.
    Blocks until the model is fully loaded and ready.
    """
    if not raw_request.app.engine:
        raise HTTPException(
            status_code=StatusCode.SERVER_ERROR,
            detail="No attached inference engine",
        )

    try:
        return await raw_request.app.engine.load_model(model_name)
    except ClientError as e:
        raise HTTPException(status_code=StatusCode.CLIENT_ERROR, detail=f"{e}")
    except ServerError as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=StatusCode.SERVER_ERROR, detail=f"{e}")
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=StatusCode.SERVER_ERROR, detail=f"{e}")


@router.post(
    "/v1/models/{model_name}/unload",
    tags=["Model Management"],
)
async def unload_model(model_name: str, raw_request: Request) -> dict:
    """
    Unloads a model by name. Only available in EXPLICIT model control mode.
    Blocks until the model is fully unloaded. In-flight requests are allowed
    to complete before the model is removed.
    """
    if not raw_request.app.engine:
        raise HTTPException(
            status_code=StatusCode.SERVER_ERROR,
            detail="No attached inference engine",
        )

    try:
        await raw_request.app.engine.unload_model(model_name)
        return {"status": "success", "model": model_name}
    except ClientError as e:
        raise HTTPException(status_code=StatusCode.CLIENT_ERROR, detail=f"{e}")
    except ServerError as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=StatusCode.SERVER_ERROR, detail=f"{e}")
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=StatusCode.SERVER_ERROR, detail=f"{e}")
