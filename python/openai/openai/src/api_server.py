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

from contextlib import asynccontextmanager

import tritonserver
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers import chat_completions, completions, models, observability
from src.utils.triton import init_tritonserver


def add_cors_middleware(app: FastAPI):
    # Allow API calls through browser /docs route for debug purposes
    origins = [
        "http://localhost",
    ]

    # TODO: Move towards logger instead of printing
    print(f"[WARNING] Adding CORS for the following origins: {origins}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the tritonserver on FastAPI app startup
    print("Starting FastAPI app lifespan...")
    server, model_metadatas = init_tritonserver()

    # NOTE: These are meant for read-only access by routes handling requests
    # with a single process, and should generally not be modified for the
    # lifetime of the application. If multiple uvicorn workers are instantiated,
    # then multiple triton servers would be started, one per worker process.
    app.server = server
    app.models = {metadata.name: metadata for metadata in model_metadatas}

    yield

    # Cleanup the tritonserver on FastAPI app shutdown
    print("Shutting down FastAPI app lifespan...")
    if app.server:
        print("Shutting down Triton Inference Server...")
        try:
            app.server.stop()
        # Log error, but don't raise on shutdown
        except tritonserver.InternalError as e:
            print(e)


def init_app():
    app = FastAPI(
        title="OpenAI API",
        description="The OpenAI REST API. Please see https://platform.openai.com/docs/api-reference for more details.",
        version="2.0.0",
        termsOfService="https://openai.com/policies/terms-of-use",
        contact={"name": "OpenAI Support", "url": "https://help.openai.com/"},
        license={
            "name": "MIT",
            "url": "https://github.com/openai/openai-openapi/blob/master/LICENSE",
        },
        lifespan=lifespan,
    )

    app.include_router(observability.router)
    app.include_router(models.router)
    app.include_router(completions.router)
    app.include_router(chat_completions.router)

    # NOTE: For debugging purposes, should generally be restricted or removed
    add_cors_middleware(app)

    return app
