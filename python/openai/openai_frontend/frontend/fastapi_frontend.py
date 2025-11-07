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

import uvicorn
from engine.triton_engine import TritonLLMEngine
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from frontend.fastapi.middleware.api_restriction import (
    APIRestrictionMiddleware,
    RestrictedFeatures,
)
from frontend.fastapi.routers import (
    chat,
    completions,
    embeddings,
    models,
    observability,
)
from frontend.frontend import OpenAIFrontend


class FastApiFrontend(OpenAIFrontend):
    def __init__(
        self,
        engine: TritonLLMEngine,
        host: str = "localhost",
        port: int = 8000,
        log_level: str = "info",
        restricted_apis: list = None,
    ):
        self.host: str = host
        self.port: int = port
        self.log_level: str = log_level
        if restricted_apis:
            self.restricted_apis: RestrictedFeatures = RestrictedFeatures(
                restricted_apis
            )
        else:
            self.restricted_apis: RestrictedFeatures = None
        self.stopped: bool = False

        self.app = self._create_app()
        # Attach the inference engine to the FastAPI app
        self.app.engine = engine

    def __del__(self):
        self.stop()

    def start(self):
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level=self.log_level,
            timeout_keep_alive=5,
        )
        server = uvicorn.Server(config)
        server.run()

    def stop(self):
        # NOTE: If the frontend owned the engine, it could do cleanup here.
        pass

    def _create_app(self):
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
        )

        app.include_router(observability.router)
        app.include_router(models.router)
        app.include_router(completions.router)
        app.include_router(chat.router)
        app.include_router(embeddings.router)

        # NOTE: For debugging purposes, should generally be restricted or removed
        self._add_cors_middleware(app)
        if self.restricted_apis != None:
            self._add_api_restriction_middleware(app)

        return app

    def _add_cors_middleware(self, app: FastAPI):
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

    def _add_api_restriction_middleware(self, app: FastAPI):
        app.add_middleware(
            APIRestrictionMiddleware, restricted_apis=self.restricted_apis
        )
        print(
            f"[INFO] API restrictions enabled. Restricted API endpoints: {self.restricted_apis.RestrictionDict()}"
        )
