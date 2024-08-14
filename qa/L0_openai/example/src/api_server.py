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
    print("Starting FastAPI app lifespan...")
    # Start the tritonserver on FastAPI app startup
    server, model_metadatas = init_tritonserver()
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

    # TODO: Add common logger and use logger.debug in place of current print
    # statements for debugging purposes.

    return app
