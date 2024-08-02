from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Union

import tritonserver
from fastapi import FastAPI
from src.routers import chat_completions, completions, models, utilities
from src.utils.triton import init_tritonserver
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting FastAPI app lifespan...")
    # Start the tritonserver on FastAPI app startup
    app.server = init_tritonserver()

    yield

    # Cleanup the tritonserver on FastAPI app shutdown
    print("Shutting down FastAPI app lifespan...")
    if app.server:
        print("Shutting down Triton Inference Server...")
        app.server.stop()


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
    servers=[{"url": "https://api.openai.com/v1"}],
    lifespan=lifespan,
)

app.include_router(utilities.router)
app.include_router(models.router)
app.include_router(completions.router)
app.include_router(chat_completions.router)

server: tritonserver.Server
model: tritonserver.Model
model_source_name: str
model_create_time: int
backend: str
tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
create_inference_request = None
