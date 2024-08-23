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

import os
import time
import typing
from dataclasses import dataclass

import numpy as np
import tritonserver
from fastapi import HTTPException
from schemas.openai import CreateChatCompletionRequest, CreateCompletionRequest
from utils.tokenizer import get_tokenizer


# TODO: Stricter pydantic validation would be better in future
@dataclass
class TritonModelMetadata:
    # Name used in Triton model repository
    name: str
    # Name of backend used by Triton
    backend: str
    # Triton model object handle
    model: tritonserver.Model
    # Tokenizers used for chat templates
    tokenizer: typing.Optional[typing.Any]
    # Time that model was loaded by Triton
    create_time: int
    # Conversion format between OpenAI and Triton requests
    request_convert_fn: typing.Optional[typing.Any]


# TODO: Expose explicit flag to catch edge cases
def determine_request_format(backend):
    # Request conversion from OpenAI format to backend-specific format
    if backend == "vllm":
        request_convert_fn = create_vllm_inference_request
    # Python included to support TRT-LLM BLS model and TRT-LLM python runtime
    elif backend in ["tensorrtllm", "python"]:
        request_convert_fn = create_trtllm_inference_request
    else:
        request_convert_fn = None

    return request_convert_fn


def load_models(server):
    model_metadata = []
    backends = []

    # TODO: Consider support for custom tokenizers
    tokenizer = None
    tokenizer_model = os.environ.get("TOKENIZER")
    if tokenizer_model:
        print(f"Using env var TOKENIZER={tokenizer_model} to determine the tokenizer")
        tokenizer = get_tokenizer(tokenizer_model)

    models = []
    backends = []
    names = []
    # Load all triton models and gather the respective backends of each
    for name, version in server.models().keys():
        # Skip models that are already loaded, if any
        if version != -1:
            continue

        model = server.load(name)
        backend = model.config()["backend"]

        names.append(name)
        models.append(model)
        backends.append(backend)
        print(f"Loaded: {name=}, {backend=}, tokenizer={tokenizer_model}")

    create_time = int(time.time())

    # One tokenizer, convert function, and creation time for all loaded models.
    # NOTE: This doesn't currently support having both a vLLM and TRT-LLM
    # model loaded at the same time.
    for name, model, backend in zip(names, models, backends):
        metadata = TritonModelMetadata(
            name=name,
            backend=backend,
            model=model,
            tokenizer=tokenizer,
            create_time=create_time,
            request_convert_fn=determine_request_format(backend),
        )
        model_metadata.append(metadata)

    return model_metadata


def init_tritonserver():
    model_repository = os.environ.get(
        "TRITON_MODEL_REPOSITORY", "/opt/tritonserver/models"
    )
    log_verbose_level = int(os.environ.get("TRITON_LOG_VERBOSE_LEVEL", "0"))

    print("Starting Triton Server...")
    server = tritonserver.Server(
        model_repository=model_repository,
        log_verbose=log_verbose_level,
        log_info=True,
        log_warn=True,
        log_error=True,
        model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
    ).start(wait_until_ready=True)

    print("Loading Models...")
    metadatas = load_models(server)
    return server, metadatas


def get_output(response):
    if "text_output" in response.outputs:
        try:
            return response.outputs["text_output"].to_string_array()[0]
        except Exception:
            return str(response.outputs["text_output"].to_bytes_array()[0])
    return ""


def validate_triton_responses(responses):
    num_responses = len(responses)
    if num_responses == 1 and responses[0].final != True:
        raise HTTPException(
            status_code=400,
            detail="Unexpected internal error with incorrect response flags",
        )
    if num_responses == 2 and responses[-1].final != True:
        raise HTTPException(
            status_code=400,
            detail="Unexpected internal error with incorrect response flags",
        )
    if num_responses > 2:
        raise HTTPException(
            status_code=400,
            detail=f"Unexpected number of responses: {num_responses}, expected 1.",
        )


def create_vllm_inference_request(
    model, prompt, request: CreateChatCompletionRequest | CreateCompletionRequest
):
    inputs = {}
    excludes = {"model", "stream", "messages", "prompt", "echo"}

    # NOTE: The exclude_none is important, as internals may not support
    # values of NoneType at this time.
    sampling_parameters = request.model_dump(
        exclude=excludes,
        exclude_none=True,
    )
    inputs["text_input"] = [prompt]
    inputs["stream"] = [request.stream]
    exclude_input_in_output = True
    echo = getattr(request, "echo", None)
    if echo:
        exclude_input_in_output = not echo
    inputs["exclude_input_in_output"] = [exclude_input_in_output]
    return model.create_request(inputs=inputs, parameters=sampling_parameters)


def create_trtllm_inference_request(
    model, prompt, request: CreateChatCompletionRequest | CreateCompletionRequest
):
    inputs = {}
    inputs["text_input"] = [[prompt]]
    inputs["stream"] = [[request.stream]]
    if request.max_tokens:
        inputs["max_tokens"] = np.int32([[request.max_tokens]])
    if request.stop:
        if isinstance(request.stop, str):
            request.stop = [request.stop]
        inputs["stop_words"] = [request.stop]
    # Check "is not None" specifically, because values of zero are valid.
    if request.top_p is not None:
        inputs["top_p"] = np.float32([[request.top_p]])
    if request.frequency_penalty is not None:
        inputs["frequency_penalty"] = np.float32([[request.frequency_penalty]])
    if request.presence_penalty is not None:
        inputs["presence_penalty"] = np.float32([[request.presence_penalty]])
    if request.seed is not None:
        inputs["random_seed"] = np.uint64([[request.seed]])
    if request.temperature is not None:
        inputs["temperature"] = np.float32([[request.temperature]])
    return model.create_request(inputs=inputs)
