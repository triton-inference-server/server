import os
import time
import typing
from dataclasses import dataclass

import numpy as np
import tritonserver
from fastapi import HTTPException
from src.schemas.openai import CreateChatCompletionRequest, CreateCompletionRequest
from src.utils.tokenizer import get_tokenizer

# TODO: Refactor
# NOTE: Allow python backend for testing purposes
SUPPORTED_BACKENDS: set = {"vllm", "tensorrtllm", "python"}
LLM_BACKENDS: set = {"vllm", "tensorrtllm"}


# TODO: pydantic validation?
@dataclass
class TritonModelMetadata:
    # Name used in Triton model repository
    name: str
    # Name of backend used by Triton
    backend: str
    # Triton model object handle
    model: tritonserver.Model
    # TODO: Address typing
    tokenizer: typing.Optional[typing.Any]
    # Time that model was loaded by Triton
    create_time: int
    # TODO: Address typing
    request_convert_fn: typing.Optional[typing.Any]


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


# TODO: Refactor:
# NOTE: We need to figure out a few things while looking at the models in the
# triton model repository.
#   1. Which model should we interact with when sending requests to Triton core?
#       a. For a single model, this is trivial, and would support any backend.
#       b. For TRT-LLM, this should be 'ensemble' or 'tensorrt_llm_bls' following
#          TRT-LLM defaults/examples. However, this could also be renamed by the user
#          to have a more intuitive front-facing name, such as "llama3-8b". Note that
#          TRT-LLM pipelines produced by the Triton CLI will generally be renamed like
#          this. FIXME: This is a relatively fragile flow and should be improved.
#   2. Which tokenizer to use for things like applying a chat template or making
#      a tool/function call. These are primarily relevant for the /chat/completions
#      endpoint, but not the /completions endpoint.
#     - For now, require user-defined TOKENIZER for simplicity.
#   3. Which inputs/outputs/parameters should be set when creating the underlying
#      triton inference request? The inference request fields required will differ
#      for vLLM, TRT-LLM, and user-defined models like a custom python model. So we
#      need to know how to correctly translate the OpenAI schema parameters to
#      a triton inference request.
#     - For now, we will look for either vllm or trtllm in list of loaded backends,
#       and we consider python==trtllm for now due to possibility of python runtime.
#       We may want to consider using Triton's "runtime" config field for this for
#       easier detection instead.
def load_models(server):
    model_metadatas = []
    backends = []

    # TODO: Support tokenizers more generically or custom tokenizers, possibly
    # by looking for tokenizer.json in a pre-specified location?
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
        # TODO: Why skip known version? Already loaded?
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
        model_metadatas.append(metadata)

    return model_metadatas


def init_tritonserver():
    model_repository = os.environ.get(
        "TRITON_MODEL_REPOSITORY", "/opt/tritonserver/models"
    )
    log_verbose_level = int(os.environ.get("TRITON_LOG_VERBOSE_LEVEL", "0"))

    print("Starting Triton Server Core...")
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
        except:
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
    print(f"[DEBUG] {sampling_parameters=}")

    inputs["text_input"] = [prompt]
    inputs["stream"] = [request.stream]
    exclude_input_in_output = True
    echo = getattr(request, "echo", None)
    if echo:
        exclude_input_in_output = not echo
    inputs["exclude_input_in_output"] = [exclude_input_in_output]

    print(f"[DEBUG] Triton Inference Request {inputs=}")
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

    print(f"[DEBUG] Triton Inference Request {inputs=}")
    return model.create_request(inputs=inputs)
