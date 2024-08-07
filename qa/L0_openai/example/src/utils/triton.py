import os
import time
import typing
from dataclasses import dataclass

import numpy as np
import tritonserver
from src.schemas.openai import CreateChatCompletionRequest, CreateCompletionRequest
from src.utils.tokenizer import get_tokenizer

# TODO: Refactor
# NOTE: Allow python backend for testing purposes
# TODO: How did this interact with BLS/TRTLLM models before this change?
SUPPORTED_BACKENDS: set = {"vllm", "tensorrtllm", "python"}
LLM_BACKENDS = {"vllm", "tensorrtllm"}  # TODO
KNOWN_MODELS = {"gpt2": "hf:gpt2"}


@dataclass
class TritonModelMetadata:
    # Name used in Triton model repository
    name: str
    # Name of backend used by Triton
    backend: str
    # Triton model object handle
    model: tritonserver.Model

    # TODO: Address typing
    tokenizer: typing.Any
    # Name in terms of a HuggingFace model or remote model registry name
    source_name: str
    # Time that model was loaded by Triton
    create_time: int


# TODO: Refactor - this function seems to load a single model,
# but iterates through all models?
def load_model(server):
    model = None
    backends = []
    tokenizer = None
    source_name = None
    model_name = None
    for model_name, version in server.models().keys():
        if version != -1:
            continue
        model = server.load(model_name)
        backends.append(model.config()["backend"])
        if model_name in KNOWN_MODELS.keys():
            source_name = KNOWN_MODELS[model_name].replace("hf:", "")
            tokenizer = get_tokenizer(source_name)

    create_time = int(time.time())
    backend = None
    for be in backends:
        if be in SUPPORTED_BACKENDS:
            backend = be
            break

    # TODO
    return TritonModelMetadata(
        name=model_name,
        backend=backend,
        model=model,
        tokenizer=tokenizer,
        source_name=source_name,
        create_time=create_time,
    )


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

    # TODO: Cleanup
    print("Loading Model...\n\n")

    # model, model_create_time, backend, tokenizer, _ = load_model(server)
    metadata = load_model(server)

    # TODO: pydantic validation?
    if not metadata.name:
        raise Exception("Unknown Model Name")

    if not metadata.model:
        raise Exception("Unknown Model")

    if not metadata.backend:
        raise Exception("Unsupported Backend")

    # NOTE: Allow no tokenizer for mock python model for testing purposes
    if not metadata.tokenizer and metadata.backend in LLM_BACKENDS:
        raise Exception("Unsupported Tokenizer")

    if not metadata.create_time:
        raise Exception("Unknown Model Creation Time")

    print(f"\n\nModel: {metadata.name} Loaded with Backend: {metadata.backend}\n\n")

    # if backend == "vllm":
    #    create_inference_request = create_vllm_inference_request
    # elif backend == "tensorrtllm":
    #    create_inference_request = create_trtllm_inference_request

    return server, metadata


def get_output(response):
    if "text_output" in response.outputs:
        try:
            return response.outputs["text_output"].to_string_array()[0]
        except:
            return str(response.outputs["text_output"].to_bytes_array()[0])
    return None


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
    print(f"[DEBUG] {inputs=}")

    return model.create_request(inputs=inputs, parameters=sampling_parameters)


# TODO: test
def create_trtllm_inference_request(
    model, prompt, request: CreateChatCompletionRequest | CreateCompletionRequest
):
    inputs = {}
    if model.name == "llama-3-8b-instruct":
        inputs["stop_words"] = [["<|eot_id|>", "<|end_of_text|>"]]
    inputs["text_input"] = [[prompt]]
    inputs["stream"] = [[request.stream]]
    if request.max_tokens:
        inputs["max_tokens"] = np.int32([[request.max_tokens]])
    if request.stop:
        if isinstance(request.stop, str):
            request.stop = [request.stop]
        inputs["stop_words"] = [request.stop]
    if request.top_p:
        inputs["top_p"] = np.float32([[request.top_p]])
    if request.frequency_penalty:
        inputs["frequency_penalty"] = np.float32([[request.frequency_penalty]])
    if request.presence_penalty:
        inputs["presence_penalty":] = np.int32([[request.presence_penalty]])
    if request.seed:
        inputs["random_seed"] = np.uint64([[request.seed]])
    if request.temperature:
        inputs["temperature"] = np.float32([[request.temperature]])

    return model.create_request(inputs=inputs)
