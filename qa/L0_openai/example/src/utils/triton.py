import os
import time

import numpy as np
import tritonserver
from src.schemas.openai import CreateChatCompletionRequest, CreateCompletionRequest
from src.utils.tokenizer import get_tokenizer

# TODO: Remove
SUPPORTED_BACKENDS: set = {"vllm", "tensorrtllm"}
KNOWN_MODELS = {"gpt2": "hf:gpt2"}


# TODO: Re-organize helpers
def load_model(server):
    model = None
    backends = []
    tokenizer = None
    model_source_name = None
    for model_name, version in server.models().keys():
        if version != -1:
            continue
        current_model = server.load(model_name)
        backends.append(current_model.config()["backend"])
        if model_name in KNOWN_MODELS.keys():
            model = current_model
            model_source_name = KNOWN_MODELS[model_name].replace("hf:", "")
            tokenizer = get_tokenizer(model_source_name)
    if model and tokenizer:
        for backend in backends:
            if backend in SUPPORTED_BACKENDS:
                return model, int(time.time()), backend, tokenizer, model_source_name
    return None, None, None, None, None


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

    model, model_create_time, backend, tokenizer, model_source_name = load_model(server)

    if not (model and backend and tokenizer and model_create_time):
        raise Exception("Unknown Model")

    print(f"\n\nModel: {model.name} Loaded with Backend: {backend}\n\n")

    # if backend == "vllm":
    #    create_inference_request = create_vllm_inference_request
    # elif backend == "tensorrtllm":
    #    create_inference_request = create_trtllm_inference_request

    return server


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
    # FIXME: It seems that some subset of these keys will cause the model to not return a response
    addl_excludes = {"user", "seed", "stop", "suffix", "logprobs", "logit_bias"}
    sampling_parameters = request.model_dump(
        exclude=excludes.union(addl_excludes),
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
