# Copyright 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import ctypes
import json
import os
import re
import sys
import traceback
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import tritonserver
from pydantic import BaseModel
from schemas.openai import (
    ChatCompletionNamedToolChoice,
    ChatCompletionTokenLogprob,
    ChatCompletionToolChoiceOption1,
    CompletionUsage,
    CreateChatCompletionRequest,
    CreateCompletionRequest,
    CreateEmbeddingRequest,
    EmbeddingUsage,
    Logprobs,
    TopLogprob,
)
from utils.utils import ClientError, ServerError


class RequestKind(Enum):
    GENERATION = 1
    EMBEDDING = 2


@dataclass
class TritonLoraConfig:
    name: str

    # Unique fields for TensorRT-LLM backend
    task_id: Optional[int] = None
    path: Optional[str] = None
    is_registered: Optional[bool] = False


def _create_vllm_generate_request(
    model,
    prompt,
    request: CreateChatCompletionRequest | CreateCompletionRequest,
    lora_config: TritonLoraConfig | None,
    echo_tensor_name: str | None,
    default_max_tokens: int,
):
    inputs = {}
    # Exclude non-sampling parameters so they aren't passed to vLLM
    excludes = {
        "model",
        "stream",
        "messages",
        "prompt",
        "echo",
        "store",
        "metadata",
        "response_format",
        "service_tier",
        "stream_options",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "user",
        "function_call",
        "functions",
        "suffix",
        "max_completion_tokens",
        # will be handled explicitly
        "max_tokens",
        "logprobs",
        "top_logprobs",
        # not supported for vLLM backend (removed from vLLM V1) but supported for TRT-LLM/Python backend
        "best_of",
    }

    # NOTE: The exclude_none is important, as internals may not support
    # values of NoneType at this time.
    sampling_parameters = request.model_dump(
        exclude=excludes,
        exclude_none=True,
    )

    request_logprobs = False
    # Indicates CreateChatCompletionRequest
    if hasattr(request, "max_completion_tokens"):
        if request.max_completion_tokens is not None:
            sampling_parameters["max_tokens"] = request.max_completion_tokens
        # Fallback to deprecated request.max_tokens
        elif request.max_tokens is not None:
            sampling_parameters["max_tokens"] = request.max_tokens
        # If neither is set, use a default value for max_tokens
        else:
            sampling_parameters["max_tokens"] = default_max_tokens

        # Handle logprobs for chat completions
        # OpenAI API: logprobs (bool), top_logprobs (int 0-20)
        # vLLM API: logprobs (int) - number of top token logprobs to return
        if request.logprobs and request.top_logprobs is not None:
            sampling_parameters["logprobs"] = request.top_logprobs
            request_logprobs = True
        elif request.logprobs:
            # If logprobs=True but top_logprobs not specified, default to 1
            sampling_parameters["logprobs"] = 1
            request_logprobs = True
    # Indicates CreateCompletionRequest
    else:
        if request.max_tokens is not None:
            sampling_parameters["max_tokens"] = request.max_tokens
        else:
            sampling_parameters["max_tokens"] = default_max_tokens

        # Handle logprobs for completions
        # OpenAI API: logprobs (int 0-5) - number of top token log probs
        # vLLM API: logprobs (int) - same behavior, pass directly
        if request.logprobs is not None and request.logprobs > 0:
            sampling_parameters["logprobs"] = request.logprobs
            request_logprobs = True
    inputs["return_logprobs"] = np.bool_([request_logprobs])

    if lora_config is not None:
        sampling_parameters["lora_name"] = lora_config.name

    guided_json = _get_guided_json_from_tool(request)
    if guided_json is not None:
        from vllm.sampling_params import StructuredOutputsParams

        sampling_parameters["structured_outputs"] = json.dumps(
            asdict(StructuredOutputsParams(json=guided_json))
        )
    sampling_parameters = json.dumps(sampling_parameters)

    exclude_input_in_output = True
    echo = getattr(request, "echo", None)
    if echo is not None:
        exclude_input_in_output = not echo

    inputs["text_input"] = [prompt]
    inputs["stream"] = np.bool_([request.stream])
    inputs[echo_tensor_name] = np.bool_([exclude_input_in_output])
    # Pass sampling_parameters as serialized JSON string input to support List
    # fields like 'stop' that aren't supported by TRITONSERVER_Parameters yet.
    inputs["sampling_parameters"] = [sampling_parameters]
    inputs["return_num_input_tokens"] = np.bool_([True])
    inputs["return_num_output_tokens"] = np.bool_([True])
    return model.create_request(inputs=inputs)


def _create_trtllm_generate_request(
    model,
    prompt,
    request: CreateChatCompletionRequest | CreateCompletionRequest,
    lora_config: TritonLoraConfig | None,
    echo_tensor_name: str | None,
    default_max_tokens: int,
):
    inputs = {}
    inputs["text_input"] = [[prompt]]
    inputs["stream"] = np.bool_([[request.stream]])

    # Indicates CreateChatCompletionRequest
    if hasattr(request, "max_completion_tokens"):
        if request.max_completion_tokens is not None:
            inputs["max_tokens"] = np.int32([[request.max_completion_tokens]])
        # Fallback to deprecated request.max_tokens
        elif request.max_tokens is not None:
            inputs["max_tokens"] = np.int32([[request.max_tokens]])
        # If neither is set, use a default value for max_tokens
        else:
            inputs["max_tokens"] = np.int32([[default_max_tokens]])
    # Indicates CreateCompletionRequest
    elif request.max_tokens is not None:
        inputs["max_tokens"] = np.int32([[request.max_tokens]])
    else:
        inputs["max_tokens"] = np.int32([[default_max_tokens]])

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
        inputs["seed"] = np.uint64([[request.seed]])
    if request.temperature is not None:
        inputs["temperature"] = np.float32([[request.temperature]])
    # Only limited TRT-LLM models support "echo" (inflight_batcher_llm, disaggregated_serving, llmapi)
    echo = getattr(request, "echo", None)
    if echo is not None and echo_tensor_name is not None:
        inputs[echo_tensor_name] = np.bool_([[not echo]])

    guided_json = _get_guided_json_from_tool(request)
    if guided_json is not None:
        inputs["guided_decoding_guide_type"] = [["json_schema"]]
        inputs["guided_decoding_guide"] = [[guided_json]]

    if lora_config is not None:
        # To perform inference with a specific LoRA for the first time `lora_task_id` `lora_weights` and `lora_config` must all be given.
        # The LoRA will be cached, so that subsequent requests for the same task only require `lora_task_id`.
        inputs["lora_task_id"] = np.uint64([[lora_config.task_id]])
        if not lora_config.is_registered:
            lora_weights_data = np.load(
                os.path.join(lora_config.path, "model.lora_weights.npy")
            )
            lora_config_data = np.load(
                os.path.join(lora_config.path, "model.lora_config.npy")
            )
            inputs["lora_weights"] = lora_weights_data
            inputs["lora_config"] = lora_config_data
            lora_config.is_registered = True

    inputs["return_num_input_tokens"] = np.bool_([[True]])
    inputs["return_num_output_tokens"] = np.bool_([[True]])
    return model.create_request(inputs=inputs)


def _create_vllm_embedding_request(
    model,
    request: CreateEmbeddingRequest,
):
    inputs = {}
    embedding_request = {}
    embedding_request["input"] = request.input

    pooling_params = {}
    dims = request.dimensions
    if dims is not None:
        pooling_params["dimensions"] = [dims]
    embedding_request["pooling_params"] = pooling_params

    inputs["embedding_request"] = [json.dumps(embedding_request)]
    inputs["return_num_input_tokens"] = np.bool_([True])
    inputs["return_num_output_tokens"] = np.bool_([True])
    return model.create_request(inputs=inputs)


def _create_trtllm_embedding_request(
    model,
    request: CreateEmbeddingRequest,
):
    raise ClientError(
        "TRT-LLM backend and Python backend do not support embedding requests"
    )


def _construct_string_from_pointer(pointer: int, size: int) -> str:
    """Constructs a Python string from a C pointer and size."""

    # Create a ctypes string buffer
    string_buffer = ctypes.create_string_buffer(size + 1)  # +1 for null terminator

    # Copy the data from the pointer to the buffer
    ctypes.memmove(string_buffer, pointer, size)

    # Convert the buffer to a Python string
    return string_buffer.value.decode("utf-8")  # Adjust encoding if needed


def _get_volume(shape: Iterable[int]) -> int:
    volume = 1
    for dim in shape:
        volume *= dim

    return volume


def _to_string(tensor: tritonserver.Tensor) -> str:
    # FIXME: This could be a bit more robust by reading byte size from first
    # 4 bytes and then just reading the first string, rather than assuming
    # single string, assuming it's of similar performance to do so.

    # The following optimization to read string directly from buffer assumes
    # there is only a single string, so enforce it to avoid obscure errors.
    volume = _get_volume(tensor.shape)
    if volume != 1:
        raise ServerError(
            f"Expected to find 1 string in the output, found {volume} instead."
        )
    if tensor.size < 4:
        raise ServerError(
            f"Expected string buffer to contain its serialized byte size, but found size of {tensor.size}."
        )

    # NOTE: +/- 4 accounts for serialized byte string length in first 4 bytes of buffer
    return _construct_string_from_pointer(tensor.data_ptr + 4, tensor.size - 4)


@dataclass
class _StreamingUsageAccumulator:
    """Helper class to accumulate token usage from a streaming response."""

    backend: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _prompt_tokens_set: bool = field(init=False, default=False)

    def update(self, response: tritonserver.InferenceResponse):
        """Extracts usage from a response and updates the token counts."""
        usage = _get_usage_from_response(response, self.backend, RequestKind.GENERATION)
        if usage:
            # The prompt_tokens is received with every chunk but should only be set once.
            if not self._prompt_tokens_set:
                self.prompt_tokens = usage.prompt_tokens
                self._prompt_tokens_set = True
            self.completion_tokens += usage.completion_tokens

    def get_final_usage(self) -> Optional[CompletionUsage]:
        """
        Returns the final populated CompletionUsage object if any tokens were tracked.
        """
        # If _prompt_tokens_set is True, it means we have received and processed
        # at least one valid usage payload.
        if self._prompt_tokens_set:
            return CompletionUsage(
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                total_tokens=self.prompt_tokens + self.completion_tokens,
            )
        return None


def _get_usage_from_response(
    response: tritonserver._api._response.InferenceResponse,
    backend: str,
    request_type: RequestKind,
) -> Optional[CompletionUsage | EmbeddingUsage]:
    """
    Extracts token usage statistics from a Triton inference response.
    """
    prompt_tokens = None
    completion_tokens = None

    if (
        "num_input_tokens" in response.outputs
        and "num_output_tokens" in response.outputs
    ):
        input_token_tensor = response.outputs["num_input_tokens"]
        output_token_tensor = response.outputs["num_output_tokens"]

        if input_token_tensor.data_type == tritonserver.DataType.UINT32:
            prompt_tokens_ptr = ctypes.cast(
                input_token_tensor.data_ptr, ctypes.POINTER(ctypes.c_uint32)
            )
            prompt_tokens = prompt_tokens_ptr[0]
        elif input_token_tensor.data_type == tritonserver.DataType.INT32:
            prompt_tokens_ptr = ctypes.cast(
                input_token_tensor.data_ptr, ctypes.POINTER(ctypes.c_int32)
            )
            prompt_tokens = prompt_tokens_ptr[0]

        if output_token_tensor.data_type == tritonserver.DataType.UINT32:
            completion_tokens_ptr = ctypes.cast(
                output_token_tensor.data_ptr, ctypes.POINTER(ctypes.c_uint32)
            )
            completion_tokens = completion_tokens_ptr[0]
        elif output_token_tensor.data_type == tritonserver.DataType.INT32:
            completion_tokens_ptr = ctypes.cast(
                output_token_tensor.data_ptr, ctypes.POINTER(ctypes.c_int32)
            )
            completion_tokens = completion_tokens_ptr[0]

        if prompt_tokens is not None:
            if request_type == RequestKind.GENERATION and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
                return CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            elif request_type == RequestKind.EMBEDDING:
                return EmbeddingUsage(
                    prompt_tokens=prompt_tokens,
                    total_tokens=prompt_tokens,
                )

    return None


# TODO: Use tritonserver.InferenceResponse when support is published
def _get_output(response: tritonserver._api._response.InferenceResponse) -> str:
    if "text_output" in response.outputs:
        tensor = response.outputs["text_output"]

        # Alternative method, creates the same string, but goes through
        # deserialization, numpy, and dlpack overhead:
        # return tensor.to_bytes_array()[0].decode("utf-8")

        # Optimized method
        return _to_string(tensor)

    return ""


def _get_logprobs_from_response(
    response: tritonserver._api._response.InferenceResponse,
) -> Optional[List[Dict]]:
    """
    Extracts logprobs from a Triton inference response (vLLM backend).

    Returns:
        List of dictionaries containing logprobs data, or None if not available.
        Format: [
            {
                token_id: {
                    "logprob": float,
                    "rank": int,
                    "decoded_token": str
                }
            },
            ...
        ]
    """
    if "logprobs" not in response.outputs:
        return None

    logprobs_tensor = response.outputs["logprobs"]
    if logprobs_tensor is None:
        return None

    # The logprobs are stored as JSON string (vLLM backend)
    logprobs_str = _to_string(logprobs_tensor)

    if logprobs_str == "null":
        return None

    try:
        logprobs_data = json.loads(logprobs_str)
        return logprobs_data
    except json.JSONDecodeError:
        return None


def _get_openai_chat_format_logprobs_from_vllm_response(
    response: tritonserver._api._response.InferenceResponse,
) -> Optional[List[ChatCompletionTokenLogprob]]:
    """
    Convert logprobs from a Triton inference response (vLLM backend) to OpenAI chat completion format.

    Args:
        response: Triton inference response containing logprobs output.

    Returns:
        List of ChatCompletionTokenLogprob objects, or None if no logprobs available.
    """
    vllm_logprobs = _get_logprobs_from_response(response)

    if not vllm_logprobs:
        return None

    openai_logprobs = []
    for token_logprobs_dict in vllm_logprobs:
        if not token_logprobs_dict:
            continue

        # Sort by rank to identify the selected token (rank=1 is always the chosen token)
        sorted_tokens = sorted(
            token_logprobs_dict.items(), key=lambda x: x[1].get("rank", sys.maxsize)
        )

        # The first token (lowest rank) is the selected token
        selected_token_id, selected_token_data = sorted_tokens[0]
        selected_token = selected_token_data["decoded_token"]
        selected_logprob = selected_token_data["logprob"]

        # Convert to bytes representation
        token_bytes = list(selected_token.encode("utf-8"))

        top_logprobs_list = []
        for token_id, token_data in sorted_tokens:
            decoded_token = token_data["decoded_token"]
            top_logprobs_list.append(
                TopLogprob(
                    token=decoded_token,
                    logprob=token_data["logprob"],
                    bytes=list(decoded_token.encode("utf-8")),
                )
            )

        openai_logprobs.append(
            ChatCompletionTokenLogprob(
                token=selected_token,
                logprob=selected_logprob,
                bytes=token_bytes,
                top_logprobs=top_logprobs_list,
            )
        )

    return openai_logprobs


def _get_openai_completion_format_logprobs_from_vllm_response(
    response: tritonserver._api._response.InferenceResponse,
) -> Optional[Logprobs]:
    """
    Convert logprobs from a Triton inference response (vLLM backend) to OpenAI completion format.

    Args:
        response: Triton inference response containing logprobs output.

    Returns:
        Logprobs object for completions API, or None if no logprobs available.
    """
    vllm_logprobs = _get_logprobs_from_response(response)

    if not vllm_logprobs:
        return None

    text_offset = []
    token_logprobs = []
    tokens = []
    top_logprobs = []

    current_offset = 0
    for token_logprobs_dict in vllm_logprobs:
        if not token_logprobs_dict:
            continue

        # Sort by rank to identify the selected token (rank=1 is always the chosen token)
        sorted_tokens = sorted(
            token_logprobs_dict.items(), key=lambda x: x[1].get("rank", sys.maxsize)
        )

        # The first token (lowest rank) is the selected token
        selected_token_id, selected_token_data = sorted_tokens[0]
        selected_token = selected_token_data["decoded_token"]
        selected_logprob = selected_token_data["logprob"]

        text_offset.append(current_offset)
        token_logprobs.append(selected_logprob)
        tokens.append(selected_token)

        # Build top_logprobs dict for this position
        top_logprobs_dict = {}
        for token_id, token_data in sorted_tokens:
            decoded_token = token_data["decoded_token"]
            top_logprobs_dict[decoded_token] = token_data["logprob"]
        top_logprobs.append(top_logprobs_dict)

        current_offset += len(selected_token)

    return Logprobs(
        text_offset=text_offset,
        token_logprobs=token_logprobs,
        tokens=tokens,
        top_logprobs=top_logprobs,
    )


def _validate_triton_responses_non_streaming(
    responses: List[tritonserver._api._response.InferenceResponse],
):
    num_responses = len(responses)
    if 1 <= num_responses <= 2:
        if responses[-1].final != True:
            raise ServerError("Unexpected internal error with incorrect response flags")
    else:
        raise ServerError(
            f"Unexpected number of responses: {num_responses}, expected 1 or 2."
        )


def _get_guided_json_from_tool(
    request: CreateChatCompletionRequest | CreateCompletionRequest,
) -> Optional[Union[str, dict, BaseModel]]:
    if isinstance(request, CreateChatCompletionRequest):
        if request.tool_choice is None or not request.tools:
            return None

        if type(request.tool_choice.root) is ChatCompletionNamedToolChoice:
            tool_name = request.tool_choice.root.function.name
        elif request.tool_choice.root == ChatCompletionToolChoiceOption1.required:
            tool_name = request.tools[0].function.name
        else:
            return None

        tools = {tool.function.name: tool.function for tool in request.tools}
        if tool_name not in tools:
            raise ClientError(f"Tool '{tool_name}' has not been passed in `tools`.")
        tool = tools[tool_name]
        return tool.parameters.model_dump_json()

    return None


def _parse_lora_configs(
    model_repository: str | list[str], model_name: str, model_version: int, backend: str
) -> None | List[tuple[str, str]]:
    if (
        len(model_name) == 0
        or model_name.isspace()
        or "/" in model_name
        or "\\" in model_name
    ):
        raise ValueError(
            f"Invalid model name: '{model_name}'. Model names must be valid file-system-path segment names."
        )

    lora_configs = []
    lora_task_id = 1
    repo_paths = model_repository
    if isinstance(repo_paths, str):
        repo_paths = [repo_paths]
    for repo_path in repo_paths:
        model_path = os.path.join(repo_path, model_name)
        if (not Path(model_path).is_relative_to(repo_path)) or (
            os.path.normpath(model_path) != model_path
        ):
            raise ValueError(
                f"Invalid model name: '{model_name}'. Model names must be valid file-system-path segment names."
            )

        model_path = os.path.normpath(model_path)
        if not os.path.isdir(model_path):
            # Cloud path?
            return None
        if model_version <= 0:
            for version_path in os.listdir(model_path):
                version = os.path.basename(version_path)
                if re.fullmatch(r"^[0-9]+$", version) is None:
                    continue
                model_version = max(model_version, int(version))
            if model_version <= 0:
                # Model directory is malformed?
                return None
        version_path = os.path.join(model_path, str(model_version))
        lora_config_path = os.path.join(version_path, "multi_lora.json")

        if backend == "vllm":
            is_lora_enabled = False
            model_file_path = os.path.join(version_path, "model.json")
            try:
                with open(model_file_path, "r") as f:
                    config = json.load(f)
                    if "enable_lora" in config:
                        # The value could be a string or a bool.
                        is_lora_enabled = str(config["enable_lora"]).lower() == "true"
            except Exception:
                # Model directory or model.json is malformed?
                return None
            if is_lora_enabled != True:
                continue
        else:
            # TRT-LLM backend does not use model.json
            if not os.path.exists(lora_config_path):
                continue

        try:
            with open(lora_config_path, "r") as f:
                lora_config = json.load(f)
                for lora_name, lora_path in lora_config.items():
                    print(f"backend: {backend}")
                    if backend == "vllm":
                        lora_configs.append(TritonLoraConfig(name=lora_name))
                    else:
                        lora_weights_path = os.path.join(
                            lora_path, "model.lora_weights.npy"
                        )
                        lora_config_path = os.path.join(
                            lora_path, "model.lora_config.npy"
                        )
                        if not os.path.exists(lora_weights_path):
                            raise ServerError(
                                f"LoRA weights file not found for '{lora_name}' at path: {lora_weights_path}"
                            )
                        if not os.path.exists(lora_config_path):
                            raise ServerError(
                                f"LoRA config file not found for '{lora_name}' at path: {lora_config_path}"
                            )

                        lora_configs.append(
                            TritonLoraConfig(
                                name=lora_name, path=lora_path, task_id=lora_task_id
                            )
                        )
                        lora_task_id += 1
        except ServerError as e:
            raise e
        except Exception as e:
            # LoRA is enabled but its list is not provided or malformed?
            print(traceback.format_exc())
            return None
    return lora_configs
