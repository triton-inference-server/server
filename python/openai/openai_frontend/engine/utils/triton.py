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

import ctypes
from typing import Iterable, List

import numpy as np
import tritonserver
from schemas.openai import CreateChatCompletionRequest, CreateCompletionRequest


def _create_vllm_inference_request(
    model, prompt, request: CreateChatCompletionRequest | CreateCompletionRequest
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
    }

    # NOTE: The exclude_none is important, as internals may not support
    # values of NoneType at this time.
    sampling_parameters = request.model_dump_json(
        exclude=excludes,
        exclude_none=True,
    )

    exclude_input_in_output = True
    echo = getattr(request, "echo", None)
    if echo is not None:
        exclude_input_in_output = not echo

    inputs["text_input"] = [prompt]
    inputs["stream"] = np.bool_([request.stream])
    inputs["exclude_input_in_output"] = np.bool_([exclude_input_in_output])
    # Pass sampling_parameters as serialized JSON string input to support List
    # fields like 'stop' that aren't supported by TRITONSERVER_Parameters yet.
    inputs["sampling_parameters"] = [sampling_parameters]
    return model.create_request(inputs=inputs)


def _create_trtllm_inference_request(
    model, prompt, request: CreateChatCompletionRequest | CreateCompletionRequest
):
    inputs = {}
    inputs["text_input"] = [[prompt]]
    inputs["stream"] = np.bool_([[request.stream]])
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
    # FIXME: TRT-LLM doesn't currently support runtime changes of 'echo' and it
    # is configured at model load time, so we don't handle it here for now.
    return model.create_request(inputs=inputs)


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
        raise Exception(
            f"Expected to find 1 string in the output, found {volume} instead."
        )
    if tensor.size < 4:
        raise Exception(
            f"Expected string buffer to contain its serialized byte size, but found size of {tensor.size}."
        )

    # NOTE: +/- 4 accounts for serialized byte string length in first 4 bytes of buffer
    return _construct_string_from_pointer(tensor.data_ptr + 4, tensor.size - 4)


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


def _validate_triton_responses_non_streaming(
    responses: List[tritonserver._api._response.InferenceResponse],
):
    num_responses = len(responses)
    if num_responses == 1 and responses[0].final != True:
        raise Exception("Unexpected internal error with incorrect response flags")
    if num_responses == 2 and responses[-1].final != True:
        raise Exception("Unexpected internal error with incorrect response flags")
    if num_responses > 2:
        raise Exception(f"Unexpected number of responses: {num_responses}, expected 1.")
