#!/usr/bin/env python3
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared helpers for the TensorRT-LLM LLM API (Python) backend tests.

The LLM API model is a Python-backend model with a text-in / text-out
interface. Every tunable is an optional input named ``sampling_param_*`` (or
``return_*`` / ``lora_*``); the model does not use server-side batching, so
all tensors have shape ``[1]``.
"""

import json
import os
import queue
import subprocess
import time

import numpy as np
import pytest
import requests
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

MODEL_NAME = os.environ.get("LLMAPI_MODEL_NAME", "tensorrt_llm")
HOST = os.environ.get("TRITON_HOST", "localhost")
HTTP_PORT = int(os.environ.get("TRITON_HTTP_PORT", "8000"))
GRPC_PORT = int(os.environ.get("TRITON_GRPC_PORT", "8001"))
METRICS_PORT = int(os.environ.get("TRITON_METRICS_PORT", "8002"))
HTTP_URL = f"http://{HOST}:{HTTP_PORT}"
GRPC_URL = f"{HOST}:{GRPC_PORT}"

# Optional inputs of the model and their numpy types (config.pbtxt).
INPUT_DTYPES = {
    "streaming": np.bool_,
    "stop": np.bool_,
    "sampling_param_best_of": np.int32,
    "sampling_param_temperature": np.float32,
    "sampling_param_top_k": np.int32,
    "sampling_param_top_p": np.float32,
    "sampling_param_frequency_penalty": np.float32,
    "sampling_param_presence_penalty": np.float32,
    "sampling_param_max_tokens": np.int32,
    "sampling_param_seed": np.uint64,
    "sampling_param_exclude_input_from_output": np.bool_,
    "sampling_param_return_perf_metrics": np.bool_,
    "return_finish_reason": np.bool_,
    "return_stop_reason": np.bool_,
    "return_cumulative_logprob": np.bool_,
    "lora_id": np.uint64,
}
STRING_INPUTS = ("sampling_param_stop", "lora_name", "lora_path", "lora_ckpt_source")

# Outputs returned when sampling_param_return_perf_metrics is set.
PERF_METRIC_OUTPUTS = (
    "kv_cache_reused_block",
    "kv_cache_missed_block",
    "kv_cache_alloc_new_blocks",
    "kv_cache_alloc_total_blocks",
    "kv_cache_hit_rate",
    "arrival_time_ns",
    "first_scheduled_time_ns",
    "first_token_time_ns",
    "last_token_time_ns",
    "acceptance_rate",
    "total_accepted_draft_tokens",
    "total_draft_tokens",
)


def _string_tensor(name, values):
    data = np.array([v.encode() for v in values], dtype=np.object_)
    tensor = grpcclient.InferInput(name, [len(values)], "BYTES")
    tensor.set_data_from_numpy(data)
    return tensor


def build_inputs(prompt, **params):
    """Return gRPC inputs for `prompt`; `params` are the optional inputs.

    ``text_input=None`` omits the prompt, to test the missing-input path.
    """
    inputs = []
    if prompt is not None:
        inputs.append(_string_tensor("text_input", [prompt]))
    for name, value in params.items():
        if name in STRING_INPUTS:
            values = value if isinstance(value, (list, tuple)) else [value]
            inputs.append(_string_tensor(name, list(values)))
        else:
            dtype = INPUT_DTYPES[name]
            # np_to_triton_dtype() recognises the Python bool, not np.bool_.
            triton_dtype = "BOOL" if dtype is np.bool_ else np_to_triton_dtype(dtype)
            tensor = grpcclient.InferInput(name, [1], triton_dtype)
            tensor.set_data_from_numpy(np.array([value], dtype=dtype))
            inputs.append(tensor)
    return inputs


def output_text(result, name="text_output"):
    return result.as_numpy(name).flatten()[0].decode()


def grpc_infer(prompt, request_id="", **params):
    """Unary gRPC request. Raises InferenceServerException on a model error."""
    with grpcclient.InferenceServerClient(GRPC_URL) as client:
        return client.infer(
            MODEL_NAME,
            build_inputs(prompt, **params),
            request_id=request_id,
            client_timeout=120,
        )


class GrpcStream:
    """One bidirectional gRPC stream, collecting (result, error) pairs."""

    def __init__(self):
        self._client = grpcclient.InferenceServerClient(GRPC_URL)
        self._responses = queue.Queue()
        self._client.start_stream(
            callback=lambda result, error: self._responses.put((result, error))
        )

    def send(self, prompt, request_id="", **params):
        self.send_inputs(MODEL_NAME, build_inputs(prompt, **params), request_id)

    def send_inputs(self, model, inputs, request_id=""):
        """Send prepared `inputs` to `model`, e.g. an ensemble over the LLM."""
        self._client.async_stream_infer(
            model,
            inputs,
            request_id=request_id,
            enable_empty_final_response=True,
        )

    def get(self, timeout=60):
        """Next (result, error); (None, None) if nothing arrives in time."""
        try:
            return self._responses.get(timeout=timeout)
        except queue.Empty:
            return None, None

    def collect(self, timeout=60):
        """Read responses until the final one. Returns (texts, error, got_final)."""
        texts, error = [], None
        while True:
            result, err = self.get(timeout)
            if result is None and err is None:
                return texts, error, False
            if err is not None:
                return texts, err, True
            params = result.get_response().parameters
            out = result.as_numpy("text_output")
            if out is not None:
                texts.append(out.flatten()[0].decode())
            final = params.get("triton_final_response")
            if final is not None and final.bool_param:
                return texts, error, True

    def close(self, cancel_requests=False):
        self._client.stop_stream(cancel_requests=cancel_requests)
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def http_generate(payload, endpoint="generate", timeout=120):
    return requests.post(
        f"{HTTP_URL}/v2/models/{MODEL_NAME}/{endpoint}", json=payload, timeout=timeout
    )


def http_generate_stream(payload, timeout=120):
    """POST /generate_stream; returns (status, [json events])."""
    events = []
    with requests.post(
        f"{HTTP_URL}/v2/models/{MODEL_NAME}/generate_stream",
        json=payload,
        stream=True,
        timeout=timeout,
    ) as response:
        for line in response.iter_lines():
            if line.startswith(b"data:"):
                events.append(json.loads(line[len(b"data:") :]))
        return response.status_code, events


def server_ready():
    try:
        return requests.get(f"{HTTP_URL}/v2/health/ready", timeout=5).status_code == 200
    except requests.RequestException:
        return False


def metric_value(name, model=MODEL_NAME):
    """Sum of Prometheus samples called `name` labelled with `model`."""
    text = requests.get(f"http://{HOST}:{METRICS_PORT}/metrics", timeout=10).text
    total = 0.0
    for line in text.splitlines():
        if line.startswith(name + "{") and f'model="{model}"' in line:
            total += float(line.rsplit(" ", 1)[1])
    return total


def server_log_path():
    """Log of the running server; each conftest sets SERVER_LOG when it starts one."""
    return os.environ.get("SERVER_LOG")


def server_log_text():
    path = server_log_path()
    if not path or not os.path.exists(path):
        return ""
    with open(path, errors="replace") as f:
        return f.read()


def server_log_count(needle):
    """Number of server log lines containing `needle` (log must be verbose)."""
    path = server_log_path()
    if not path or not os.path.exists(path):
        return 0
    result = subprocess.run(
        ["grep", "-c", "-F", needle, path], capture_output=True, text=True
    )
    return int(result.stdout.strip() or 0)


def wait_until(predicate, timeout, interval=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def gpu_count():
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return 0
    return sum(1 for line in out.splitlines() if line.startswith("GPU "))


def gpu_memory_used_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sum(int(v) for v in out.split())


def require_gpus(needed):
    """Skip loudly, or fail when REQUIRE_GPUS is set, if too few GPUs exist.

    A silent skip would let a TP1-only run report green for a matrix that also
    contains TP>1 cases, which is how coverage is lost without anyone noticing.
    """
    available = gpu_count()
    if available >= needed:
        return
    message = f"needs {needed} GPU(s), {available} available"
    if os.environ.get("REQUIRE_GPUS") == "1":
        pytest.fail(f"REQUIRE_GPUS=1 but test {message}", pytrace=False)
    pytest.skip(message)
