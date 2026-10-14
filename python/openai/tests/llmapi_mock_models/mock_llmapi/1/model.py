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

import json

import numpy as np
import triton_python_backend_utils as pb_utils

OPTIONAL_INPUTS = (
    "streaming",
    "sampling_param_max_tokens",
    "sampling_param_stop",
    "sampling_param_temperature",
    "sampling_param_top_p",
    "sampling_param_frequency_penalty",
    "sampling_param_presence_penalty",
    "sampling_param_seed",
    "sampling_param_exclude_input_from_output",
)
# A prompt starting with this makes the model answer with the optional inputs
# it received, so a test can check how a request was converted.
ECHO_PARAMS_PREFIX = "echo-params:"


def _value(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return None
    array = tensor.as_numpy()
    if array.dtype == np.object_:
        values = [
            v.decode() if isinstance(v, bytes) else str(v) for v in array.flatten()
        ]
        return values[0] if name == "text_input" else values
    return array.flatten()[0].item()


class TritonPythonModel:
    def execute(self, requests):
        for request in requests:
            sender = request.get_response_sender()
            prompt = _value(request, "text_input")
            received = {name: _value(request, name) for name in OPTIONAL_INPUTS}
            received = {k: v for k, v in received.items() if v is not None}

            if prompt.startswith(ECHO_PARAMS_PREFIX):
                pieces = ["PARAMS:" + json.dumps(received, sort_keys=True)]
            else:
                count = received.get("sampling_param_max_tokens", 4)
                pieces = [f" w{i}" for i in range(count)]
            # As the real model does: the prompt is echoed unless excluded.
            prefix = (
                ""
                if received.get("sampling_param_exclude_input_from_output")
                else prompt
            )

            streaming = bool(received.get("streaming", False))
            steps = range(1, len(pieces) + 1) if streaming else [len(pieces)]
            for step in steps:
                text = prefix + "".join(pieces[:step])
                output = pb_utils.Tensor(
                    "text_output", np.array([text], dtype=np.object_)
                )
                flags = (
                    pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    if step == steps[-1]
                    else 0
                )
                sender.send(
                    pb_utils.InferenceResponse(output_tensors=[output]), flags=flags
                )
        return None
