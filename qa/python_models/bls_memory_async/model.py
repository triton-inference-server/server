# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import triton_python_backend_utils as pb_utils


async def _send_identity_tensor(size, is_decoupled):
    tensor_size = [1, size]
    input0_np = np.random.randn(*tensor_size)
    input0 = pb_utils.Tensor("INPUT0", input0_np.astype(np.float32))
    infer_request = pb_utils.InferenceRequest(
        model_name="identity_fp32", inputs=[input0], requested_output_names=["OUTPUT0"]
    )

    if is_decoupled:
        infer_responses = await infer_request.async_exec(decoupled=True)
        infer_response = next(infer_responses)
    else:
        infer_response = await infer_request.async_exec()

    return input0_np, infer_response


async def test_bls_out_of_memory():
    is_decoupled = True if os.environ["BLS_KIND"] == "decoupled" else False

    tensor_size = 256 * 1024 * 1024
    input0_np, infer_response = await _send_identity_tensor(tensor_size, is_decoupled)

    out_of_memory_message = "Failed to increase the shared memory pool size for key"

    if infer_response.has_error():
        if not (out_of_memory_message in infer_response.error().message()):
            return False
    else:
        output0 = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT0")
        if output0 is None:
            return False
        if not np.allclose(output0.as_numpy(), input0_np):
            return False

    tensor_size = 50 * 1024 * 1024
    for _ in range(4):
        input0_np, infer_response = await _send_identity_tensor(
            tensor_size, is_decoupled
        )

        if infer_response.has_error():
            if not (out_of_memory_message in infer_response.error().message()):
                return False
        else:
            output0 = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT0")
            if output0 is None:
                return False
            if not np.allclose(output0.as_numpy(), input0_np):
                return False

    return True


class TritonPythonModel:
    async def execute(self, requests):
        responses = []
        for _ in requests:
            # Run the unittest and store the results in InferenceResponse.
            result = await test_bls_out_of_memory()
            responses.append(
                pb_utils.InferenceResponse(
                    [pb_utils.Tensor("OUTPUT0", np.array([result], dtype=np.float16))]
                )
            )
        return responses
