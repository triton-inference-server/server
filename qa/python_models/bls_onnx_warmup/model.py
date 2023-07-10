# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest

import numpy as np
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack


class PBBLSONNXWarmupTest(unittest.TestCase):
    def test_onnx_output_mem_type(self):
        input0_np = np.random.randn(*[16])
        input0_np = input0_np.astype(np.float32)
        input1_np = np.random.randn(*[16])
        input1_np = input1_np.astype(np.float32)
        input0 = pb_utils.Tensor("INPUT0", input0_np)
        input1 = pb_utils.Tensor("INPUT1", input1_np)
        infer_request = pb_utils.InferenceRequest(
            model_name="onnx_nobatch_float32_float32_float32",
            inputs=[input0, input1],
            requested_output_names=["OUTPUT0", "OUTPUT1"],
        )

        infer_response = infer_request.exec()

        self.assertFalse(infer_response.has_error())

        output0 = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT0")
        output1 = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT1")

        self.assertIsNotNone(output0)
        self.assertIsNotNone(output1)

        # The memory type of output tensor should be GPU
        self.assertFalse(output0.is_cpu())
        self.assertFalse(output1.is_cpu())

        expected_output_0 = input0.as_numpy() - input1.as_numpy()
        expected_output_1 = input0.as_numpy() + input1.as_numpy()

        output0 = from_dlpack(output0.to_dlpack()).to("cpu").cpu().detach().numpy()
        output1 = from_dlpack(output1.to_dlpack()).to("cpu").cpu().detach().numpy()

        self.assertTrue(np.all(output0 == expected_output_0))
        self.assertTrue(np.all(output1 == expected_output_1))


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for _ in requests:
            # Run the unittest and store the results in InferenceResponse.
            test = unittest.main("model", exit=False)
            responses.append(
                pb_utils.InferenceResponse(
                    [
                        pb_utils.Tensor(
                            "OUTPUT0",
                            np.array([test.result.wasSuccessful()], dtype=np.float16),
                        )
                    ]
                )
            )
        return responses
