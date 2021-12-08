# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import unittest
import triton_python_backend_utils as pb_utils


class PBBLSMemoryTest(unittest.TestCase):

    def _send_identity_tensor(self, size):
        tensor_size = [1, size]
        input0_np = np.random.randn(*tensor_size)
        input0 = pb_utils.Tensor('INPUT0', input0_np.astype(np.float32))
        infer_request = pb_utils.InferenceRequest(
            model_name='identity_fp32',
            inputs=[input0],
            requested_output_names=['OUTPUT0'])
        return input0_np, infer_request.exec()

    def test_bls_out_of_memory(self):
        tensor_size = 1024 * 1024 * 1024
        input0_np, infer_response = self._send_identity_tensor(tensor_size)
        out_of_memory_message = "Failed to increase the shared memory pool size for key"

        if infer_response.has_error():
            self.assertIn(out_of_memory_message,
                          infer_response.error().message())
        else:
            self.assertFalse(infer_response.has_error())
            output0 = pb_utils.get_output_tensor_by_name(
                infer_response, 'OUTPUT0')
            self.assertIsNotNone(output0)
            self.assertTrue(np.allclose(output0.as_numpy(), input0_np))

        tensor_size = 50 * 1024 * 1024
        for _ in range(4):
            input0_np, infer_response = self._send_identity_tensor(tensor_size)
            if infer_response.has_error():
                self.assertIn(out_of_memory_message,
                              infer_response.error().message())
            else:
                self.assertFalse(infer_response.has_error())
                output0 = pb_utils.get_output_tensor_by_name(
                    infer_response, 'OUTPUT0')
                self.assertIsNotNone(output0)
                self.assertTrue(np.allclose(output0.as_numpy(), input0_np))


class TritonPythonModel:

    def execute(self, requests):
        responses = []
        for _ in requests:
            # Run the unittest and store the results in InferenceResponse.
            test = unittest.main('model', exit=False)
            responses.append(
                pb_utils.InferenceResponse([
                    pb_utils.Tensor(
                        'OUTPUT0',
                        np.array([test.result.wasSuccessful()],
                                 dtype=np.float16))
                ]))
        return responses
