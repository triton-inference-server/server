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


class PBBLSTest(unittest.TestCase):
    def _bls_add_sub(self):
        input0_np = np.random.randn(*[16])
        input0_np = input0_np.astype(np.float32)
        input1_np = np.random.randn(*[16])
        input1_np = input1_np.astype(np.float32)
        input0 = pb_utils.Tensor('INPUT0', input0_np)
        input1 = pb_utils.Tensor('INPUT1', input1_np)
        infer_request = pb_utils.InferenceRequest(
            model_name='add_sub',
            inputs=[input0, input1],
            requested_output_names=['OUTPUT0', 'OUTPUT1'])
        infer_response = infer_request.exec()
        self.assertFalse(infer_response.has_error())
        output0 = pb_utils.get_output_tensor_by_name(infer_response, 'OUTPUT0')
        output1 = pb_utils.get_output_tensor_by_name(infer_response, 'OUTPUT1')
        self.assertIsNotNone(output0)
        self.assertIsNotNone(output1)

        expected_output_0 = input0.as_numpy() + input1.as_numpy()
        expected_output_1 = input0.as_numpy() - input1.as_numpy()
        self.assertTrue(np.all(expected_output_0 == output0.as_numpy()))
        self.assertTrue(np.all(expected_output_1 == output1.as_numpy()))

    def test_bls_wrong_inputs(self):
        input0 = pb_utils.Tensor('INPUT0', np.random.randn(*[1, 16]))

        infer_request = pb_utils.InferenceRequest(
            model_name='add_sub',
            inputs=[input0],
            requested_output_names=['OUTPUT0', 'OUTPUT1'])
        infer_response = infer_request.exec()
        self.assertTrue(infer_response.has_error())

    def test_bls_incorrect_args(self):
        with self.assertRaises(TypeError):
            pb_utils.InferenceRequest(
                inputs=[], requested_output_names=['OUTPUT0', 'OUTPUT1'])

        with self.assertRaises(TypeError):
            pb_utils.InferenceRequest(
                model_name='add_sub',
                requested_output_names=['OUTPUT0', 'OUTPUT1'])

        with self.assertRaises(TypeError):
            pb_utils.InferenceRequest(model_name='add_sub', inputs=[])

    def test_bls_sync(self):
        infer_request = pb_utils.InferenceRequest(
            model_name='non_existent_model',
            inputs=[],
            requested_output_names=[])
        infer_response = infer_request.exec()

        # Because the model doesn't exist, the inference response must have an
        # error
        self.assertTrue(infer_response.has_error())

        # Make sure that the inference requests can be performed properly after
        # an error.
        self._bls_add_sub()

    def test_bls_execute_error(self):
        # Test BLS with a model that has an error during execution.
        infer_request = pb_utils.InferenceRequest(model_name='execute_error',
                                                  inputs=[],
                                                  requested_output_names=[])
        infer_response = infer_request.exec()
        self.assertTrue(infer_response.has_error())

    def test_multiple_bls(self):
        # Test running multiple BLS requests together
        for _ in range(100):
            self._bls_add_sub()


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
