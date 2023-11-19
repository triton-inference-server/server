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

import time
import unittest

import numpy as np
import triton_python_backend_utils as pb_utils


class RequestReschedulingTest(unittest.TestCase):
    def _reload_model(self, model_name):
        # Reload the model to reset the flag for multiple iterations
        pb_utils.unload_model(model_name)
        # TODO: Make this more robust to wait until fully unloaded
        print("Sleep 10 seconds to make sure model finishes unloading...", flush=True)
        time.sleep(10)
        print("Done sleeping.", flush=True)
        pb_utils.load_model(model_name)

    def test_wrong_return_type(self):
        input0 = pb_utils.Tensor("INPUT0", (np.random.randn(*[4])).astype(np.float32))
        infer_request = pb_utils.InferenceRequest(
            model_name="wrong_return_type",
            inputs=[input0],
            requested_output_names=["OUTPUT0"],
        )

        infer_response = infer_request.exec()
        self.assertTrue(infer_response.has_error())
        self.assertIn(
            "Expected a None object in the execute function return list for reschduled request",
            infer_response.error().message(),
        )

    def test_non_decoupled_e2e(self):
        model_name = "request_rescheduling_addsub"
        self._reload_model(model_name)

        input0_np = np.random.randn(*[16])
        input0_np = input0_np.astype(np.float32)
        input1_np = np.random.randn(*[16])
        input1_np = input1_np.astype(np.float32)
        input0 = pb_utils.Tensor("INPUT0", input0_np)
        input1 = pb_utils.Tensor("INPUT1", input1_np)
        infer_request = pb_utils.InferenceRequest(
            model_name=model_name,
            inputs=[input0, input1],
            requested_output_names=["OUTPUT0", "OUTPUT1"],
        )
        infer_response = infer_request.exec()

        self.assertFalse(infer_response.has_error())

        output0 = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT0")
        output1 = pb_utils.get_output_tensor_by_name(infer_response, "OUTPUT1")

        self.assertIsNotNone(output0)
        self.assertIsNotNone(output1)

        expected_output_0 = input0.as_numpy() + input1.as_numpy()
        expected_output_1 = input0.as_numpy() - input1.as_numpy()

        self.assertEqual(expected_output_0[0], output0.as_numpy()[0])
        self.assertEqual(expected_output_1[0], output1.as_numpy()[0])

    def test_decoupled_e2e(self):
        model_name = "iterative_sequence"
        self._reload_model(model_name)

        input_value = 3
        input0 = pb_utils.Tensor("IN", np.array([input_value], dtype=np.int32))
        infer_request = pb_utils.InferenceRequest(
            model_name=model_name,
            inputs=[input0],
            requested_output_names=["OUT"],
        )
        infer_responses = infer_request.exec(decoupled=True)

        expected_output = input_value - 1

        if infer_responses:
            for infer_response in infer_responses:
                self.assertFalse(infer_response.has_error())

                if len(infer_response.output_tensors()) > 0:
                    output0 = pb_utils.get_output_tensor_by_name(infer_response, "OUT")
                    self.assertIsNotNone(output0)

                    self.assertEqual(expected_output, output0.as_numpy()[0])
                    expected_output -= 1


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
