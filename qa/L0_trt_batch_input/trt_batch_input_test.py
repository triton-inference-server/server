# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import sys
sys.path.append("../common")

import unittest
import numpy as np
import infer_util as iu
import test_util as tu
import tritonhttpclient
from tritonclientutils import InferenceServerException

class TRTBatchInputTest(tu.TestResultCollector):
    def setUp(self):
        self.dtype_ = np.float32
        self.model_name_ = 'plan'
        self.inputs = []
        # 4 set of inputs with shape [2], [4], [1], [3]
        for value in [2, 4, 1, 3]:
            self.inputs.append([tritonhttpclient.InferInput('INPUT0', [1, value], "FP32")])
            self.inputs[-1][0].set_data_from_numpy(np.full([1, value], value, np.float32))
        self.client = tritonhttpclient.InferenceServerClient(
                url="localhost:8000", concurrency=len(self.inputs))

    def test_ragged_input(self):
        model_name = "ragged_acc_zero_shape"

        output_name = 'OUTPUT0'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]

        async_requests = []
        try:
            for inputs in self.inputs:
                # Asynchronous inference call.
                async_requests.append(
                    self.client.async_infer(model_name=model_name,
                                            inputs=inputs,
                                            outputs=outputs))
            
            expected_values = [np.full([1, v], v, np.float32) for v in [2, 4, 1, 3]]
            for idx in range(len(async_requests)):
                # Get the result from the initiated asynchronous inference request.
                # Note the call will block till the server responds.
                result = async_requests[idx].get_result()

                # Validate the results by comparing with precomputed values.
                output_data = result.as_numpy(output_name)
                self.assertTrue(np.array_equal(output_data, expected_values[idx]),
                    "Expect response {} to have value {}, got {}".format(idx, expected_values[idx], output_data))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_element_count(self):
        model_name = "ragged_element_count_acc"

        output_name = 'OUTPUT1'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]

        async_requests = []
        try:
            for inputs in self.inputs:
                # Asynchronous inference call.
                async_requests.append(
                    self.client.async_infer(model_name=model_name,
                                            inputs=inputs,
                                            outputs=outputs))
            
            expected_values = [np.full([1, 1], v, np.float32) for v in [2, 4, 1, 3]]
            for idx in range(len(async_requests)):
                # Get the result from the initiated asynchronous inference request.
                # Note the call will block till the server responds.
                result = async_requests[idx].get_result()

                # Validate the results by comparing with precomputed values.
                output_data = result.as_numpy(output_name)
                self.assertTrue(np.array_equal(output_data, expected_values[idx]),
                    "Expect response {} to have value {}, got {}".format(idx, expected_values[idx], output_data))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_accumulated_element_count(self):
        model_name = "ragged_element_count_acc"

        output_name = 'OUTPUT2'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]

        async_requests = []
        try:
            for inputs in self.inputs:
                # Asynchronous inference call.
                async_requests.append(
                    self.client.async_infer(model_name=model_name,
                                            inputs=inputs,
                                            outputs=outputs))
            
            expected_values = [np.full([1, 1], v, np.float32) for v in [2, 6, 7, 10]]
            for idx in range(len(async_requests)):
                # Get the result from the initiated asynchronous inference request.
                # Note the call will block till the server responds.
                result = async_requests[idx].get_result()

                # Validate the results by comparing with precomputed values.
                output_data = result.as_numpy(output_name)
                self.assertTrue(np.array_equal(output_data, expected_values[idx]),
                    "Expect response {} to have value {}, got {}".format(idx, expected_values[idx], output_data))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_accumulated_element_count_with_zero(self):
        model_name = "ragged_acc_zero_shape"

        output_name = 'OUTPUT1'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]

        async_requests = []
        try:
            for inputs in self.inputs:
                # Asynchronous inference call.
                async_requests.append(
                    self.client.async_infer(model_name=model_name,
                                            inputs=inputs,
                                            outputs=outputs))
            
            expected_values = [2, 6, 7, 10]
            for idx in range(len(async_requests)):
                # Get the result from the initiated asynchronous inference request.
                # Note the call will block till the server responds.
                result = async_requests[idx].get_result()

                # Validate the results by comparing with precomputed values.
                output_data = result.as_numpy(output_name)
                if idx == 0:
                    self.assertEqual(output_data.shape, (2, 1),
                        "Expect the first response to have shape to represent 0 and accumulated element count, got {}".format(output_data.shape))
                    self.assertEqual(output_data[0][0], 0,
                        "Expect the first response to have 0, got {}".format(output_data[0][0]))
                    self.assertEqual(output_data[1][0], expected_values[idx],
                        "Expect response {} to have {}, got {}".format(idx, expected_values[idx], output_data[1][0]))
                else:
                    self.assertEqual(output_data.shape, (1, 1),
                        "Expect the non-first response to have shape to represent accumulated element count, got {}".format(output_data.shape))
                    self.assertEqual(output_data[0][0], expected_values[idx],
                        "Expect response {} to have {}, got {}".format(idx, expected_values[idx], output_data[0][0]))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_max_element_count_as_shape(self):
        model_name = "ragged_acc_zero_shape"

        output_name = 'OUTPUT2'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]

        async_requests = []
        try:
            for inputs in self.inputs:
                # Asynchronous inference call.
                async_requests.append(
                    self.client.async_infer(model_name=model_name,
                                            inputs=inputs,
                                            outputs=outputs))
            
            for idx in range(len(async_requests)):
                # Get the result from the initiated asynchronous inference request.
                # Note the call will block till the server responds.
                result = async_requests[idx].get_result()

                # Validate the results by comparing with precomputed values.
                output_data = result.as_numpy(output_name)
                if idx == 0:
                    self.assertEqual(output_data.shape, (4, 1),
                        "Expect response {} to have shape to represent max element count {} among the batch , got {}".format(idx, 4, output_data.shape))
                else:
                    self.assertEqual(output_data.shape, (0, 1),
                        "Expect response {} to have 0 shape, got {}".format(idx, output_data.shape))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))


if __name__ == '__main__':
    unittest.main()
