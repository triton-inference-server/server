# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class BatchInputTest(tu.TestResultCollector):

    def setUp(self):
        self.dtype_ = np.float32
        self.inputs = []
        # 4 set of inputs with shape [2], [4], [1], [3]
        for value in [2, 4, 1, 3]:
            self.inputs.append([
                tritonhttpclient.InferInput('RAGGED_INPUT', [1, value], "FP32")
            ])
            self.inputs[-1][0].set_data_from_numpy(
                np.full([1, value], value, np.float32))
        self.client = tritonhttpclient.InferenceServerClient(
            url="localhost:8000", concurrency=len(self.inputs))

    def test_ragged_output(self):
        model_name = "ragged_io"

        # The model is identity model
        self.inputs = []
        for value in [2, 4, 1, 3]:
            self.inputs.append(
                [tritonhttpclient.InferInput('INPUT0', [1, value], "FP32")])
            self.inputs[-1][0].set_data_from_numpy(
                np.full([1, value], value, np.float32))
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

            expected_value_list = [[v] * v for v in [2, 4, 1, 3]]
            expected_value_list = [
                np.asarray([expected_value], dtype=np.float32)
                for expected_value in expected_value_list
            ]
            for idx in range(len(async_requests)):
                # Get the result from the initiated asynchronous inference request.
                # Note the call will block till the server responds.
                result = async_requests[idx].get_result()

                # Validate the results by comparing with precomputed values.
                output_data = result.as_numpy(output_name)
                self.assertTrue(
                    np.array_equal(output_data, expected_value_list[idx]),
                    "Expect response {} to have value {}, got {}".format(
                        idx, expected_value_list[idx], output_data))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_ragged_input(self):
        model_name = "ragged_acc_shape"

        output_name = 'RAGGED_OUTPUT'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]

        async_requests = []
        try:
            for inputs in self.inputs:
                # Asynchronous inference call.
                async_requests.append(
                    self.client.async_infer(model_name=model_name,
                                            inputs=inputs,
                                            outputs=outputs))

            value_lists = [[v] * v for v in [2, 4, 1, 3]]
            expected_value = []
            for value_list in value_lists:
                expected_value += value_list
            expected_value = np.asarray([expected_value], dtype=np.float32)
            for idx in range(len(async_requests)):
                # Get the result from the initiated asynchronous inference request.
                # Note the call will block till the server responds.
                result = async_requests[idx].get_result()

                # Validate the results by comparing with precomputed values.
                output_data = result.as_numpy(output_name)
                self.assertTrue(
                    np.array_equal(output_data, expected_value),
                    "Expect response {} to have value {}, got {}".format(
                        idx, expected_value, output_data))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_element_count(self):
        model_name = "ragged_element_count_acc_zero"

        output_name = 'BATCH_AND_SIZE_OUTPUT'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]

        async_requests = []
        try:
            for inputs in self.inputs:
                # Asynchronous inference call.
                async_requests.append(
                    self.client.async_infer(model_name=model_name,
                                            inputs=inputs,
                                            outputs=outputs))

            expected_value = np.asarray([[2, 4, 1, 3]], np.float32)
            for idx in range(len(async_requests)):
                # Get the result from the initiated asynchronous inference request.
                # Note the call will block till the server responds.
                result = async_requests[idx].get_result()

                # Validate the results by comparing with precomputed values.
                output_data = result.as_numpy(output_name)
                self.assertTrue(
                    np.array_equal(output_data, expected_value),
                    "Expect response {} to have value {}, got {}".format(
                        idx, expected_value, output_data))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_accumulated_element_count(self):
        model_name = "ragged_acc_shape"

        output_name = 'BATCH_AND_SIZE_OUTPUT'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]

        async_requests = []
        try:
            for inputs in self.inputs:
                # Asynchronous inference call.
                async_requests.append(
                    self.client.async_infer(model_name=model_name,
                                            inputs=inputs,
                                            outputs=outputs))

            expected_value = np.asarray([[2, 6, 7, 10]], np.float32)
            for idx in range(len(async_requests)):
                # Get the result from the initiated asynchronous inference request.
                # Note the call will block till the server responds.
                result = async_requests[idx].get_result()

                # Validate the results by comparing with precomputed values.
                output_data = result.as_numpy(output_name)
                self.assertTrue(
                    np.array_equal(output_data, expected_value),
                    "Expect response {} to have value {}, got {}".format(
                        idx, expected_value, output_data))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_accumulated_element_count_with_zero(self):
        model_name = "ragged_element_count_acc_zero"

        output_name = 'BATCH_OUTPUT'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]

        async_requests = []
        try:
            for inputs in self.inputs:
                # Asynchronous inference call.
                async_requests.append(
                    self.client.async_infer(model_name=model_name,
                                            inputs=inputs,
                                            outputs=outputs))

            expected_value = np.asarray([[0, 2, 6, 7, 10]], np.float32)
            for idx in range(len(async_requests)):
                # Get the result from the initiated asynchronous inference request.
                # Note the call will block till the server responds.
                result = async_requests[idx].get_result()

                # Validate the results by comparing with precomputed values.
                output_data = result.as_numpy(output_name)
                self.assertTrue(
                    np.array_equal(output_data, expected_value),
                    "Expect response {} to have value {}, got {}".format(
                        idx, expected_value, output_data))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_max_element_count_as_shape(self):
        model_name = "ragged_acc_shape"

        output_name = 'BATCH_OUTPUT'
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
                self.assertEqual(
                    output_data.shape, (1, 4),
                    "Expect response {} to have shape to represent max element count {} among the batch , got {}"
                    .format(idx, 4, output_data.shape))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_batch_item_shape_flatten(self):
        # Use 4 set of inputs with shape
        # [1, 4, 1], [1, 1, 2], [1, 1, 2], [1, 2, 2]
        # Note that the test only checks the formation of "BATCH_INPUT" where
        # the value of "RAGGED_INPUT" is irrelevant, only the shape matters
        self.inputs = []
        for value in [[1, 4, 1], [1, 1, 2], [1, 1, 2], [1, 2, 2]]:
            self.inputs.append(
                [tritonhttpclient.InferInput('RAGGED_INPUT', value, "FP32")])
            self.inputs[-1][0].set_data_from_numpy(
                np.full(value, value[0], np.float32))
        self.client = tritonhttpclient.InferenceServerClient(
            url="localhost:8000", concurrency=len(self.inputs))

        model_name = "batch_item_flatten"

        output_name = 'BATCH_OUTPUT'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]

        async_requests = []
        try:
            for inputs in self.inputs:
                # Asynchronous inference call.
                async_requests.append(
                    self.client.async_infer(model_name=model_name,
                                            inputs=inputs,
                                            outputs=outputs))

            expected_value = np.asarray([[4, 1, 1, 2, 1, 2, 2, 2]], np.float32)
            for idx in range(len(async_requests)):
                # Get the result from the initiated asynchronous inference request.
                # Note the call will block till the server responds.
                result = async_requests[idx].get_result()

                # Validate the results by comparing with precomputed values.
                output_data = result.as_numpy(output_name)
                self.assertTrue(
                    np.array_equal(output_data, expected_value),
                    "Expect response {} to have value {}, got {}".format(
                        idx, expected_value, output_data))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_batch_item_shape(self):
        # Use 3 set of inputs with shape [2, 1, 2], [1, 1, 2], [1, 2, 2]
        # Note that the test only checks the formation of "BATCH_INPUT" where
        # the value of "RAGGED_INPUT" is irrelevant, only the shape matters
        inputs = []
        for value in [[2, 1, 2], [1, 1, 2], [1, 2, 2]]:
            inputs.append(
                [tritonhttpclient.InferInput('RAGGED_INPUT', value, "FP32")])
            inputs[-1][0].set_data_from_numpy(
                np.full(value, value[0], np.float32))
        client = tritonhttpclient.InferenceServerClient(url="localhost:8000",
                                                        concurrency=len(inputs))

        expected_outputs = [
            np.array([[1.0, 2.0], [1.0, 2.0]]),
            np.array([[1.0, 2.0]]),
            np.array([[2.0, 2.0]]),
        ]

        model_name = "batch_item"

        output_name = 'BATCH_OUTPUT'
        outputs = [tritonhttpclient.InferRequestedOutput(output_name)]

        async_requests = []
        try:
            for request_inputs in inputs:
                # Asynchronous inference call.
                async_requests.append(
                    client.async_infer(model_name=model_name,
                                       inputs=request_inputs,
                                       outputs=outputs))

            for idx in range(len(async_requests)):
                # Get the result from the initiated asynchronous inference request.
                # Note the call will block till the server responds.
                result = async_requests[idx].get_result()

                # Validate the results by comparing with precomputed values.
                output_data = result.as_numpy(output_name)
                self.assertTrue(
                    np.allclose(output_data, expected_outputs[idx]),
                    "Expect response to have value:\n{}, got:\n{}\nEqual matrix:\n{}"
                    .format(expected_outputs[idx], output_data,
                            np.isclose(expected_outputs[idx], output_data)))
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))


if __name__ == '__main__':
    unittest.main()
