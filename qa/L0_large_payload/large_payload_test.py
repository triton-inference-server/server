# Copyright 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
import unittest
import numpy as np
import test_util as tu
import tritongrpcclient as grpcclient
import tritonhttpclient as httpclient
from tritonclientutils import np_to_triton_dtype, InferenceServerException


class LargePayLoadTest(tu.TestResultCollector):

    def setUp(self):
        self._data_type = np.float32

        # Very large tensor will always fail for gRPC because the Protobuf has a
        # hard limit on 2GBs for the size of input tensors. All backends except
        # plan backend should be able to handle payloads larger than 2GBs using
        # HTTP.
        very_large_tensor_shape = (math.trunc(
            3 * (1024 * 1024 * 1024) / np.dtype(self._data_type).itemsize),)
        self._very_large_in0 = np.random.random(very_large_tensor_shape).astype(
            self._data_type)

        # 1.9 GBs allows us to test gRPC with moderate sizes too.
        large_tensor_shape = (math.trunc(1.9 * (1024 * 1024 * 1024) //
                                         np.dtype(self._data_type).itemsize),)
        self._large_in0 = np.random.random(large_tensor_shape).astype(
            self._data_type)

        small_tensor_shape = (1,)
        self._small_in0 = np.random.random(small_tensor_shape).astype(
            self._data_type)

        self._clients = ((httpclient,
                          httpclient.InferenceServerClient('localhost:8000')),
                         (grpcclient,
                          grpcclient.InferenceServerClient('localhost:8001')))

    def _test_helper(self,
                     client,
                     model_name,
                     input_name='INPUT0',
                     output_name='OUTPUT0'):
        # plan does not supoort large batch sizes.
        if not model_name.startswith('plan'):
            inputs = [
                client[0].InferInput(input_name, self._large_in0.shape,
                                     np_to_triton_dtype(self._data_type))
            ]
            inputs[0].set_data_from_numpy(self._large_in0)
            results = client[1].infer(model_name, inputs)

            # if the inference is completed, examine results to ensure that
            # the framework and protocol do support large payload
            self.assertTrue(
                np.array_equal(self._large_in0, results.as_numpy(output_name)),
                "output is different from input")

        if client[0] == httpclient:
            # FIXME HTTPServer cannot support large payloads. See DLIS-1776.
            inputs = [
                client[0].InferInput(input_name, self._very_large_in0.shape,
                                     np_to_triton_dtype(self._data_type))
            ]
            inputs[0].set_data_from_numpy(self._very_large_in0)
            with self.assertRaises(InferenceServerException):
                results = client[1].infer(model_name, inputs)

        # FIXME Test is terminated due to libprotobuf FATAL error when GRPC sends
        # the second request with input tensors larger than 1.3GBs. In this test
        # GRPC has been currently exempted from testing for Very Large tensor(3GBs)
        # until the problem is resolved. Should be uncommented once the GRPC issue is resolved.
        # See DLIS-2474.
        # if client[0] == grpcclient:
        #     inputs = [
        #         client[0].InferInput(input_name, self._very_large_in0.shape,
        #                              np_to_triton_dtype(self._data_type))
        #     ]
        #     inputs[0].set_data_from_numpy(self._very_large_in0)
        #     # GRPC must fail for large payloads because of a 2GB protobuf limit
        #     with self.assertRaises(InferenceServerException):
        #         results = client[1].infer(model_name, inputs)

        # Send a small payload to verify if the server is still functional
        inputs = [
            client[0].InferInput(input_name, self._small_in0.shape,
                                 np_to_triton_dtype(self._data_type))
        ]
        inputs[0].set_data_from_numpy(self._small_in0)
        results = client[1].infer(model_name, inputs)
        self.assertTrue(
            np.array_equal(self._small_in0, results.as_numpy(output_name)),
            "output is different from input")

    def test_graphdef(self):
        # graphdef_nobatch_zero_1_float32 is identity model with input shape [-1]
        for client in self._clients:
            model_name = tu.get_zero_model_name("graphdef_nobatch", 1,
                                                self._data_type)
            self._test_helper(client, model_name)

    def test_savedmodel(self):
        # savedmodel_nobatch_zero_1_float32 is identity model with input shape [-1]
        for client in self._clients:
            model_name = tu.get_zero_model_name("savedmodel_nobatch", 1,
                                                self._data_type)
            self._test_helper(client, model_name)

    def test_onnx(self):
        # onnx_nobatch_zero_1_float32 is identity model with input shape [-1]
        for client in self._clients:
            model_name = tu.get_zero_model_name("onnx_nobatch", 1,
                                                self._data_type)
            self._test_helper(client, model_name)

    def test_python(self):
        # python_nobatch_zero_1_float32 is identity model with input shape [-1]
        for client in self._clients:
            model_name = tu.get_zero_model_name("python_nobatch", 1,
                                                self._data_type)
            self._test_helper(client, model_name)

    def test_plan(self):
        # plan_nobatch_zero_1_float32 is identity model with input shape [-1]
        for client in self._clients:
            model_name = tu.get_zero_model_name("plan_nobatch", 1,
                                                self._data_type)
            self._test_helper(client, model_name)

    def test_libtorch(self):
        # libtorch_nobatch_zero_1_float32 is identity model with input shape [-1]
        for client in self._clients:
            model_name = tu.get_zero_model_name("libtorch_nobatch", 1,
                                                self._data_type)
            self._test_helper(client, model_name, 'INPUT__0', 'OUTPUT__0')

    def test_custom(self):
        # custom_zero_1_float32 is identity model with input shape [-1]
        for client in self._clients:
            model_name = tu.get_zero_model_name("custom", 1, self._data_type)
            self._test_helper(client, model_name)


if __name__ == '__main__':
    unittest.main()
