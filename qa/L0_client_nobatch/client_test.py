# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

from builtins import range
from future.utils import iteritems
import unittest
import numpy as np
import tritonhttpclient
import tritongrpcclient
from tritonclientutils import InferenceServerException
import test_util as tu


class ClientNoBatchTest(tu.TestResultCollector):

    def test_nobatch_request_for_batching_model(self):
        input_size = 16

        # graphdef_int32_int8_int8 has a batching version with max batch size of 8.
        # The server should return an error if the batch size is not included in the
        # input shapes.
        tensor_shape = (input_size,)
        for protocol in ["http", "grpc"]:
            model_name = tu.get_model_name("graphdef", np.int32, np.int8,
                                           np.int8)
            in0 = np.random.randint(low=0,
                                    high=100,
                                    size=tensor_shape,
                                    dtype=np.int32)
            in1 = np.random.randint(low=0,
                                    high=100,
                                    size=tensor_shape,
                                    dtype=np.int32)

            inputs = []
            outputs = []
            if protocol == "http":
                triton_client = tritonhttpclient.InferenceServerClient(
                    url='localhost:8000', verbose=True)
                inputs.append(
                    tritonhttpclient.InferInput('INPUT0', tensor_shape,
                                                "INT32"))
                inputs.append(
                    tritonhttpclient.InferInput('INPUT1', tensor_shape,
                                                "INT32"))
                outputs.append(tritonhttpclient.InferRequestedOutput('OUTPUT0'))
                outputs.append(tritonhttpclient.InferRequestedOutput('OUTPUT1'))
            else:
                triton_client = tritongrpcclient.InferenceServerClient(
                    url='localhost:8001', verbose=True)
                inputs.append(
                    tritongrpcclient.InferInput('INPUT0', tensor_shape,
                                                "INT32"))
                inputs.append(
                    tritongrpcclient.InferInput('INPUT1', tensor_shape,
                                                "INT32"))
                outputs.append(tritongrpcclient.InferRequestedOutput('OUTPUT0'))
                outputs.append(tritongrpcclient.InferRequestedOutput('OUTPUT1'))

            # Initialize the data
            inputs[0].set_data_from_numpy(in0)
            inputs[1].set_data_from_numpy(in1)

            try:
                results = triton_client.infer(model_name,
                                              inputs,
                                              outputs=outputs)
                self.assertTrue(
                    False,
                    "expected failure with no batch request for batching model")
            except InferenceServerException as ex:
                pass

    def test_batch_request_for_nobatching_model(self):
        input_size = 16

        # graphdef_nobatch_int32_int8_int8 is non batching version.
        # The server should return an error if the batch size dimension
        # is included in the shape
        tensor_shape = (1, input_size)
        for protocol in ["http", "grpc"]:
            model_name = tu.get_model_name("graphdef_nobatch", np.int32,
                                           np.int8, np.int8)
            in0 = np.random.randint(low=0,
                                    high=100,
                                    size=tensor_shape,
                                    dtype=np.int32)
            in1 = np.random.randint(low=0,
                                    high=100,
                                    size=tensor_shape,
                                    dtype=np.int32)

            inputs = []
            outputs = []
            if protocol == "http":
                triton_client = tritonhttpclient.InferenceServerClient(
                    url='localhost:8000', verbose=True)
                inputs.append(
                    tritonhttpclient.InferInput('INPUT0', tensor_shape,
                                                "INT32"))
                inputs.append(
                    tritonhttpclient.InferInput('INPUT1', tensor_shape,
                                                "INT32"))
                outputs.append(tritonhttpclient.InferRequestedOutput('OUTPUT0'))
                outputs.append(tritonhttpclient.InferRequestedOutput('OUTPUT1'))
            else:
                triton_client = tritongrpcclient.InferenceServerClient(
                    url='localhost:8001', verbose=True)
                inputs.append(
                    tritongrpcclient.InferInput('INPUT0', tensor_shape,
                                                "INT32"))
                inputs.append(
                    tritongrpcclient.InferInput('INPUT1', tensor_shape,
                                                "INT32"))
                outputs.append(tritongrpcclient.InferRequestedOutput('OUTPUT0'))
                outputs.append(tritongrpcclient.InferRequestedOutput('OUTPUT1'))

            # Initialize the data
            inputs[0].set_data_from_numpy(in0)
            inputs[1].set_data_from_numpy(in1)

            try:
                results = triton_client.infer(model_name,
                                              inputs,
                                              outputs=outputs)
                self.assertTrue(
                    False,
                    "expected failure with batched request for non-batching model"
                )
            except InferenceServerException as ex:
                pass

    def test_nobatch_request_for_nonbatching_model(self):
        input_size = 16

        # graphdef_int32_int8_int8 has a batching version with max batch size of 8.
        # The server should return an error if the batch size is not included in the
        # input shapes.
        tensor_shape = (input_size,)
        for protocol in ["http", "grpc"]:
            model_name = tu.get_model_name("graphdef_nobatch", np.int32,
                                           np.int8, np.int8)
            in0 = np.random.randint(low=0,
                                    high=100,
                                    size=tensor_shape,
                                    dtype=np.int32)
            in1 = np.random.randint(low=0,
                                    high=100,
                                    size=tensor_shape,
                                    dtype=np.int32)

            inputs = []
            outputs = []
            if protocol == "http":
                triton_client = tritonhttpclient.InferenceServerClient(
                    url='localhost:8000', verbose=True)
                inputs.append(
                    tritonhttpclient.InferInput('INPUT0', tensor_shape,
                                                "INT32"))
                inputs.append(
                    tritonhttpclient.InferInput('INPUT1', tensor_shape,
                                                "INT32"))
                outputs.append(tritonhttpclient.InferRequestedOutput('OUTPUT0'))
                outputs.append(tritonhttpclient.InferRequestedOutput('OUTPUT1'))
            else:
                triton_client = tritongrpcclient.InferenceServerClient(
                    url='localhost:8001', verbose=True)
                inputs.append(
                    tritongrpcclient.InferInput('INPUT0', tensor_shape,
                                                "INT32"))
                inputs.append(
                    tritongrpcclient.InferInput('INPUT1', tensor_shape,
                                                "INT32"))
                outputs.append(tritongrpcclient.InferRequestedOutput('OUTPUT0'))
                outputs.append(tritongrpcclient.InferRequestedOutput('OUTPUT1'))

            # Initialize the data
            inputs[0].set_data_from_numpy(in0)
            inputs[1].set_data_from_numpy(in1)

            results = triton_client.infer(model_name, inputs, outputs=outputs)

    def test_batch_request_for_batching_model(self):
        input_size = 16

        # graphdef_nobatch_int32_int8_int8 is non batching version.
        # The server should return an error if the batch size dimension
        # is included in the shape
        tensor_shape = (1, input_size)
        for protocol in ["http", "grpc"]:
            model_name = tu.get_model_name("graphdef", np.int32, np.int8,
                                           np.int8)
            in0 = np.random.randint(low=0,
                                    high=100,
                                    size=tensor_shape,
                                    dtype=np.int32)
            in1 = np.random.randint(low=0,
                                    high=100,
                                    size=tensor_shape,
                                    dtype=np.int32)

            inputs = []
            outputs = []
            if protocol == "http":
                triton_client = tritonhttpclient.InferenceServerClient(
                    url='localhost:8000', verbose=True)
                inputs.append(
                    tritonhttpclient.InferInput('INPUT0', tensor_shape,
                                                "INT32"))
                inputs.append(
                    tritonhttpclient.InferInput('INPUT1', tensor_shape,
                                                "INT32"))
                outputs.append(tritonhttpclient.InferRequestedOutput('OUTPUT0'))
                outputs.append(tritonhttpclient.InferRequestedOutput('OUTPUT1'))
            else:
                triton_client = tritongrpcclient.InferenceServerClient(
                    url='localhost:8001', verbose=True)
                inputs.append(
                    tritongrpcclient.InferInput('INPUT0', tensor_shape,
                                                "INT32"))
                inputs.append(
                    tritongrpcclient.InferInput('INPUT1', tensor_shape,
                                                "INT32"))
                outputs.append(tritongrpcclient.InferRequestedOutput('OUTPUT0'))
                outputs.append(tritongrpcclient.InferRequestedOutput('OUTPUT1'))

            # Initialize the data
            inputs[0].set_data_from_numpy(in0)
            inputs[1].set_data_from_numpy(in1)

            results = triton_client.infer(model_name, inputs, outputs=outputs)


if __name__ == '__main__':
    unittest.main()
