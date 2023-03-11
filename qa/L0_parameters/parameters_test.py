# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

import numpy as np
import infer_util as iu
import test_util as tu
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import tritonclient.http.aio as asynchttpclient
import tritonclient.grpc.aio as asyncgrpcclient
import asyncio
import json


class InferenceParametersTest(tu.TestResultCollector):

    def setUp(self):
        self.http = httpclient.InferenceServerClient(url='localhost:8000')
        self.async_http = asynchttpclient.InferenceServerClient(
            url='localhost:8000')
        self.grpc = grpcclient.InferenceServerClient(url='localhost:8001')
        self.async_grpc = asyncgrpcclient.InferenceServerClient(
            url='localhost:8001')

    def send_request_and_verify(self, client, is_async=False):
        inputs = []
        inputs.append(client.InferInput('INPUT0', [1], "FP32"))

        # Initialize the data
        inputs[0].set_data_from_numpy(np.asarray([1], dtype=np.float))
        parameters = {'key1': 'value1', 'key2': 'value2'}
        if is_async:
            result = asyncio.run(
                client.infer(model_name='parameter',
                             inputs=inputs,
                             parameters=parameters))
        else:
            result = client.infer(model_name='parameter',
                                  inputs=inputs,
                                  parameters=parameters)

        result = result.as_numpy('OUTPUT0')
        self.assertEqual(json.loads(result[0]) == parameters)

    def test_grpc_parameter(self):
        self.send_request_and_verify(self.grpc)

    def test_http_parameter(self):
        self.send_request_and_verify(self.http)

    def test_async_http_parameter(self):
        self.send_request_and_verify(self.http, is_async=True)

    def test_async_grpc_parameter(self):
        self.send_request_and_verify(self.grpc, is_async=True)
