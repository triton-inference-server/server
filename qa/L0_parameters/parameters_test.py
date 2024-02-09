#!/usr/bin/env python3

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

import sys

sys.path.append("../common")

import os
import queue
import unittest
from functools import partial
from unittest import IsolatedAsyncioTestCase

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.grpc.aio as asyncgrpcclient
import tritonclient.http as httpclient
import tritonclient.http.aio as asynchttpclient
from tritonclient.utils import InferenceServerException

TEST_HEADER = os.environ.get("TEST_HEADER")


class InferenceParametersTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.http = httpclient.InferenceServerClient(url="localhost:8000")
        self.async_http = asynchttpclient.InferenceServerClient(url="localhost:8000")
        self.grpc = grpcclient.InferenceServerClient(url="localhost:8001")
        self.async_grpc = asyncgrpcclient.InferenceServerClient(url="localhost:8001")

        self.parameter_list = []
        self.parameter_list.append({"key1": "value1", "key2": "value2"})
        self.parameter_list.append({"key1": 1, "key2": 2})
        self.parameter_list.append({"key1": 123.123, "key2": 321.321})
        self.parameter_list.append({"key1": True, "key2": "value2"})
        self.parameter_list.append({"triton_": True, "key2": "value2"})

        if TEST_HEADER == "1":
            self.headers = {
                "header_1": "value_1",
                "header_2": "value_2",
                "my_header_1": "my_value_1",
                "my_header_2": "my_value_2",
                "my_header_3": 'This is a "quoted" string with a backslash\ ',
            }

            # only these headers should be forwarded to the model.
            self.expected_headers = {
                "my_header_1": "my_value_1",
                "my_header_2": "my_value_2",
                "my_header_3": 'This is a "quoted" string with a backslash\ ',
            }
        else:
            self.headers = {}
            self.expected_headers = {}

        def callback(user_data, result, error):
            if error:
                user_data.put(error)
            else:
                user_data.put(result)

        self.grpc_callback = callback

    def create_inputs(self, client_type):
        inputs = []
        inputs.append(client_type.InferInput("INPUT0", [1], "FP32"))

        # Initialize the data
        inputs[0].set_data_from_numpy(np.asarray([1], dtype=np.float32))
        return inputs

    async def send_request_and_verify(
        self, client_type, client, is_async=False, model_name="parameter"
    ):
        inputs = self.create_inputs(client_type)
        for parameters in self.parameter_list:
            # Setup infer callable to re-use below for brevity
            infer_callable = partial(
                client.infer,
                model_name=model_name,
                inputs=inputs,
                parameters=parameters,
                headers=self.headers,
            )

            # The `triton_` prefix is reserved for Triton usage
            should_error = False
            if "triton_" in parameters.keys():
                should_error = True

            if is_async:
                if should_error:
                    with self.assertRaises(InferenceServerException):
                        await infer_callable()
                    return
                else:
                    result = await infer_callable()
            else:
                if should_error:
                    with self.assertRaises(InferenceServerException):
                        infer_callable()
                    return
                else:
                    result = infer_callable()

            self.verify_outputs(result, parameters)

    def verify_outputs(self, result, parameters):
        keys = result.as_numpy("key")
        values = result.as_numpy("value")
        keys = keys.astype(str).tolist()
        expected_keys = list(parameters.keys()) + list(self.expected_headers.keys())
        self.assertEqual(set(keys), set(expected_keys))

        # We have to convert the parameter values to string
        expected_values = []
        for expected_value in list(parameters.values()):
            expected_values.append(str(expected_value))
        for value in self.expected_headers.values():
            expected_values.append(value)
        self.assertEqual(set(values.astype(str).tolist()), set(expected_values))

    async def test_grpc_parameter(self):
        await self.send_request_and_verify(grpcclient, self.grpc)

    async def test_http_parameter(self):
        await self.send_request_and_verify(httpclient, self.http)

    async def test_async_http_parameter(self):
        await self.send_request_and_verify(
            asynchttpclient, self.async_http, is_async=True
        )

    async def test_async_grpc_parameter(self):
        await self.send_request_and_verify(
            asyncgrpcclient, self.async_grpc, is_async=True
        )

    def test_http_async_parameter(self):
        inputs = self.create_inputs(httpclient)
        # Skip the parameter that returns an error
        parameter_list = self.parameter_list[:-1]
        for parameters in parameter_list:
            result = self.http.async_infer(
                model_name="parameter",
                inputs=inputs,
                parameters=parameters,
                headers=self.headers,
            ).get_result()
            self.verify_outputs(result, parameters)

    def test_grpc_async_parameter(self):
        user_data = queue.Queue()
        inputs = self.create_inputs(grpcclient)
        # Skip the parameter that returns an error
        parameter_list = self.parameter_list[:-1]
        for parameters in parameter_list:
            self.grpc.async_infer(
                model_name="parameter",
                inputs=inputs,
                parameters=parameters,
                headers=self.headers,
                callback=partial(self.grpc_callback, user_data),
            )
            result = user_data.get()
            self.assertFalse(result is InferenceServerException)
            self.verify_outputs(result, parameters)

    def test_grpc_stream_parameter(self):
        user_data = queue.Queue()
        self.grpc.start_stream(
            callback=partial(self.grpc_callback, user_data), headers=self.headers
        )
        inputs = self.create_inputs(grpcclient)
        # Skip the parameter that returns an error
        parameter_list = self.parameter_list[:-1]
        for parameters in parameter_list:
            # async stream infer
            self.grpc.async_stream_infer(
                model_name="parameter", inputs=inputs, parameters=parameters
            )
            result = user_data.get()
            self.assertFalse(result is InferenceServerException)
            self.verify_outputs(result, parameters)
        self.grpc.stop_stream()

    async def test_ensemble_parameter_forwarding(self):
        await self.send_request_and_verify(httpclient, self.http, model_name="ensemble")

    async def asyncTearDown(self):
        self.http.close()
        self.grpc.close()
        await self.async_grpc.close()
        await self.async_http.close()


if __name__ == "__main__":
    unittest.main()
