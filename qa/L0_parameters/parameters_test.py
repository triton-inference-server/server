#!/usr/bin/env python3

# Copyright 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import queue
import unittest
from functools import partial

import numpy as np
import requests
import tritonclient.grpc as grpcclient
import tritonclient.grpc.aio as asyncgrpcclient
import tritonclient.http as httpclient
import tritonclient.http.aio as asynchttpclient
from tritonclient.utils import InferenceServerException

_TRITON_RESERVED_CLIENT_ERROR = "is a reserved parameter and cannot be specified"
_TRITON_RESERVED_SERVER_ERROR = "reserved for Triton usage"

# docs/protocol/extension_parameters.md — reserved names
_RESERVED_PARAMETER_KEYS = (
    "sequence_id",
    "sequence_start",
    "sequence_end",
    "priority",
    "timeout",
    "headers",
    "binary_data_output",
    "triton_enable_empty_final_response",
    "triton_final_response",
    "triton_injected_via_header",
)

_HTTP_LOCALHOST = "http://localhost:8000"


def _infer_url(model_name="parameter"):
    return f"{_HTTP_LOCALHOST}/v2/models/{model_name}/infer"


def _minimal_fp32_infer_body(parameters=None):
    """KServe HTTP infer JSON matching qa/L0_parameters model `parameter` (INPUT0 FP32 [1])."""
    body = {
        "inputs": [
            {
                "name": "INPUT0",
                "shape": [1],
                "datatype": "FP32",
                "data": [1.0],
            }
        ],
    }
    if parameters is not None:
        body["parameters"] = parameters
    return body


_FORWARD_HEADERS = {
    "header_1": "value_1",
    "header_2": "value_2",
    "my_header_1": "my_value_1",
    "my_header_2": "my_value_2",
    "my_header_3": 'This is a "quoted" string with a backslash\\ ',
}


def _grpc_stream_callback(user_data, result, error):
    if error:
        user_data.put(error)
    else:
        user_data.put(result)


class InferenceParametersTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.httpclient = httpclient.InferenceServerClient(url="localhost:8000")
        self.async_httpclient = asynchttpclient.InferenceServerClient(
            url="localhost:8000"
        )
        self.grpcclient = grpcclient.InferenceServerClient(url="localhost:8001")
        self.async_grpcclient = asyncgrpcclient.InferenceServerClient(
            url="localhost:8001"
        )
        self.grpcclient_callback = _grpc_stream_callback

    def create_inputs(self, client_type):
        inputs = []
        inputs.append(client_type.InferInput("INPUT0", [1], "FP32"))

        # Initialize the data
        inputs[0].set_data_from_numpy(np.asarray([1], dtype=np.float32))
        return inputs

    async def send_request_and_verify(
        self,
        client_type,
        client,
        parameters,
        headers,
        expected_headers,
        is_async_infer=False,
        model_name="parameter",
    ):
        inputs = self.create_inputs(client_type)

        if is_async_infer:
            if client_type == httpclient:
                result = client.async_infer(
                    model_name=model_name,
                    inputs=inputs,
                    parameters=parameters,
                    headers=headers,
                ).get_result()
            elif client_type == grpcclient:
                user_data = queue.Queue()
                client.async_infer(
                    model_name=model_name,
                    inputs=inputs,
                    parameters=parameters,
                    headers=headers,
                    callback=partial(self.grpcclient_callback, user_data),
                )
                result = user_data.get()
                self.assertIsNot(result, InferenceServerException)
            else:
                raise ValueError(f"Unsupported client type: {client_type}")
        else:
            infer_callable = partial(
                client.infer,
                model_name=model_name,
                inputs=inputs,
                parameters=parameters,
                headers=headers,
            )
            if client_type == asynchttpclient or client_type == asyncgrpcclient:
                result = await infer_callable()
            else:
                result = infer_callable()

        self.verify_outputs(result, parameters, expected_headers)

    def verify_outputs(self, result, parameters, expected_headers):
        keys = result.as_numpy("key")
        values = result.as_numpy("value")
        keys = keys.astype(str).tolist()
        expected_keys = list(parameters.keys()) + list(expected_headers.keys())
        self.assertEqual(
            set(keys),
            set(expected_keys),
            msg=f"keys: {keys}, expected_keys: {expected_keys}",
        )

        # We have to convert the parameter values to string
        expected_values = []
        for expected_value in list(parameters.values()):
            expected_values.append(str(expected_value))
        for value in expected_headers.values():
            expected_values.append(value)
        self.assertEqual(
            set(values.astype(str).tolist()),
            set(expected_values),
            msg=f"values: {values.astype(str).tolist()}, expected_values: {expected_values}",
        )

    async def _verify_grpc_stream_infer(self, parameters, headers, expected_headers):
        user_data = queue.Queue()
        self.grpcclient.start_stream(
            callback=partial(self.grpcclient_callback, user_data), headers=headers
        )
        inputs = self.create_inputs(grpcclient)
        self.grpcclient.async_stream_infer(
            model_name="parameter", inputs=inputs, parameters=parameters
        )
        result = user_data.get()
        self.assertIsNot(result, InferenceServerException)
        self.verify_outputs(result, parameters, expected_headers)
        self.grpcclient.stop_stream()

    def _raw_http_post_infer(self, body_dict, extra_headers=None):
        """POST /v2/models/.../infer without tritonclient (no header normalization)."""
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        return requests.post(
            _infer_url("parameter"),
            json=body_dict,
            headers=headers,
            timeout=60,
        )

    def _assert_raw_http_400(self, response, text_substr):
        """Assert status 400 and body contains `text_substr`."""
        snippet = response.text[:2000] if response.text else ""
        self.assertEqual(response.status_code, 400, msg=snippet)
        self.assertIn(text_substr, response.text, msg=snippet)

    async def _run_client_infer_suite(self, parameters, headers, expected_headers):
        """
        Full client matrix: gRPC/HTTP sync+async, stream, ensemble.
        """
        await self.send_request_and_verify(
            grpcclient, self.grpcclient, parameters, headers.copy(), expected_headers
        )
        await self.send_request_and_verify(
            httpclient, self.httpclient, parameters, headers.copy(), expected_headers
        )
        await self.send_request_and_verify(
            asynchttpclient,
            self.async_httpclient,
            parameters,
            headers.copy(),
            expected_headers,
        )
        await self.send_request_and_verify(
            asyncgrpcclient,
            self.async_grpcclient,
            parameters,
            headers.copy(),
            expected_headers,
        )
        await self.send_request_and_verify(
            httpclient,
            self.httpclient,
            parameters,
            headers.copy(),
            expected_headers,
            is_async_infer=True,
        )
        await self.send_request_and_verify(
            grpcclient,
            self.grpcclient,
            parameters,
            headers.copy(),
            expected_headers,
            is_async_infer=True,
        )
        await self._verify_grpc_stream_infer(
            parameters, headers.copy(), expected_headers
        )
        await self.send_request_and_verify(
            httpclient,
            self.httpclient,
            parameters,
            headers.copy(),
            expected_headers,
            model_name="ensemble",
        )

    async def test_params(self):
        for parameters in [
            {"key1": "value1", "key2": "value2"},
            {"key1": 1, "key2": 2},
            {"key1": 123.123, "key2": 321.321},
            {"key1": True, "key2": "value2"},
        ]:
            await self._run_client_infer_suite(parameters, {}, {})

    async def test_params_reserved_rejected(self):
        for reserved_key in _RESERVED_PARAMETER_KEYS:
            parameters = {reserved_key: "dummy-value"}
            body = _minimal_fp32_infer_body(parameters)

            # Raw HTTP
            if reserved_key.startswith("triton_"):
                r = self._raw_http_post_infer(body)
                self._assert_raw_http_400(r, _TRITON_RESERVED_SERVER_ERROR)

            # Python clients
            for client_type, client in (
                (httpclient, self.httpclient),
                (asynchttpclient, self.async_httpclient),
                (grpcclient, self.grpcclient),
                (asyncgrpcclient, self.async_grpcclient),
            ):
                inputs = self.create_inputs(client_type)
                with self.assertRaises(InferenceServerException) as cm:
                    if client_type in (asynchttpclient, asyncgrpcclient):
                        await client.infer(
                            model_name="parameter",
                            inputs=inputs,
                            parameters=parameters,
                        )
                    else:
                        client.infer(
                            model_name="parameter",
                            inputs=inputs,
                            parameters=parameters,
                        )
                msg = str(cm.exception)
                # Reserved parameters are rejected by the client
                self.assertIn(_TRITON_RESERVED_CLIENT_ERROR, msg, msg=msg)

    async def test_headers(self):
        expected_headers = {
            "my_header_1": "my_value_1",
            "my_header_2": "my_value_2",
            "my_header_3": 'This is a "quoted" string with a backslash\\ ',
        }
        await self._run_client_infer_suite({}, _FORWARD_HEADERS, expected_headers)

    async def test_grpc_header_forward_pattern_case_sensitive(self):
        expected_headers = {}
        await self._run_client_infer_suite({}, _FORWARD_HEADERS, expected_headers)

    async def test_headers_reserved_rejected(self):
        body = _minimal_fp32_infer_body({})
        for reserved_key in _RESERVED_PARAMETER_KEYS:
            # Raw HTTP
            r = self._raw_http_post_infer(
                body, extra_headers={reserved_key: "dummy-value"}
            )
            self._assert_raw_http_400(r, _TRITON_RESERVED_SERVER_ERROR)

            # Python clients
            for client_type, client in (
                (httpclient, self.httpclient),
                (asynchttpclient, self.async_httpclient),
                (grpcclient, self.grpcclient),
                (asyncgrpcclient, self.async_grpcclient),
            ):
                inputs = self.create_inputs(client_type)
                with self.assertRaises(InferenceServerException) as cm:
                    if client_type in (asynchttpclient, asyncgrpcclient):
                        await client.infer(
                            model_name="parameter",
                            inputs=inputs,
                            parameters={},
                            headers={reserved_key: "dummy-value"},
                        )
                    else:
                        client.infer(
                            model_name="parameter",
                            inputs=inputs,
                            parameters={},
                            headers={reserved_key: "dummy-value"},
                        )
                msg = str(cm.exception)
                # Headers are not rejected by the client
                self.assertIn(_TRITON_RESERVED_SERVER_ERROR, msg, msg=msg)

    async def asyncTearDown(self):
        self.httpclient.close()
        self.grpcclient.close()
        await self.async_grpcclient.close()
        await self.async_httpclient.close()


if __name__ == "__main__":
    unittest.main()
