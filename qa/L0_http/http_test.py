#!/usr/bin/python
# Copyright 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import base64
import json
import threading
import time
import unittest

import numpy as np
import requests
import test_util as tu
import tritonclient.http as tritonhttpclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype


class HttpTest(tu.TestResultCollector):
    def _get_infer_url(self, model_name):
        return "http://localhost:8000/v2/models/{}/infer".format(model_name)

    def _get_load_model_url(self, model_name):
        return "http://localhost:8000/v2/repository/models/{}/load".format(model_name)

    def _raw_binary_helper(
        self, model, input_bytes, expected_output_bytes, extra_headers={}
    ):
        # Select model that satisfies constraints for raw binary request
        headers = {"Inference-Header-Content-Length": "0"}
        # Add extra headers (if any) before sending request
        headers.update(extra_headers)
        r = requests.post(self._get_infer_url(model), data=input_bytes, headers=headers)
        r.raise_for_status()

        # Get the inference header size so we can locate the output binary data
        header_size = int(r.headers["Inference-Header-Content-Length"])
        # Assert input == output since this tests an identity model
        self.assertEqual(
            expected_output_bytes,
            r.content[header_size:],
            "Expected response body contains correct output binary data: {}; got: {}".format(
                expected_output_bytes, r.content[header_size:]
            ),
        )

    def test_raw_binary(self):
        model = "onnx_zero_1_float32"
        input_bytes = np.arange(8, dtype=np.float32).tobytes()
        self._raw_binary_helper(model, input_bytes, input_bytes)

    def test_raw_binary_longer(self):
        # Similar to test_raw_binary but test with different data size
        model = "onnx_zero_1_float32"
        input_bytes = np.arange(32, dtype=np.float32).tobytes()
        self._raw_binary_helper(model, input_bytes, input_bytes)

    def test_byte(self):
        # Select model that satisfies constraints for raw binary request
        # i.e. BYTE type the element count must be 1
        model = "onnx_zero_1_object_1_element"
        input = "427"
        headers = {"Inference-Header-Content-Length": "0"}
        r = requests.post(self._get_infer_url(model), data=input, headers=headers)
        r.raise_for_status()

        # Get the inference header size so we can locate the output binary data
        header_size = int(r.headers["Inference-Header-Content-Length"])
        # Triton returns BYTES tensor with byte size prepended
        output = r.content[header_size + 4 :].decode()
        self.assertEqual(
            input,
            output,
            "Expected response body contains correct output binary data: {}; got: {}".format(
                input, output
            ),
        )

    def test_byte_too_many_elements(self):
        # Select model that doesn't satisfy constraints for raw binary request
        # i.e. BYTE type the element count must be 1
        model = "onnx_zero_1_object"
        input = "427"
        headers = {"Inference-Header-Content-Length": "0"}
        r = requests.post(self._get_infer_url(model), data=input, headers=headers)
        self.assertEqual(
            400,
            r.status_code,
            "Expected error code {} returned for the request; got: {}".format(
                400, r.status_code
            ),
        )
        self.assertIn(
            "For BYTE datatype raw input 'INPUT0', the model must have input shape [1]",
            r.content.decode(),
        )

    def test_multi_variable_dimensions(self):
        # Select model that doesn't satisfy constraints for raw binary request
        # i.e. this model has multiple variable-sized dimensions
        model = "onnx_zero_1_float16"
        input = np.ones([2, 2], dtype=np.float16)
        headers = {"Inference-Header-Content-Length": "0"}
        r = requests.post(
            self._get_infer_url(model), data=input.tobytes(), headers=headers
        )
        self.assertEqual(
            400,
            r.status_code,
            "Expected error code {} returned for the request; got: {}".format(
                400, r.status_code
            ),
        )
        self.assertIn(
            "The shape of the raw input 'INPUT0' can not be deduced because there are more than one variable-sized dimension",
            r.content.decode(),
        )

    def test_multi_inputs(self):
        # Select model that doesn't satisfy constraints for raw binary request
        # i.e. input count must be 1
        model = "onnx_zero_3_float32"
        # Use one numpy array, after tobytes() it can be seen as three inputs
        # each with 8 elements (this ambiguity is why this is not allowed)
        input = np.arange(24, dtype=np.float32)
        headers = {"Inference-Header-Content-Length": "0"}
        r = requests.post(
            self._get_infer_url(model), data=input.tobytes(), headers=headers
        )
        self.assertEqual(
            400,
            r.status_code,
            "Expected error code {} returned for the request; got: {}".format(
                400, r.status_code
            ),
        )
        self.assertIn(
            "Raw request must only have 1 input (found 1) to be deduced but got 3 inputs in",
            r.content.decode(),
        )

    # This is to test that a properly chunk-encoded request by the caller works,
    # though Triton does not specifically do any special chunk handling outside
    # of underlying HTTP libraries used
    # Future Enhancement: Test other encodings as they come up
    def test_content_encoding_chunked_manually(self):
        # Similar to test_raw_binary but test with extra headers
        extra_headers = {"Transfer-Encoding": "chunked"}
        model = "onnx_zero_1_float32"
        input_bytes = np.arange(8, dtype=np.float32).tobytes()
        # Encode input into a single chunk (for simplicity) following chunked
        # encoding format: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Transfer-Encoding
        chunk_encoded_input = b""
        # Length of chunk in hexadecimal and line separator
        chunk_encoded_input += f"{len(input_bytes):X}\r\n".encode("utf-8")
        # Chunk bytes and line separator
        chunk_encoded_input += input_bytes + b"\r\n"
        # Final byte (0) and end message
        chunk_encoded_input += b"0\r\n\r\n"
        self._raw_binary_helper(model, chunk_encoded_input, input_bytes, extra_headers)

    # Test that Python client rejects any "Transfer-Encoding" HTTP headers
    # as we don't specially handle encoding requests for the user through
    # these headers. There are special arguments exposed in the client to
    # handle some "Content-Encoding" headers.
    def test_content_encoding_unsupported_client(self):
        for encoding in ["chunked", "compress", "deflate", "gzip"]:
            with self.subTest(encoding=encoding):
                headers = {"Transfer-Encoding": encoding}
                np_input = np.arange(8, dtype=np.float32).reshape(1, -1)
                model = "onnx_zero_1_float32"
                # Setup inputs
                inputs = []
                inputs.append(
                    tritonhttpclient.InferInput(
                        "INPUT0", np_input.shape, np_to_triton_dtype(np_input.dtype)
                    )
                )
                inputs[0].set_data_from_numpy(np_input)

                with tritonhttpclient.InferenceServerClient("localhost:8000") as client:
                    # Python client is expected to raise an exception to reject
                    # 'content-encoding' HTTP headers.
                    with self.assertRaisesRegex(
                        InferenceServerException, "Unsupported HTTP header"
                    ):
                        client.infer(model_name=model, inputs=inputs, headers=headers)

    def test_descriptive_status_code(self):
        model = "onnx_zero_1_float32_queue"
        input_bytes = np.arange(8, dtype=np.float32).tobytes()

        # Send two requests to model that only queues 1 request at the maximum,
        # Expect the second request will be rejected with HTTP status code that
        # aligns with error detail (server unavailable).
        t = threading.Thread(
            target=self._raw_binary_helper, args=(model, input_bytes, input_bytes)
        )
        t.start()
        time.sleep(0.5)
        with self.assertRaises(requests.exceptions.HTTPError) as context:
            self._raw_binary_helper(model, input_bytes, input_bytes)
        self.assertEqual(
            503,
            context.exception.response.status_code,
            "Expected error code {} returned for the request; got: {}".format(
                503,
                context.exception.response.status_code,
            ),
        )
        t.join()

    def test_buffer_size_overflow(self):
        model = "onnx_zero_1_float32"

        # Test for overflow within GetElementCount()
        payload1 = {
            "inputs": [
                {
                    "name": "INPUT0",
                    "shape": [
                        2**4,
                        2**60 + 2,
                    ],  # This evaluates to 2^64 + 32 during GetElementCount()
                    "datatype": "FP32",
                    "data": [1.0],
                }
            ]
        }

        # Test for overflow with type_byte_size multiplication
        payload2 = {
            "inputs": [
                {
                    "name": "INPUT0",
                    "shape": [
                        2**2,
                        2**60 + 2,
                    ],  # This evaluates to 2^64 + 32 during type_byte_size multiplication since FP32 is 4 bytes
                    "datatype": "FP32",
                    "data": [1.0],
                }
            ]
        }

        # Send request and expect a 400 error with specific overflow message
        headers = {"Content-Type": "application/json"}

        # Test the first payload (GetElementCount overflow)
        r1 = requests.post(self._get_infer_url(model), json=payload1, headers=headers)

        self.assertEqual(
            400,
            r1.status_code,
            "Expected error code 400 for GetElementCount overflow check; got: {}".format(
                r1.status_code
            ),
        )

        error_message1 = r1.content.decode()
        self.assertIn(
            "causes total element count to exceed maximum size of", error_message1
        )

        # Test the second payload (type_byte_size multiplication overflow)
        r2 = requests.post(self._get_infer_url(model), json=payload2, headers=headers)

        self.assertEqual(
            400,
            r2.status_code,
            "Expected error code 400 for type_byte_size multiplication overflow check; got: {}".format(
                r2.status_code
            ),
        )

        error_message2 = r2.content.decode()
        self.assertIn("byte size overflow for input", error_message2)

    def test_negative_dimensions(self):
        model = "onnx_zero_1_float32"

        payload = {
            "inputs": [
                {
                    "name": "INPUT0",
                    "shape": [2, -5],  # Negative dimension should be invalid
                    "datatype": "FP32",
                    "data": [1.0],
                }
            ]
        }

        # Send request and expect a 500 error
        headers = {"Content-Type": "application/json"}
        r = requests.post(self._get_infer_url(model), json=payload, headers=headers)

        self.assertEqual(
            500,
            r.status_code,
            "Expected error code 500 for negative dimension; got: {}".format(
                r.status_code
            ),
        )

        error_message = r.content.decode()
        self.assertIn(
            "Unable to parse 'shape': attempt to access JSON non-unsigned-integer as unsigned-integer",
            error_message,
        )

    def test_loading_large_invalid_model(self):
        # Generate large base64 encoded data
        data_length = 1 << 31
        int_max = (1 << 31) - 1
        random_data = b"A" * data_length
        encoded_data = base64.b64encode(random_data)

        assert (
            len(encoded_data) > int_max
        ), "Encoded data length does not match the required length."

        # Prepare payload with large base64 encoded data
        payload = {
            "parameters": {
                "config": json.dumps({"backend": "onnxruntime"}),
                "file:1/model.onnx": encoded_data.decode("utf-8"),
            }
        }
        headers = {"Content-Type": "application/json"}

        # Send POST request
        response = requests.post(
            self._get_load_model_url("invalid_onnx"), headers=headers, json=payload
        )

        # Assert the response is not successful
        self.assertNotEqual(response.status_code, 200)
        try:
            error_message = response.json().get("error", "")
            self.assertIn(
                "'file:1/model.onnx' exceeds the maximum allowed data size limit "
                "INT_MAX",
                error_message,
            )
        except ValueError:
            self.fail("Response is not valid JSON")

    def test_json_recursion_depth_limit(self):
        """Test that server properly handles and rejects deeply nested JSON."""

        def create_nested_json(depth, value):
            for _ in range(depth):
                value = f"[{value}]"
            return json.loads(value)

        headers = {"Content-Type": "application/json"}
        test_matrix = [
            # (datatype, data, model, json_depth, should_succeed)
            ("BYTES", '"hello"', "simple_identity", 120, False),
            ("BYTES", '"hello"', "simple_identity", 50, True),
            ("INT64", "123", "simple_identity_int64", 120, False),
            ("INT64", "123", "simple_identity_int64", 50, True),
        ]

        for dtype, data, model, json_depth, should_succeed in test_matrix:
            with self.subTest(
                datatype=dtype, depth=json_depth, should_succeed=should_succeed
            ):
                payload = {
                    "inputs": [
                        {
                            "name": "INPUT0",
                            "datatype": dtype,
                            "shape": [1, 1],
                            "data": create_nested_json(json_depth, data),
                        }
                    ]
                }

                response = requests.post(
                    self._get_infer_url(model), headers=headers, json=payload
                )

                if should_succeed:
                    self.assertEqual(response.status_code, 200)
                else:
                    self.assertNotEqual(response.status_code, 200)
                    try:
                        error_message = response.json().get("error", "")
                        self.assertIn(
                            "JSON nesting depth exceeds maximum allowed limit (100)",
                            error_message,
                        )
                    except ValueError:
                        self.fail("Response is not valid JSON")


if __name__ == "__main__":
    unittest.main()
