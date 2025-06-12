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

import unittest

import numpy as np
import requests
import test_util as tu

# Constants for size calculations
# Each FP32 value is 4 bytes, so we need to divide target byte sizes by 4 to get element counts
BYTES_PER_FP32 = 4
MB = 2**20  # 1 MB = 1,048,576 bytes
DEFAULT_LIMIT_BYTES = 64 * MB  # 64MB default limit
INCREASED_LIMIT_BYTES = 128 * MB  # 128MB increased limit

# Calculate element counts for size limits
DEFAULT_LIMIT_ELEMENTS = DEFAULT_LIMIT_BYTES // BYTES_PER_FP32  # 16,777,216 elements
INCREASED_LIMIT_ELEMENTS = (
    INCREASED_LIMIT_BYTES // BYTES_PER_FP32
)  # 33,554,432 elements

# Small offsets to go just over/under the limits
OFFSET_ELEMENTS = 32


class InferSizeLimitTest(tu.TestResultCollector):
    def _get_infer_url(self, model_name):
        return "http://localhost:8000/v2/models/{}/infer".format(model_name)

    def test_default_limit_rejection_raw_binary(self):
        """Test raw binary inputs with default limit"""
        model = "onnx_zero_1_float32"

        # Test case 1: Input just over the 64MB limit (should fail)
        # (2^24 + 32) elements * 4 bytes = 64MB + 128 bytes = 67,108,992 bytes
        large_input = np.ones(
            DEFAULT_LIMIT_ELEMENTS + OFFSET_ELEMENTS, dtype=np.float32
        )
        input_bytes = large_input.tobytes()
        assert len(input_bytes) > 64 * MB  # Verify we're actually over the 64MB limit

        headers = {"Inference-Header-Content-Length": "0"}
        response = requests.post(
            self._get_infer_url(model), data=input_bytes, headers=headers
        )

        # Should fail with 400 bad request with default limit
        self.assertEqual(
            400,
            response.status_code,
            "Expected error code for oversized request, got: {}".format(
                response.status_code
            ),
        )

        # Verify error message contains size limit info
        error_msg = response.content.decode()
        self.assertIn(
            "exceeds the maximum allowed value",
            error_msg,
            "Expected error message about exceeding max input size",
        )

        # Test case 2: Input just under the 64MB limit (should succeed)
        # (2^24 - 32) elements * 4 bytes = 64MB - 128 bytes = 67,108,736 bytes
        small_input = np.ones(
            DEFAULT_LIMIT_ELEMENTS - OFFSET_ELEMENTS, dtype=np.float32
        )
        input_bytes = small_input.tobytes()
        assert len(input_bytes) < 64 * MB  # Verify we're actually under the 64MB limit

        response = requests.post(
            self._get_infer_url(model), data=input_bytes, headers=headers
        )

        # Should succeed with 200 OK
        self.assertEqual(
            200,
            response.status_code,
            "Expected success code for request within size limit, got: {}".format(
                response.status_code
            ),
        )

        # Verify output matches our input (identity model)
        header_size = int(response.headers["Inference-Header-Content-Length"])
        output_data = response.content[header_size:]

        # Convert output bytes back to numpy array for comparison
        output_array = np.frombuffer(output_data, dtype=np.float32)
        self.assertTrue(
            np.array_equal(output_array, small_input),
            "Response data does not match input data",
        )

    def test_default_limit_rejection_json(self):
        """Test JSON inputs with default limit"""
        model = "onnx_zero_1_float32"

        # Test case 1: Input just over the 64MB limit (should fail)
        # (2^24 + 32) elements * 4 bytes = 64MB + 128 bytes = 67,108,992 bytes
        shape_size = DEFAULT_LIMIT_ELEMENTS + OFFSET_ELEMENTS

        payload = {
            "inputs": [
                {
                    "name": "INPUT0",
                    "datatype": "FP32",
                    "shape": [1, shape_size],
                    "data": [1.0] * shape_size,
                }
            ]
        }
        assert (
            shape_size * BYTES_PER_FP32 > 64 * MB
        )  # Verify we're actually over the 64MB limit

        headers = {"Content-Type": "application/json"}
        response = requests.post(
            self._get_infer_url(model), headers=headers, json=payload
        )

        # Should fail with 400 bad request with default limit
        self.assertEqual(
            400,
            response.status_code,
            "Expected error code for oversized JSON request, got: {}".format(
                response.status_code
            ),
        )

        # Verify error message contains size limit info
        error_msg = response.content.decode()
        self.assertIn(
            "exceeds the maximum allowed value",
            error_msg,
            "Expected error message about exceeding max input size",
        )

        # Test case 2: Input just under the 64MB limit (should succeed)
        # (2^24 - 32) elements * 4 bytes = 64MB - 128 bytes = 67,108,736 bytes
        shape_size = DEFAULT_LIMIT_ELEMENTS - OFFSET_ELEMENTS

        payload = {
            "inputs": [
                {
                    "name": "INPUT0",
                    "datatype": "FP32",
                    "shape": [1, shape_size],
                    "data": [1.0] * shape_size,
                }
            ]
        }
        assert (
            shape_size * BYTES_PER_FP32 < 64 * MB
        )  # Verify we're actually under the 64MB limit

        response = requests.post(
            self._get_infer_url(model), headers=headers, json=payload
        )

        # Should succeed with 200 OK
        self.assertEqual(
            200,
            response.status_code,
            "Expected success code for JSON request within size limit, got: {}".format(
                response.status_code
            ),
        )

        # Verify we got a valid response
        result = response.json()
        self.assertIn("outputs", result, "Response missing outputs field")
        self.assertEqual(1, len(result["outputs"]), "Expected 1 output")
        self.assertEqual(
            shape_size,
            result["outputs"][0]["shape"][1],
            f"Expected shape {[1, shape_size]}, got {result['outputs'][0]['shape']}",
        )

    def test_large_input_raw_binary(self):
        """Test raw binary input larger with custom limit set"""
        model = "onnx_zero_1_float32"

        # Test case 1: Input just over the 128MB configured limit (should fail)
        # (2^25 + 32) elements * 4 bytes = 128MB + 128 bytes = 134,217,856 bytes
        large_input = np.ones(
            INCREASED_LIMIT_ELEMENTS + OFFSET_ELEMENTS, dtype=np.float32
        )
        input_bytes = large_input.tobytes()
        assert len(input_bytes) > 128 * MB  # Verify we're actually over the 128MB limit

        headers = {"Inference-Header-Content-Length": "0"}
        response = requests.post(
            self._get_infer_url(model), data=input_bytes, headers=headers
        )

        # Should fail with 400 bad request with our increased limit
        self.assertEqual(
            400,
            response.status_code,
            "Expected error code for oversized request, got: {}".format(
                response.status_code
            ),
        )

        # Verify error message contains size limit info
        error_msg = response.content.decode()
        self.assertIn(
            "exceeds the maximum allowed value",
            error_msg,
            "Expected error message about exceeding max input size",
        )

        # Test case 2: Input just under the 128MB configured limit (should succeed)
        # (2^25 - 32) elements * 4 bytes = 128MB - 128 bytes = 134,217,600 bytes
        small_input = np.ones(
            INCREASED_LIMIT_ELEMENTS - OFFSET_ELEMENTS, dtype=np.float32
        )
        input_bytes = small_input.tobytes()
        assert (
            len(input_bytes) < 128 * MB
        )  # Verify we're actually under the 128MB limit

        response = requests.post(
            self._get_infer_url(model), data=input_bytes, headers=headers
        )

        # Should succeed with 200 OK
        self.assertEqual(
            200,
            response.status_code,
            "Expected success code for request within increased limit, got: {}".format(
                response.status_code
            ),
        )

        # Verify output matches our input (identity model)
        header_size = int(response.headers["Inference-Header-Content-Length"])
        output_data = response.content[header_size:]

        # Convert output bytes back to numpy array for comparison
        output_array = np.frombuffer(output_data, dtype=np.float32)
        self.assertTrue(
            np.array_equal(output_array, small_input),
            "Response data does not match input data",
        )

    def test_large_input_json(self):
        """Test JSON input larger with custom limit set"""
        model = "onnx_zero_1_float32"

        # Test case 1: Input just over the 128MB configured limit (should fail)
        # (2^25 + 32) elements * 4 bytes = 128MB + 128 bytes = 134,217,856 bytes
        shape_size = INCREASED_LIMIT_ELEMENTS + OFFSET_ELEMENTS

        payload = {
            "inputs": [
                {
                    "name": "INPUT0",
                    "datatype": "FP32",
                    "shape": [1, shape_size],
                    "data": [1.0] * shape_size,
                }
            ]
        }
        assert (
            shape_size * BYTES_PER_FP32 > 128 * MB
        )  # Verify we're actually over the 128MB limit

        headers = {"Content-Type": "application/json"}
        response = requests.post(
            self._get_infer_url(model), headers=headers, json=payload
        )

        # Should fail with 400 bad request with our increased limit
        self.assertEqual(
            400,
            response.status_code,
            "Expected error code for oversized JSON request, got: {}".format(
                response.status_code
            ),
        )

        # Verify error message contains size limit info
        error_msg = response.content.decode()
        self.assertIn(
            "exceeds the maximum allowed value",
            error_msg,
            "Expected error message about exceeding max input size",
        )

        # Test case 2: Input just under the 128MB configured limit (should succeed)
        # (2^25 - 32) elements * 4 bytes = 128MB - 128 bytes = 134,217,600 bytes
        shape_size = INCREASED_LIMIT_ELEMENTS - OFFSET_ELEMENTS

        payload = {
            "inputs": [
                {
                    "name": "INPUT0",
                    "datatype": "FP32",
                    "shape": [1, shape_size],
                    "data": [1.0] * shape_size,
                }
            ]
        }
        assert (
            shape_size * BYTES_PER_FP32 < 128 * MB
        )  # Verify we're actually under the 128MB limit

        response = requests.post(
            self._get_infer_url(model), headers=headers, json=payload
        )

        # Should succeed with 200 OK
        self.assertEqual(
            200,
            response.status_code,
            "Expected success code for request within increased limit, got: {}".format(
                response.status_code
            ),
        )

        # Verify we got a valid response
        result = response.json()
        self.assertIn("outputs", result, "Response missing outputs field")
        self.assertEqual(1, len(result["outputs"]), "Expected 1 output")
        self.assertEqual(
            shape_size,
            result["outputs"][0]["shape"][1],
            f"Expected shape {[1, shape_size]}, got {result['outputs'][0]['shape']}",
        )


if __name__ == "__main__":
    unittest.main()
