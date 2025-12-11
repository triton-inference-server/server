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

import gzip
import io
import json
import unittest

import numpy as np
import requests
import test_util as tu

# Constants for size calculations
# Each FP32 value is 4 bytes, so we need to divide target byte sizes by 4 to get element counts
BYTES_PER_FP32 = 4
MB = 2**20  # 1 MB = 1,048,576 bytes
GB = 2**30  # 1 GB = 1,073,741,824 bytes
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

    def test_default_limit_raw_binary(self):
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

    def test_default_limit_json(self):
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
        # The test creates a JSON payload with data, which adds overhead compared
        # to raw binary format. We adjust the shape size to ensure the final
        # JSON payload is under the size limit. An element is roughly 5
        # bytes in JSON, compared to 4 bytes as a raw FP32.
        shape_size = (DEFAULT_LIMIT_ELEMENTS - OFFSET_ELEMENTS) * 4 // 5

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
        # Verify we're actually under the 64MB limit
        self.assertLess(len(json.dumps(payload).encode("utf-8")), DEFAULT_LIMIT_BYTES)

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
        # The test creates a JSON payload with data, which adds overhead compared
        # to raw binary format. We adjust the shape size to ensure the final
        # JSON payload is under the size limit. An element is roughly 5
        # bytes in JSON, compared to 4 bytes as a raw FP32.
        shape_size = (INCREASED_LIMIT_ELEMENTS - OFFSET_ELEMENTS) * 4 // 5

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
        # Verify we're actually under the 128MB limit
        self.assertLess(len(json.dumps(payload).encode("utf-8")), INCREASED_LIMIT_BYTES)

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

    def test_large_string_in_json(self):
        """Test JSON request with large string input"""
        model = "simple_identity"

        # Create a string that is larger (large payload about 2GB) than the default limit of 64MB
        # (2^31 + 64) elements * 1 bytes = 2GB + 64 bytes = 2,147,483,712 bytes
        large_string_size = 2 * GB + 64
        large_string = "A" * large_string_size

        payload = {
            "inputs": [
                {
                    "name": "INPUT0",
                    "datatype": "BYTES",
                    "shape": [1, 1],
                    "data": [large_string],
                }
            ]
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(
            self._get_infer_url(model), headers=headers, json=payload
        )

        # Should fail with 400 bad request
        self.assertEqual(
            400,
            response.status_code,
            "Expected error code for oversized JSON request, got: {}".format(
                response.status_code
            ),
        )

        # Verify error message
        error_msg = response.content.decode()
        self.assertIn(
            "Request JSON size",
            error_msg,
        )
        self.assertIn(
            "exceeds the maximum allowed value",
            error_msg,
        )
        self.assertIn(
            "Use --http-max-input-size to increase the limit",
            error_msg,
        )

    def _create_compressed_payload(self, target_size):
        """Helper to create a gzip-compressed JSON payload of specified decompressed size."""
        shape_size = 1000  # Small actual data
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
        json_str = json.dumps(payload, indent=4)

        # Pad with whitespace to reach target size (whitespace before closing brace is valid JSON)
        padding_needed = target_size - len(json_str)
        padded_json = json_str[:-1] + (" " * padding_needed) + json_str[-1]

        # Compress the payload
        compressed_buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed_buffer, mode="wb") as gz:
            gz.write(padded_json.encode("utf-8"))

        return compressed_buffer.getvalue(), len(padded_json.encode("utf-8"))

    def test_default_limit_compressed(self):
        """Test compressed inputs with default 64MB limit.

        This test verifies that the --http-max-input-size limit is enforced on
        the decompressed data size, not just the compressed request size.
        """
        model = "onnx_zero_1_float32"

        headers = {
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
        }

        # Test case 1: Payload that decompresses to 64MB + 1MB (over limit) should fail
        large_target_size = DEFAULT_LIMIT_BYTES + MB
        (
            large_compressed_data,
            large_uncompressed_size,
        ) = self._create_compressed_payload(large_target_size)

        # Verify uncompressed size is over 64MB limit
        self.assertGreater(
            large_uncompressed_size,
            DEFAULT_LIMIT_BYTES,
            f"Large payload should decompress to > 64MB, got {large_uncompressed_size}",
        )

        # Verify compressed size is under the limit
        self.assertLess(
            len(large_compressed_data),
            DEFAULT_LIMIT_BYTES,
            f"Compressed size should be under limit, got {len(large_compressed_data)}",
        )

        response = requests.post(
            self._get_infer_url(model), data=large_compressed_data, headers=headers
        )

        # Should fail with 400 bad request - decompressed size exceeds limit
        self.assertEqual(
            400,
            response.status_code,
            f"Expected 400 for compressed request that decompresses to >64MB, got: {response.status_code}",
        )

        # Verify error message contains size limit info
        error_msg = response.content.decode()
        self.assertIn(
            "exceeds the maximum allowed value",
            error_msg,
            "Expected error message about exceeding max input size",
        )

        # Test case 2: Payload that decompresses to 64MB - 1MB (under limit) should succeed
        small_target_size = DEFAULT_LIMIT_BYTES - MB
        (
            small_compressed_data,
            small_uncompressed_size,
        ) = self._create_compressed_payload(small_target_size)

        # Verify uncompressed size is under 64MB limit
        self.assertLess(
            small_uncompressed_size,
            DEFAULT_LIMIT_BYTES,
            f"Small payload should decompress to < 64MB, got {small_uncompressed_size}",
        )

        response = requests.post(
            self._get_infer_url(model), data=small_compressed_data, headers=headers
        )

        # Should succeed with 200 OK
        self.assertEqual(
            200,
            response.status_code,
            f"Expected 200 for compressed request within limit, got: {response.status_code}",
        )

        # Verify we got a valid response
        result = response.json()
        self.assertIn("outputs", result, "Response missing outputs field")

    def test_large_input_compressed(self):
        """Test compressed inputs with custom 128MB limit set.

        This test verifies that compressed inputs work correctly when the
        --http-max-input-size limit is increased.
        """
        model = "onnx_zero_1_float32"

        headers = {
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
        }

        # Test case 1: Input that decompresses to 128MB + 1MB (over limit) should fail
        large_target_size = INCREASED_LIMIT_BYTES + MB
        (
            large_compressed_data,
            large_uncompressed_size,
        ) = self._create_compressed_payload(large_target_size)

        # Verify sizes
        self.assertGreater(
            large_uncompressed_size,
            INCREASED_LIMIT_BYTES,
            f"Large payload should decompress to > 128MB, got {large_uncompressed_size}",
        )

        response = requests.post(
            self._get_infer_url(model), data=large_compressed_data, headers=headers
        )

        # Should fail with 400 bad request
        self.assertEqual(
            400,
            response.status_code,
            f"Expected 400 for compressed request exceeding 128MB limit, got: {response.status_code}",
        )

        error_msg = response.content.decode()
        self.assertIn(
            "exceeds the maximum allowed value",
            error_msg,
            "Expected error message about exceeding max input size",
        )

        # Test case 2: Input that decompresses to 128MB - 1MB (under limit) should succeed
        small_target_size = INCREASED_LIMIT_BYTES - MB
        (
            small_compressed_data,
            small_uncompressed_size,
        ) = self._create_compressed_payload(small_target_size)

        # Verify sizes
        self.assertLess(
            small_uncompressed_size,
            INCREASED_LIMIT_BYTES,
            f"Small payload should decompress to < 128MB, got {small_uncompressed_size}",
        )
        self.assertGreater(
            small_uncompressed_size,
            DEFAULT_LIMIT_BYTES,
            f"Small payload should decompress to > 64MB (default), got {small_uncompressed_size}",
        )

        response = requests.post(
            self._get_infer_url(model), data=small_compressed_data, headers=headers
        )

        # Should succeed with 200 OK
        self.assertEqual(
            200,
            response.status_code,
            f"Expected 200 for compressed request within 128MB limit, got: {response.status_code}",
        )

        # Verify we got a valid response
        result = response.json()
        self.assertIn("outputs", result, "Response missing outputs field")


if __name__ == "__main__":
    unittest.main()
