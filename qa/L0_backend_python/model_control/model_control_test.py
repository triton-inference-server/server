#!/usr/bin/env python3

# Copyright 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os
import sys

sys.path.append("../../common")

import unittest

import numpy as np
import shm_util
import tritonclient.http as httpclient
from tritonclient.utils import *

# By default, find tritonserver on "localhost", but for windows tests
# we overwrite the IP address with the TRITONSERVER_IPADDR envvar
_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class ExplicitModelTest(unittest.TestCase):
    def setUp(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()

    def send_identity_request(self, client, model_name):
        inputs = []
        inputs.append(httpclient.InferInput("INPUT0", [1, 16], "FP32"))
        input0_data = np.arange(start=0, stop=16, dtype=np.float32)
        input0_data = np.expand_dims(input0_data, axis=0)
        inputs[0].set_data_from_numpy(input0_data)

        with self._shm_leak_detector.Probe() as shm_probe:
            result = client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=[httpclient.InferRequestedOutput("OUTPUT0")],
            )
        output_numpy = result.as_numpy("OUTPUT0")
        self.assertTrue(np.all(input0_data == output_numpy))

    def test_model_reload(self):
        model_name = "identity_fp32"
        ensemble_model_name = "simple_" + "identity_fp32"
        with httpclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8000") as client:
            for _ in range(5):
                self.assertFalse(client.is_model_ready(model_name))
                # Load the model before the ensemble model to make sure reloading the
                # model works properly in Python backend.
                client.load_model(model_name)
                client.load_model(ensemble_model_name)
                self.assertTrue(client.is_model_ready(model_name))
                self.assertTrue(client.is_model_ready(ensemble_model_name))
                self.send_identity_request(client, model_name)
                self.send_identity_request(client, ensemble_model_name)
                client.unload_model(ensemble_model_name)
                client.unload_model(model_name)
                self.assertFalse(client.is_model_ready(model_name))
                self.assertFalse(client.is_model_ready(ensemble_model_name))


class InputValidationTest(unittest.TestCase):
    """
    Test input validation for user-provided inputs
    """

    def setUp(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()
        self._client = httpclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8000")
        self._triton_host = _tritonserver_ipaddr
        self._triton_port = 8000

        # Check if curl is available
        try:
            subprocess.run(["curl", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.skipTest("curl command not available - required for raw HTTP testing")

    def _send_load_model_request(self, model_name):
        """Send HTTP request to load model for testing input validation using curl"""
        payload = {
            "parameters": {
                "config": f'{{"name": "{model_name}", "backend": "python", "max_batch_size": 4}}',
                "file:/1/model.py": "print('Hello from Python Model')",
            }
        }

        url = f"http://{self._triton_host}:{self._triton_port}/v2/repository/models/{model_name}/load"

        # Create a temporary file for the JSON payload
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            payload_file = f.name

        try:
            # Use curl to send the request
            curl_cmd = [
                "curl",
                "-s",
                "-w",
                "\n%{http_code}",
                "-X",
                "POST",
                "-H",
                "Content-Type: application/json",
                "-d",
                f"@{payload_file}",
                "--connect-timeout",
                "10",
            ]

            # Add the URL as a separate argument to avoid shell interpretation issues
            curl_cmd.append(url)

            # Debug: print the exact URL being requested
            print(f"DEBUG: Curl URL: {url}")

            result = subprocess.run(
                curl_cmd, capture_output=True, text=True, timeout=15
            )

            # Parse curl output - last line is status code, rest is response body
            output_lines = (
                result.stdout.strip().split("\n") if result.stdout.strip() else []
            )
            if len(output_lines) >= 2:
                try:
                    status_code = int(output_lines[-1])
                    response_text = "\n".join(output_lines[:-1])
                except ValueError:
                    status_code = 0
                    response_text = result.stdout or result.stderr or "Invalid response"
            elif len(output_lines) == 1 and output_lines[0].isdigit():
                status_code = int(output_lines[0])
                response_text = result.stderr or "No response body"
            else:
                status_code = 0
                response_text = result.stdout or result.stderr or "No response"

            # Return an object similar to requests.Response
            class CurlResponse:
                def __init__(self, status_code, text):
                    self.status_code = status_code
                    self.text = text
                    self.content = text.encode()

            return CurlResponse(status_code, response_text)

        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            ValueError,
        ) as e:
            # Return a mock response for errors
            class ErrorResponse:
                def __init__(self, error_msg):
                    self.status_code = 0
                    self.text = f"Error: {error_msg}"
                    self.content = self.text.encode()

            return ErrorResponse(str(e))
        finally:
            # Clean up temporary file
            try:
                os.unlink(payload_file)
            except:
                pass

    def test_invalid_character_model_names(self):
        """Test that model names with invalid characters are properly rejected"""

        # Model names with various invalid characters that should be rejected
        invalid_model_names = [
            "model$(test)",
            "model\{test\}",
            "model`test`",
            "model;test",
            "model|test",
            "model&test",
            "model'test'",
            "model*test",
            "model!test",
        ]

        for invalid_name in invalid_model_names:
            with self.subTest(model_name=invalid_name):
                print(f"Testing invalid model name: {invalid_name}")

                response = self._send_load_model_request(invalid_name)
                print(
                    f"Response for '{invalid_name}': Status {response.status_code}, Text: {response.text[:200]}..."
                )

                # Should not get a successful 200 response
                self.assertNotEqual(
                    200,
                    response.status_code,
                    f"Invalid model name '{invalid_name}' should not get 200 OK response",
                )

                self.assertIn(
                    "Invalid stub name: contains invalid characters",
                    response.text,
                    f"invalid response for '{invalid_name}' should contain 'Invalid stub name: contains invalid characters'",
                )

    def test_valid_model_names(self):
        """Test that valid model names work"""

        valid_model_names = [
            "TestModel123",
            "model-with-hyphens",
            "model_with_underscores",
        ]

        for valid_name in valid_model_names:
            with self.subTest(model_name=valid_name):
                print(f"Testing valid model name: {valid_name}")

                response = self._send_load_model_request(valid_name)
                print(
                    f"Response for valid '{valid_name}': Status {response.status_code}, Text: {response.text[:100]}..."
                )

                # Valid names might still fail for other reasons (model doesn't exist, etc.)
                # but they should not be rejected due to character validation
                # We just check it's not a validation error
                self.assertNotIn(
                    "Invalid stub name: contains invalid characters",
                    response.text,
                    f"valid response for '{valid_name}' should not contain 'Invalid stub name: contains invalid characters'",
                )


if __name__ == "__main__":
    unittest.main()
