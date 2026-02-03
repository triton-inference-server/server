# Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest
import numpy as np
import time
import threading
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

URL_HTTP = "localhost:8000"
URL_GRPC = "localhost:8001"


def call_inference(model_name, protocol, client):
    """Helper to test inference functionality"""
    input_data = np.random.randn(16).astype(np.float32)

    if protocol == "http":
        inputs = [httpclient.InferInput("INPUT0", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        result = client.infer(model_name, inputs)
        output_data = result.as_numpy("OUTPUT0")
    else:
        inputs = [grpcclient.InferInput("INPUT0", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        result = client.infer(model_name, inputs)
        output_data = result.as_numpy("OUTPUT0")

    np.testing.assert_array_almost_equal(
        input_data,
        output_data,
        err_msg=f"Inference output mismatch for {model_name}",
    )


class TestModelReadiness(unittest.TestCase):
    def setUp(self):
        self.model_name = "identity_fp32"
        self.client_http = httpclient.InferenceServerClient(url=URL_HTTP)
        self.client_grpc = grpcclient.InferenceServerClient(url=URL_GRPC)

    def test_model_ready(self):
        # Check HTTP
        try:
            is_ready = self.client_http.is_model_ready(self.model_name)
            self.assertTrue(
                is_ready, f"[HTTP] Model {self.model_name} should be READY but is NOT"
            )
            call_inference(self.model_name, "http", self.client_http)
        except Exception as e:
            self.fail(f"[HTTP] Unexpected error: {str(e)}")

        # Check gRPC
        try:
            is_ready = self.client_grpc.is_model_ready(self.model_name)
            self.assertTrue(
                is_ready, f"[gRPC] Model {self.model_name} should be READY but is NOT"
            )
            call_inference(self.model_name, "grpc", self.client_grpc)
        except Exception as e:
            self.fail(f"[gRPC] Unexpected error: {str(e)}")

    def test_model_not_ready(self):
        # Check HTTP
        try:
            is_ready = self.client_http.is_model_ready(self.model_name)
            self.assertFalse(
                is_ready,
                f"[HTTP] Model {self.model_name} should be NOT READY but is READY",
            )
        except Exception as e:
            self.fail(f"[HTTP] Unexpected error: {str(e)}")

        # Check gRPC
        try:
            is_ready = self.client_grpc.is_model_ready(self.model_name)
            self.assertFalse(
                is_ready,
                f"[gRPC] Model {self.model_name} should be NOT READY but is READY.",
            )
        except Exception as e:
            self.fail(f"[gRPC] Unexpected error: {str(e)}")


class TestUserDefinedModelReadinessFunction(unittest.TestCase):
    """
    Test user-defined is_model_ready() function
    """

    def setUp(self):
        self.client_http = httpclient.InferenceServerClient(url=URL_HTTP)
        self.client_grpc = grpcclient.InferenceServerClient(url=URL_GRPC)

    def test_is_model_ready_returns_true(self):
        model_name = "is_model_ready_fn_returns_true"

        # Send many sequential requests to ensure consistent behavior
        for i in range(10):
            self.assertTrue(
                self.client_http.is_model_ready(model_name),
                f"Model {model_name} should be READY",
            )
            self.assertTrue(
                self.client_grpc.is_model_ready(model_name),
                f"Model {model_name} should be READY",
            )

        # Inference should work
        # readiness check functionality should not affect inference
        call_inference(model_name, "http", self.client_http)
        call_inference(model_name, "grpc", self.client_grpc)

    def test_is_model_ready_returns_false(self):
        model_name = "is_model_ready_fn_returns_false"

        # Send many sequential requests to ensure consistent behavior
        for i in range(10):
            self.assertFalse(
                self.client_http.is_model_ready(model_name),
                f"Model {model_name} should be NOT READY",
            )
            self.assertFalse(
                self.client_grpc.is_model_ready(model_name),
                f"Model {model_name} should be NOT READY",
            )

        # Inference should still work
        # readiness check functionality should not affect inference
        call_inference(model_name, "http", self.client_http)
        call_inference(model_name, "grpc", self.client_grpc)

    def test_is_model_ready_raises_exception(self):
        model_name = "is_model_ready_fn_raises_error"

        # Send many sequential requests to ensure consistent behavior
        for i in range(10):
            self.assertFalse(
                self.client_http.is_model_ready(model_name),
                f"Model {model_name} should be NOT READY (exception)",
            )
            self.assertFalse(
                self.client_grpc.is_model_ready(model_name),
                f"Model {model_name} should be NOT READY (exception)",
            )
        
        # Test good model afterwards to ensure server is healthy
        model_name = "is_model_ready_fn_returns_true"
        for i in range(10):
            self.assertTrue(
                self.client_http.is_model_ready(model_name),
                f"Model {model_name} should be READY",
            )
            self.assertTrue(
                self.client_grpc.is_model_ready(model_name),
                f"Model {model_name} should be READY",
            )

        # Inference should still work
        # readiness check functionality should not affect inference
        call_inference(model_name, "http", self.client_http)
        call_inference(model_name, "grpc", self.client_grpc)

    def test_is_model_ready_returns_non_boolean(self):
        model_name = "is_model_ready_fn_returns_non_boolean"

        # Send many sequential requests to ensure consistent behavior
        for i in range(10):
            self.assertFalse(
                self.client_http.is_model_ready(model_name),
                f"Model {model_name} should be NOT READY (wrong return type)",
            )
            self.assertFalse(
                self.client_grpc.is_model_ready(model_name),
                f"Model {model_name} should be NOT READY (wrong return type)",
            )

        # Inference should still work
        # readiness check functionality should not affect inference
        call_inference(model_name, "http", self.client_http)
        call_inference(model_name, "grpc", self.client_grpc)

    def test_is_model_ready_takes_longs_time(self):
        model_name = "is_model_ready_fn_timeout"

        # Send many sequential requests to ensure consistent behavior
        for i in range(10):
            # Should timeout and return NOT ready
            # Note: Stub will still continue execution in background
            # even though request timeout after in built python backend 5s timeout for ready request
            is_ready = self.client_http.is_model_ready(model_name)
            self.assertFalse(
                is_ready, f"Model {model_name} should timeout and return NOT READY"
            )

            # Wait for stub to complete the execution in previous call
            time.sleep(6)
            call_inference(model_name, "http", self.client_http)

            is_ready = self.client_grpc.is_model_ready(model_name)
            self.assertFalse(
                is_ready, f"Model {model_name} should timeout and be NOT READY"
            )

            # Wait again before calling inference
            time.sleep(6)
            call_inference(model_name, "grpc", self.client_grpc)

    def test_multiple_concurrent_ready_and_infer_requests(self):
        model_name = "is_model_ready_fn_returns_true"
        ready_results = {"http": [], "grpc": []}
        ready_errors = {"http": [], "grpc": []}
        infer_results = {"http": [], "grpc": []}
        infer_errors = {"http": [], "grpc": []}

        def check_model_readiness(protocol, index):
            try:
                if protocol == "http":
                    with httpclient.InferenceServerClient(url=URL_HTTP) as client_http:
                        is_ready = client_http.is_model_ready(model_name)
                        ready_results["http"].append((index, is_ready))
                else:
                    with grpcclient.InferenceServerClient(url=URL_GRPC) as client_grpc:
                        is_ready = client_grpc.is_model_ready(model_name)
                        ready_results["grpc"].append((index, is_ready))
            except Exception as e:
                ready_errors[protocol].append((index, str(e)))

        def do_inference(protocol, index):
            try:
                if protocol == "http":
                    with httpclient.InferenceServerClient(url=URL_HTTP) as client_http:
                        start = time.time()
                        call_inference(model_name, protocol, client_http)
                        elapsed = time.time() - start
                        infer_results["http"].append((index, True, elapsed))
                else:
                    with grpcclient.InferenceServerClient(url=URL_GRPC) as client_grpc:
                        start = time.time()
                        call_inference(model_name, protocol, client_grpc)
                        elapsed = time.time() - start
                        infer_results["grpc"].append((index, True, elapsed))
            except Exception as e:
                infer_errors[protocol].append((index, str(e)))

        # Launch 10 concurrent ready checks
        threads = []
        for i in range(5):
            t1 = threading.Thread(target=check_model_readiness, args=("http", i))
            t2 = threading.Thread(target=check_model_readiness, args=("grpc", i))
            t3 = threading.Thread(target=do_inference, args=("http", i))
            t4 = threading.Thread(target=do_inference, args=("grpc", i))
            threads.extend([t1, t2, t3, t4])
            t1.start()
            t2.start()
            t3.start()
            t4.start()

        # Wait for all requests to complete
        for t in threads:
            t.join(timeout=30)

        # Verify all succeeded
        self.assertEqual(
            len(ready_errors["http"]), 0, f"HTTP errors: {ready_errors['http']}"
        )
        self.assertEqual(
            len(ready_errors["grpc"]), 0, f"gRPC errors: {ready_errors['grpc']}"
        )
        self.assertEqual(len(ready_results["http"]), 5, "Expected 5 HTTP results")
        self.assertEqual(len(ready_results["grpc"]), 5, "Expected 5 gRPC results")

        # All should be True
        for idx, ready in ready_results["http"]:
            self.assertTrue(ready, f"HTTP check {idx} should be ready")
        for idx, ready in ready_results["grpc"]:
            self.assertTrue(ready, f"gRPC check {idx} should be ready")

        # Verify no errors
        self.assertEqual(
            len(infer_errors["http"]), 0, f"Errors occurred: {infer_errors['http']}"
        )
        self.assertEqual(
            len(infer_errors["grpc"]), 0, f"Errors occurred: {infer_errors['grpc']}"
        )
        self.assertEqual(
            len(infer_results["http"]), 5, "Expected 5 HTTP inference results"
        )
        self.assertEqual(
            len(infer_results["grpc"]), 5, "Expected 5 gRPC inference results"
        )

        for idx, success, elapsed in infer_results["http"]:
            print(
                f"Inference http[{idx}]: {'OK' if success else 'FAIL'} ({elapsed:.3f}s)"
            )
        for idx, success, elapsed in infer_results["grpc"]:
            print(
                f"Inference grpc[{idx}]: {'OK' if success else 'FAIL'} ({elapsed:.3f}s)"
            )

if __name__ == "__main__":
    unittest.main()
