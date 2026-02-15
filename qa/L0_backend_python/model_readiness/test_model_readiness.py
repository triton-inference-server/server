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
import queue
import time
import threading
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from functools import partial

URL_HTTP = "localhost:8000"
URL_GRPC = "localhost:8001"
DEFAULT_RESPONSE_TIMEOUT = 60


class UserData:
    def __init__(self):
        self._response_queue = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._response_queue.put(error)
    else:
        user_data._response_queue.put(result)


def prepare_infer_args(input_value):
    """
    Create InferInput/InferRequestedOutput lists
    """
    input_data = np.array([[input_value]], dtype=np.int32)
    infer_input = [grpcclient.InferInput("IN", input_data.shape, "INT32")]
    infer_input[0].set_data_from_numpy(input_data)
    outputs = [grpcclient.InferRequestedOutput("OUT")]
    return infer_input, outputs


def collect_responses(user_data, expected_responses_count):
    """
    Collect responses from user_data until the final response flag is seen.
    """
    errors = []
    responses = []
    recv_count = 0
    while recv_count < expected_responses_count:
        try:
            result = user_data._response_queue.get(timeout=DEFAULT_RESPONSE_TIMEOUT)
        except queue.Empty:
            raise Exception(
                f"No response received within {DEFAULT_RESPONSE_TIMEOUT} seconds."
            )
        if type(result) == InferenceServerException:
            errors.append(result)
            break
        else:
            responses.append(result.as_numpy("OUT")[0])
        recv_count = recv_count + 1

    return errors, responses


def call_inference_identity_model(model_name, protocol, client):
    """Helper to test inference functionality"""
    shape = (1, 8)
    input_data = np.ones(shape, dtype=np.float32)

    if protocol == "http":
        inputs = [httpclient.InferInput("INPUT0", input_data.shape, "FP32")]
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
            call_inference_identity_model(self.model_name, "http", self.client_http)
        except Exception as e:
            self.fail(f"[HTTP] Unexpected error: {str(e)}")

        # Check gRPC
        try:
            is_ready = self.client_grpc.is_model_ready(self.model_name)
            self.assertTrue(
                is_ready, f"[gRPC] Model {self.model_name} should be READY but is NOT"
            )
            call_inference_identity_model(self.model_name, "grpc", self.client_grpc)
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

    def _run_inference_decoupled(self, index, model_name, expected_responses_count):
        """
        Helper function for streaming inference.
        """
        user_data = UserData()
        with grpcclient.InferenceServerClient(URL_GRPC) as triton_client:
            try:
                inputs, outputs = prepare_infer_args(expected_responses_count)
                triton_client.start_stream(callback=partial(callback, user_data))
                triton_client.async_stream_infer(
                    model_name=model_name, inputs=inputs, outputs=outputs
                )

                # Collect and verify responses
                errors, responses = collect_responses(
                    user_data, expected_responses_count
                )
                self.assertEqual(
                    len(responses),
                    expected_responses_count,
                    f"Index: {index} - Expected {expected_responses_count} responses, got {len(responses)}",
                )
                self.assertEqual(
                    len(errors),
                    0,
                    f"Index: {index} - Expected 0 errors, got {len(errors)}",
                )

                # Verify correctness of successful responses
                for idx, output in enumerate(responses):
                    self.assertEqual(
                        output,
                        expected_responses_count,
                        msg=f"Response {idx} has incorrect value - {output}",
                    )
            finally:
                triton_client.stop_stream()

    def test_multiple_concurrent_ready_and_infer_requests_decoupled(self):
        model_name = "is_model_ready_fn_returns_true_decoupled"
        num_requests = 16
        response_count = num_requests
        readiness_errors = []
        infer_errors = []

        def readiness_wrapper(index, model_name):
            try:
                with grpcclient.InferenceServerClient(url=URL_GRPC) as triton_client:
                    is_ready = triton_client.is_model_ready(model_name)
                    if not is_ready:
                        raise AssertionError(
                            f"Index: {index} - GRPC client - Model {model_name} should be READY"
                        )
            except Exception as e:
                readiness_errors.append((index, str(e)))

        def inference_wrapper(index, model_name):
            try:
                self._run_inference_decoupled(index, model_name, response_count)
            except Exception as e:
                infer_errors.append((index, str(e)))

        # Launch concurrent threads
        threads = []
        for i in range(num_requests):
            # Start threads with slight delay
            time.sleep(0.1)
            t1 = threading.Thread(
                target=inference_wrapper, args=(i, model_name), name=f"infer-{i}"
            )
            t2 = threading.Thread(
                target=readiness_wrapper, args=(i, model_name), name=f"ready-{i}"
            )
            threads.extend([t1, t2])
            t1.start()
            t2.start()

        # Wait for all requests to complete
        for t in threads:
            t.join(timeout=120)

        for t in threads:
            self.assertFalse(t.is_alive(), f"Threads are not completed: {t.name}")

        self.assertEqual(
            len(readiness_errors), 0, f"Readiness errors: {readiness_errors}"
        )
        self.assertEqual(len(infer_errors), 0, f"Inference errors: {infer_errors}")

    def test_is_model_ready_coroutine_returns_true(self):
        model_name = "is_model_ready_fn_coroutine_returns_true"
        for _ in range(5):
            self.assertTrue(
                self.client_http.is_model_ready(model_name),
                f"HTTP - Model {model_name} (coroutine) should be READY",
            )
            self.assertTrue(
                self.client_grpc.is_model_ready(model_name),
                f"gRPC - Model {model_name} (coroutine) should be READY",
            )
        call_inference_identity_model(model_name, "http", self.client_http)
        call_inference_identity_model(model_name, "grpc", self.client_grpc)

    def test_is_model_ready_returns_true(self):
        model_name = "is_model_ready_fn_returns_true"
        num_requests = 10

        # Send multiple requests in sequence to ensure consistent behavior
        for i in range(num_requests):
            self.assertTrue(
                self.client_http.is_model_ready(model_name),
                f"iteration {i} - HTTP client - Model {model_name} should be READY",
            )
            self.assertTrue(
                self.client_grpc.is_model_ready(model_name),
                f"iteration {i} - GRPC client - Model {model_name} should be READY",
            )

            # Inference should work normally
            # readiness check functionality should not affect inference
            call_inference_identity_model(model_name, "http", self.client_http)
            call_inference_identity_model(model_name, "grpc", self.client_grpc)

    def test_is_model_ready_returns_false(self):
        model_name = "is_model_ready_fn_returns_false"
        num_requests = 10

        # Send multiple requests in sequence to ensure consistent behavior
        for i in range(num_requests):
            self.assertFalse(
                self.client_http.is_model_ready(model_name),
                f"iteration {i} - HTTP client - Model {model_name} should be NOT READY",
            )
            self.assertFalse(
                self.client_grpc.is_model_ready(model_name),
                f"iteration {i} - GRPC client - Model {model_name} should be NOT READY",
            )

            # Inference should work normally
            # readiness check functionality should not affect inference
            call_inference_identity_model(model_name, "http", self.client_http)
            call_inference_identity_model(model_name, "grpc", self.client_grpc)

    def test_is_model_ready_raises_exception(self):
        model_name = "is_model_ready_fn_raises_error"
        num_requests = 10

        # Send multiple requests in sequence to ensure consistent behavior
        for i in range(num_requests):
            self.assertFalse(
                self.client_http.is_model_ready(model_name),
                f"iteration {i} - HTTP client - Model {model_name} should be NOT READY (exception)",
            )
            self.assertFalse(
                self.client_grpc.is_model_ready(model_name),
                f"iteration {i} - GRPC client - Model {model_name} should be NOT READY (exception)",
            )

            # Inference should work normally
            # readiness check functionality should not affect inference
            call_inference_identity_model(model_name, "http", self.client_http)
            call_inference_identity_model(model_name, "grpc", self.client_grpc)

        # Test good model afterwards to ensure server is healthy
        model_name = "is_model_ready_fn_returns_true"
        for i in range(num_requests):
            self.assertTrue(
                self.client_http.is_model_ready(model_name),
                f"iteration {i} - HTTP client - Model {model_name} should be READY",
            )
            self.assertTrue(
                self.client_grpc.is_model_ready(model_name),
                f"iteration {i} - GRPC client - Model {model_name} should be READY",
            )

            # Inference should work normally
            # readiness check functionality should not affect inference
            call_inference_identity_model(model_name, "http", self.client_http)
            call_inference_identity_model(model_name, "grpc", self.client_grpc)

    def test_is_model_ready_returns_non_boolean(self):
        model_name = "is_model_ready_fn_returns_non_boolean"
        num_requests = 10

        # Send multiple requests in sequence to ensure consistent behavior
        for i in range(num_requests):
            self.assertFalse(
                self.client_http.is_model_ready(model_name),
                f"iteration {i} - HTTP client - Model {model_name} should be NOT READY (wrong return type)",
            )
            self.assertFalse(
                self.client_grpc.is_model_ready(model_name),
                f"iteration {i} - GRPC client - Model {model_name} should be NOT READY (wrong return type)",
            )

            # Inference should work normally
            # readiness check functionality should not affect inference
            call_inference_identity_model(model_name, "http", self.client_http)
            call_inference_identity_model(model_name, "grpc", self.client_grpc)

        # Test good model afterwards to ensure server is healthy
        model_name = "is_model_ready_fn_returns_true"
        for i in range(num_requests):
            self.assertTrue(
                self.client_http.is_model_ready(model_name),
                f"iteration {i} - HTTP client - Model {model_name} should be READY",
            )
            self.assertTrue(
                self.client_grpc.is_model_ready(model_name),
                f"iteration {i} - GRPC client - Model {model_name} should be READY",
            )

            # Inference should work normally
            # readiness check functionality should not affect inference
            call_inference_identity_model(model_name, "http", self.client_http)
            call_inference_identity_model(model_name, "grpc", self.client_grpc)

    def test_is_model_ready_takes_longs_time(self):
        model_name = "is_model_ready_fn_timeout"
        num_requests = 10

        # Send multiple requests in sequence to ensure consistent behavior
        for i in range(num_requests):
            # This call should time out and return NOT_READY.
            # Note: the stub will continue running is_model_ready()
            # in the background (similar to the inference flow)
            # even after the backend readiness timeout expires.
            is_ready = self.client_http.is_model_ready(model_name)
            self.assertFalse(
                is_ready,
                f"iteration {i} - HTTP client - Model {model_name} should timeout and return NOT READY",
            )

            call_inference_identity_model(model_name, "http", self.client_http)

            # This call should not create another internal IPC message.
            # It must wait for the in-flight readiness check
            # and return READY once that check completes.
            is_ready = self.client_grpc.is_model_ready(model_name)
            self.assertTrue(
                is_ready,
                f"iteration {i} - GRPC client - Model {model_name} should be READY",
            )

            call_inference_identity_model(model_name, "grpc", self.client_grpc)

    def test_multiple_concurrent_ready_and_infer_requests(self):
        model_name = "is_model_ready_fn_returns_true"
        ready_results = {"http": [], "grpc": []}
        ready_errors = {"http": [], "grpc": []}
        infer_results = {"http": [], "grpc": []}
        infer_errors = {"http": [], "grpc": []}
        num_requests = 16

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
                        call_inference_identity_model(model_name, protocol, client_http)
                        elapsed = time.time() - start
                        infer_results["http"].append((index, True, elapsed))
                else:
                    with grpcclient.InferenceServerClient(url=URL_GRPC) as client_grpc:
                        start = time.time()
                        call_inference_identity_model(model_name, protocol, client_grpc)
                        elapsed = time.time() - start
                        infer_results["grpc"].append((index, True, elapsed))
            except Exception as e:
                infer_errors[protocol].append((index, str(e)))

        # Launch concurrent ready checks
        http_threads = []
        for i in range(num_requests):
            t1 = threading.Thread(target=check_model_readiness, args=("http", i))
            t2 = threading.Thread(target=do_inference, args=("http", i))
            http_threads.extend([t1, t2])
            t1.start()
            t2.start()

        # Wait for all requests to complete
        for t in http_threads:
            t.join(timeout=60)

        for t in http_threads:
            self.assertFalse(t.is_alive(), f"HTTP threads are not completed")

        time.sleep(5)

        grpc_threads = []
        for i in range(num_requests):
            t1 = threading.Thread(target=check_model_readiness, args=("grpc", i))
            t2 = threading.Thread(target=do_inference, args=("grpc", i))
            grpc_threads.extend([t1, t2])
            t1.start()
            t2.start()

        # Wait for all requests to complete
        for t in grpc_threads:
            t.join(timeout=60)

        for t in grpc_threads:
            self.assertFalse(t.is_alive(), f"gRPC threads are not completed")

        time.sleep(5)

        # Verify no errors in readiness checks
        self.assertEqual(
            len(ready_errors["http"]), 0, f"HTTP errors: {ready_errors['http']}"
        )
        self.assertEqual(
            len(ready_errors["grpc"]), 0, f"gRPC errors: {ready_errors['grpc']}"
        )
        self.assertEqual(
            len(ready_results["http"]),
            num_requests,
            f"Expected {num_requests} HTTP results",
        )
        self.assertEqual(
            len(ready_results["grpc"]),
            num_requests,
            f"Expected {num_requests} gRPC results",
        )

        # All should be True
        for idx, ready in ready_results["http"]:
            self.assertTrue(ready, f"HTTP check {idx} should be ready")
        for idx, ready in ready_results["grpc"]:
            self.assertTrue(ready, f"gRPC check {idx} should be ready")

        # Verify no errors in inference
        self.assertEqual(
            len(infer_errors["http"]), 0, f"Errors occurred: {infer_errors['http']}"
        )
        self.assertEqual(
            len(infer_errors["grpc"]), 0, f"Errors occurred: {infer_errors['grpc']}"
        )
        self.assertEqual(
            len(infer_results["http"]),
            num_requests,
            f"Expected {num_requests} HTTP inference results",
        )
        self.assertEqual(
            len(infer_results["grpc"]),
            num_requests,
            f"Expected {num_requests} gRPC inference results",
        )


if __name__ == "__main__":
    unittest.main()
