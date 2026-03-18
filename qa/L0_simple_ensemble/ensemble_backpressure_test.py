#!/usr/bin/env python3

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

import sys

sys.path.append("../common")

import queue
import threading
import time
import unittest
from contextlib import ExitStack
from functools import partial

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

SERVER_URL = "localhost:8001"
DEFAULT_RESPONSE_TIMEOUT = 60
EXPECTED_INFER_OUTPUT = 0.5

NUM_REQUESTS = 16
NUM_RESPONSES_PER_REQUEST = 8


class UserData:
    def __init__(self):
        self._response_queue = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._response_queue.put(error)
    else:
        user_data._response_queue.put(result)


def prepare_infer_args(input_value, enable_batching=False):
    """
    Create InferInput/InferRequestedOutput lists
    """
    if enable_batching:
        input_data = np.array([[input_value]], dtype=np.int32)
    else:
        input_data = np.array([input_value], dtype=np.int32)
    infer_input = [grpcclient.InferInput("IN", input_data.shape, "INT32")]
    infer_input[0].set_data_from_numpy(input_data)
    outputs = [grpcclient.InferRequestedOutput("OUT")]
    return infer_input, outputs


def collect_responses(user_data):
    """
    Collect responses from user_data until the final response flag is seen.
    """
    errors = []
    responses = []
    while True:
        try:
            result = user_data._response_queue.get(timeout=DEFAULT_RESPONSE_TIMEOUT)
        except queue.Empty:
            raise Exception(
                f"No response received within {DEFAULT_RESPONSE_TIMEOUT} seconds."
            )

        if isinstance(result, InferenceServerException):
            errors.append(result)
            # error responses are final - stream terminates
            break

        response = result.get_response()
        # Add response to list if it has data (not empty final-only response)
        if len(response.outputs) > 0:
            responses.append(result)

        # Check if this is the final response
        final = response.parameters.get("triton_final_response")
        if final and final.bool_param:
            break

    return errors, responses


class EnsembleBackpressureTest(tu.TestResultCollector):
    """
    Tests for ensemble backpressure feature (max_inflight_requests).
    """

    def _run_inference(
        self, model_name, expected_responses_per_request, num_concurrent_requests=1
    ):
        """
        Send num_concurrent_requests streaming requests to model_name, each expecting
        expected_responses_per_request responses. Verify all complete with correct data.
        """
        user_datas = [UserData() for _ in range(num_concurrent_requests)]

        with ExitStack() as stack:
            clients = [
                stack.enter_context(grpcclient.InferenceServerClient(SERVER_URL))
                for _ in range(num_concurrent_requests)
            ]

            inputs, outputs = prepare_infer_args(expected_responses_per_request, True)

            # Start all concurrent requests
            for i in range(num_concurrent_requests):
                clients[i].start_stream(callback=partial(callback, user_datas[i]))
                clients[i].async_stream_infer(
                    model_name=model_name, inputs=inputs, outputs=outputs
                )

            # Collect and verify responses for all requests
            for i, ud in enumerate(user_datas):
                errors, responses = collect_responses(ud)
                self.assertEqual(
                    len(responses),
                    expected_responses_per_request,
                    f"Request {i}: expected {expected_responses_per_request} responses, got {len(responses)}",
                )
                self.assertEqual(
                    len(errors), 0, f"Request {i}: unexpected errors: {errors}"
                )
                # Verify correctness of responses
                for idx, resp in enumerate(responses):
                    output = resp.as_numpy("OUT")
                    self.assertAlmostEqual(
                        output[0],
                        EXPECTED_INFER_OUTPUT,
                        places=5,
                        msg=f"Request {i} response {idx}: expected "
                        f"{EXPECTED_INFER_OUTPUT}, got {output[0]}",
                    )

            # Stop all streams
            for client in clients:
                client.stop_stream()

    def test_single_request_with_different_limits(self):
        """
        Single streaming request producing 16 responses through a 3-step
        ensemble (decoupled_producer -> consumer_high_delay -> consumer_low_delay)
        under various max_inflight_requests settings.
        """
        cases = [
            ("ensemble_limit_4", "max_inflight_requests=4"),
            ("ensemble_limit_1", "max_inflight_requests=1"),
            ("ensemble_disabled", "max_inflight_requests is disabled"),
        ]
        for model_name, desc in cases:
            with self.subTest(limit=desc):
                self._run_inference(
                    model_name=model_name, expected_responses_per_request=16
                )

    def test_concurrent_requests_across_topologies(self):
        """
        NUM_REQUESTS concurrent streaming requests (NUM_RESPONSES_PER_REQUEST
        responses each) exercise the global max_inflight_requests limit.
        Subtests cover: limit=4, limit=1, and the limit disabled.
        """
        cases = [
            ("ensemble_limit_4", "max_inflight_requests=4"),
            ("ensemble_limit_1", "max_inflight_requests=1"),
            ("ensemble_disabled", "max_inflight_requests is disabled"),
        ]
        for model_name, desc in cases:
            with self.subTest(topology=desc):
                self._run_inference(
                    model_name=model_name,
                    expected_responses_per_request=NUM_RESPONSES_PER_REQUEST,
                    num_concurrent_requests=NUM_REQUESTS,
                )

    def test_sequential_requests_limiter_resets_cleanly(self):
        """
        Send NUM_REQUESTS sequential requests one after another. If the limiter
        leaks a slot on any request, subsequent requests will deadlock or time out.
        """
        for seq_idx in range(NUM_REQUESTS):
            with self.subTest(request=seq_idx):
                self._run_inference(
                    model_name="ensemble_limit_4",
                    expected_responses_per_request=NUM_RESPONSES_PER_REQUEST,
                )

    def test_request_cancellation_under_backpressure(self):
        """
        Start a long-running request (32 responses), cancel mid-stream,
        and verify the server sends a CANCELLED status and only a partial set of
        responses is received.
        """
        input_value = 32
        user_data = UserData()

        with grpcclient.InferenceServerClient(SERVER_URL) as triton_client:
            inputs, outputs = prepare_infer_args(input_value, True)
            triton_client.start_stream(callback=partial(callback, user_data))

            # Start the request
            triton_client.async_stream_infer(
                model_name="ensemble_limit_4", inputs=inputs, outputs=outputs
            )

            responses = []
            try:
                result = user_data._response_queue.get(timeout=5)
                if isinstance(result, InferenceServerException):
                    self.fail(f"Got error before cancellation: {result}")
                resp = result.get_response()
                if len(resp.outputs) > 0:
                    responses.append(result)
            except queue.Empty:
                self.fail("Stream did not produce any response before cancellation.")

            # Cancel the stream. This should unblock any waiting producers and result in a CANCELLED error.
            triton_client.stop_stream(cancel_requests=True)

            # Allow some time for cancellation
            time.sleep(1)

            cancellation_found = False
            while True:
                try:
                    result = user_data._response_queue.get(timeout=1)
                    if isinstance(result, InferenceServerException):
                        self.assertEqual(
                            result.status(),
                            "StatusCode.CANCELLED",
                            f"Expected CANCELLED status, got: {result.status()}",
                        )
                        cancellation_found = True
                        break
                    else:
                        response = result.get_response()
                        if len(response.outputs) > 0:
                            responses.append(result)
                        # Check for final response
                        final = response.parameters.get("triton_final_response")
                        if final and final.bool_param:
                            break
                except queue.Empty:
                    break

            # Verify the cancellation error was received
            self.assertTrue(
                cancellation_found,
                "Did not receive the expected cancellation error from the server.",
            )

            # Verify we received only a partial set of responses
            self.assertLess(
                len(responses),
                input_value,
                "Expected partial responses due to cancellation, but received all of them.",
            )
            self.assertGreater(
                len(responses),
                0,
                "Expected to receive at least one response before cancellation.",
            )


class EnsembleStepMaxQueueSizeTest(tu.TestResultCollector):
    def _run_inference(self, model_name, expected_responses_count):
        """
        Helper function for streaming inference.

        For decoupled streaming ensembles with queue limit on internal step:
        - Each producer response creates an independent flow through the ensemble
        - Flows that complete before error is set send their outputs successfully
        - Once error occurs (queue full), stream terminates with error
        - Result: 0-N successful responses + 1 error (N depends on timing)
        """
        user_data = UserData()
        with grpcclient.InferenceServerClient(SERVER_URL) as triton_client:
            try:
                inputs, outputs = prepare_infer_args(expected_responses_count)
                triton_client.start_stream(callback=partial(callback, user_data))
                triton_client.async_stream_infer(
                    model_name=model_name, inputs=inputs, outputs=outputs
                )

                # Collect and verify responses
                errors, responses = collect_responses(user_data)
                self.assertGreaterEqual(
                    len(responses),
                    0,
                    "May have 0 or more successful responses depending on timing",
                )
                self.assertLess(
                    len(responses),
                    expected_responses_count,
                    f"Should have fewer than {expected_responses_count} responses (some flows failed)",
                )
                self.assertEqual(
                    len(errors),
                    1,
                    "Expected exactly one error when queue full terminates stream",
                )

                # Verify correctness of successful responses
                for idx, resp in enumerate(responses):
                    output = resp.as_numpy("OUT")
                    self.assertAlmostEqual(
                        output[0],
                        EXPECTED_INFER_OUTPUT,
                        places=5,
                        msg=f"Response {idx} has incorrect value - {output[0]}",
                    )

                # Verify error is queue-full error
                self.assertIn(
                    "Exceeds maximum queue size",
                    str(errors[0]),
                    f"Expected queue size error, got: {str(errors[0])}",
                )
            finally:
                triton_client.stop_stream()

    def _run_concurrent_inference(self, model_name, expected_responses_count):
        """
        Helper function for concurrent independent requests.
        Each request either succeeds completely or fails completely.
        Returns: (num_successes, num_errors) tuple
        """
        user_data = UserData()
        with grpcclient.InferenceServerClient(SERVER_URL) as triton_client:
            try:
                inputs, outputs = prepare_infer_args(expected_responses_count)
                triton_client.start_stream(callback=partial(callback, user_data))
                triton_client.async_stream_infer(
                    model_name=model_name, inputs=inputs, outputs=outputs
                )

                # Collect responses
                errors, responses = collect_responses(user_data)

                # For concurrent independent requests with queue limit on internal step:
                # - Requests that arrive before queue fills: succeed with all outputs
                # - Requests that arrive after queue fills: fail with error
                total = len(responses) + len(errors)
                self.assertEqual(
                    total,
                    expected_responses_count,
                    f"Expected {expected_responses_count} total responses, got {total}",
                )

                if len(errors) > 0:
                    # This request failed
                    self.assertEqual(
                        len(responses),
                        0,
                        "Failed request should have no successful outputs",
                    )
                    self.assertEqual(
                        len(errors), 1, "Failed request should have exactly one error"
                    )
                    self.assertIn(
                        "Exceeds maximum queue size",
                        str(errors[0]),
                        f"Expected queue size error, got: {str(errors[0])}",
                    )
                    return (0, 1)  # 0 successes, 1 error
                else:
                    # This request succeeded
                    self.assertEqual(
                        len(responses),
                        expected_responses_count,
                        f"Successful request should have all {expected_responses_count} outputs",
                    )
                    # Verify correctness of successful responses
                    for idx, resp in enumerate(responses):
                        output = resp.as_numpy("OUT")
                        self.assertAlmostEqual(
                            output[0],
                            EXPECTED_INFER_OUTPUT,
                            places=5,
                            msg=f"Response {idx} has incorrect value - {output[0]}",
                        )
                    return (expected_responses_count, 0)  # N successes, 0 errors
            finally:
                triton_client.stop_stream()

    def test_step1_max_queue_size(self):
        """
        Test max_queue_size on step 1 (decoupled_producer).

        Trigger 32 concurrent ensemble requests, each producing 1 response
        - Step 1 (producer) has max_queue_size limit
        - Some ensemble requests succeed completely (before queue fills)
        - Some fail completely (when producer queue is full)
        """
        model_name = "ensemble_step1_enabled_max_queue_size"
        num_requests = 32

        # Store results from each thread
        results = []

        def thread_wrapper(model_name, expected_count, results_list):
            """Wrapper to capture thread results"""
            result = self._run_concurrent_inference(model_name, expected_count)
            results_list.append(result)

        # Launch concurrent threads to perform infer requests
        threads = []
        for i in range(num_requests):
            t = threading.Thread(target=thread_wrapper, args=(model_name, 1, results))
            threads.append(t)
            t.start()

        # Wait for all requests to complete
        for t in threads:
            t.join(timeout=60)

        # Aggregate results from all threads
        total_successes = sum(r[0] for r in results)
        total_errors = sum(r[1] for r in results)

        # Verify aggregate behavior
        self.assertEqual(
            total_successes + total_errors,
            num_requests,
            f"Expected {num_requests} total results (successes + errors), "
            f"got {total_successes} successes + {total_errors} errors = {total_successes + total_errors}",
        )

        # Verify at least some errors occurred (queue limit was hit)
        self.assertGreater(
            total_errors,
            0,
            f"Expected some errors due to max_queue_size limit, "
            f"but all {num_requests} requests succeeded.",
        )

        # Verify at least some successes occurred (not all rejected)
        self.assertGreater(
            total_successes,
            0,
            f"Expected some successful requests before queue filled, "
            f"but all {num_requests} requests failed.",
        )

    def test_step2_max_queue_size(self):
        """
        Test max_queue_size on step 2 (slow_consumer).

        Trigger 1 streaming ensemble request producing 32 responses
        - Step 1 (producer) generates 32 responses rapidly (every 100ms)
        - Step 2 (consumer) has max_queue_size=5 and processes slowly (500ms each)
        - Each producer response is an independent request to the second step through
        - the ensemble flow. Some requests complete successfully before queue fills
        - When queue fills, error is set and stream terminates
        - All inflight steps drain, then error response sent to client
        """
        model_name = "ensemble_step2_enabled_max_queue_size"
        self._run_inference(model_name=model_name, expected_responses_count=32)


if __name__ == "__main__":
    unittest.main()
