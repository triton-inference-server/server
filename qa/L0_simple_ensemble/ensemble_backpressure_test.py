#!/usr/bin/env python3

# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
MODEL_ENSEMBLE_ENABLED = "ensemble_enabled_max_inflight_responses"
MODEL_ENSEMBLE_DISABLED = "ensemble_disabled_max_inflight_responses"


class UserData:
    def __init__(self):
        self._response_queue = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._response_queue.put(error)
    else:
        user_data._response_queue.put(result)


class EnsembleBackpressureTest(tu.TestResultCollector):
    """
    Tests for ensemble backpressure feature (max_inflight_responses).
    """

    def _prepare_infer_args(self, input_value):
        """
        Create InferInput/InferRequestedOutput lists
        """
        input_data = np.array([input_value], dtype=np.int32)
        infer_input = [grpcclient.InferInput("IN", input_data.shape, "INT32")]
        infer_input[0].set_data_from_numpy(input_data)
        outputs = [grpcclient.InferRequestedOutput("OUT")]
        return infer_input, outputs

    def _collect_responses(self, user_data):
        """
        Collect responses from user_data until the final response flag is seen.
        """
        responses = []
        while True:
            try:
                result = user_data._response_queue.get(timeout=DEFAULT_RESPONSE_TIMEOUT)
            except queue.Empty:
                self.fail(
                    f"No response received within {DEFAULT_RESPONSE_TIMEOUT} seconds."
                )

            self.assertNotIsInstance(
                result, Exception, f"Callback returned an exception: {result}"
            )

            response = result.get_response()
            # Add response to list if it has data (not empty final-only response)
            if len(response.outputs) > 0:
                responses.append(result)

            # Check if this is the final response
            final = response.parameters.get("triton_final_response")
            if final and final.bool_param:
                break

        return responses

    def _run_inference(self, model_name, expected_count):
        """
        Helper function to run inference and verify responses.
        """
        user_data = UserData()
        with grpcclient.InferenceServerClient(SERVER_URL) as triton_client:
            try:
                inputs, outputs = self._prepare_infer_args(expected_count)
                triton_client.start_stream(callback=partial(callback, user_data))
                triton_client.async_stream_infer(
                    model_name=model_name, inputs=inputs, outputs=outputs
                )

                # Collect and verify responses
                responses = self._collect_responses(user_data)
                self.assertEqual(
                    len(responses),
                    expected_count,
                    f"Expected {expected_count} responses, got {len(responses)}",
                )

                # Verify correctness of responses
                for idx, resp in enumerate(responses):
                    output = resp.as_numpy("OUT")
                    self.assertEqual(
                        output[0],
                        EXPECTED_INFER_OUTPUT,
                        msg=f"Response {idx} has  - {output[0]}",
                    )
            finally:
                triton_client.stop_stream()

    def test_backpressure_limits_inflight(self):
        """
        Test that max_inflight_responses correctly limits concurrent
        responses.
        """
        self._run_inference(model_name=MODEL_ENSEMBLE_ENABLED, expected_count=32)

    def test_backpressure_disabled(self):
        """
        Test that an ensemble model without max_inflight_responses parameter works correctly.
        """
        self._run_inference(model_name=MODEL_ENSEMBLE_DISABLED, expected_count=32)

    def test_backpressure_concurrent_requests(self):
        """
        Test that backpressure works correctly with multiple concurrent requests.
        Each request should have independent backpressure state.
        """
        num_concurrent = 8
        expected_per_request = 8
        user_datas = [UserData() for _ in range(num_concurrent)]

        with ExitStack() as stack:
            clients = [
                stack.enter_context(grpcclient.InferenceServerClient(SERVER_URL))
                for _ in range(num_concurrent)
            ]

            inputs, outputs = self._prepare_infer_args(expected_per_request)

            # Start all concurrent requests
            for i in range(num_concurrent):
                clients[i].start_stream(callback=partial(callback, user_datas[i]))
                clients[i].async_stream_infer(
                    model_name=MODEL_ENSEMBLE_ENABLED, inputs=inputs, outputs=outputs
                )

            # Collect and verify responses for all requests
            for i, ud in enumerate(user_datas):
                responses = self._collect_responses(ud)
                self.assertEqual(
                    len(responses),
                    expected_per_request,
                    f"Request {i}: expected {expected_per_request} responses, got {len(responses)}",
                )
                # Verify correctness of responses
                for idx, resp in enumerate(responses):
                    output = resp.as_numpy("OUT")
                    self.assertEqual(
                        output[0],
                        EXPECTED_INFER_OUTPUT,
                        msg=f"Response {idx} for request {i} has incorrect value - {output[0]}",
                    )

            # Stop all streams
            for client in clients:
                client.stop_stream()

    def test_backpressure_request_cancellation(self):
        """
        Test that cancellation unblocks producers waiting on backpressure and that
        the client receives a cancellation error.
        """
        # Use a large count to ensure the producer gets blocked by backpressure.
        # The model is configured with max_inflight_responses = 4.
        input_value = 32
        user_data = UserData()

        with grpcclient.InferenceServerClient(SERVER_URL) as triton_client:
            inputs, outputs = self._prepare_infer_args(input_value)
            triton_client.start_stream(callback=partial(callback, user_data))

            # Start the request
            triton_client.async_stream_infer(
                model_name=MODEL_ENSEMBLE_ENABLED, inputs=inputs, outputs=outputs
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


if __name__ == "__main__":
    unittest.main()
