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
import unittest
from functools import partial

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient

SERVER_URL = "localhost:8001"
DEFAULT_RESPONSE_TIMEOUT = 60


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
    Tests for ensemble backpressure feature (max_ensemble_inflight_responses).
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

            # Add response to list if it has data (not empty final-only response)
            response = result.get_response()
            if len(response.outputs) > 0:
                responses.append(result)

            # Check if this is the final response
            final = response.parameters.get("triton_final_response")
            if final and final.bool_param:
                break

        return responses

    def test_backpressure_limits_inflight(self):
        """
        Test that max_ensemble_inflight_responses correctly limits concurrent
        responses and prevents unbounded memory growth.
        """
        model_name = "ensemble_enabled_max_inflight_responses"
        expected_count = 16
        user_data = UserData()

        with grpcclient.InferenceServerClient(SERVER_URL) as triton_client:
            try:
                inputs, outputs = self._prepare_infer_args(expected_count)

                triton_client.start_stream(callback=partial(callback, user_data))

                triton_client.async_stream_infer(
                    model_name=model_name, inputs=inputs, outputs=outputs
                )

                # Collect responses
                responses = self._collect_responses(user_data)

                # Verify we received the expected number of responses
                self.assertEqual(
                    len(responses),
                    expected_count,
                    f"Expected {expected_count} responses, got {len(responses)}",
                )

                # Verify correctness of responses
                for idx, resp in enumerate(responses):
                    output = resp.as_numpy("OUT")
                    self.assertEqual(
                        output[0], idx, f"Response {idx} has incorrect value"
                    )

            finally:
                triton_client.stop_stream()

    def test_backpressure_disabled(self):
        """
        Test that ensemble model without max_ensemble_inflight_responses parameter
        works fine (original behavior).
        """
        model_name = "ensemble_disabled_max_inflight_responses"
        expected_count = 16
        user_data = UserData()

        with grpcclient.InferenceServerClient(SERVER_URL) as triton_client:
            try:
                inputs, outputs = self._prepare_infer_args(expected_count)

                triton_client.start_stream(callback=partial(callback, user_data))

                triton_client.async_stream_infer(
                    model_name=model_name, inputs=inputs, outputs=outputs
                )

                # Collect responses
                responses = self._collect_responses(user_data)

                # Verify we received the expected number of responses
                self.assertEqual(
                    len(responses),
                    expected_count,
                    f"Expected {expected_count} responses, got {len(responses)}",
                )

                # Verify correctness of responses
                for idx, resp in enumerate(responses):
                    output = resp.as_numpy("OUT")
                    self.assertEqual(
                        output[0], idx, f"Response {idx} has incorrect value"
                    )

            finally:
                triton_client.stop_stream()

    def test_backpressure_concurrent_requests(self):
        """
        Test that backpressure works correctly with multiple concurrent requests.
        Each request should have independent backpressure state.
        """
        model_name = "ensemble_enabled_max_inflight_responses"
        num_concurrent = 8
        expected_per_request = 8

        clients = []
        user_datas = []

        try:
            inputs, outputs = self._prepare_infer_args(expected_per_request)

            # Create separate client for each concurrent request
            for i in range(num_concurrent):
                client = grpcclient.InferenceServerClient(SERVER_URL)
                user_data = UserData()

                client.start_stream(callback=partial(callback, user_data))
                client.async_stream_infer(
                    model_name=model_name, inputs=inputs, outputs=outputs
                )

                clients.append(client)
                user_datas.append(user_data)

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
                        output[0], idx, f"Response {idx} has incorrect value"
                    )

        finally:
            for client in clients:
                try:
                    client.stop_stream()
                    client.close()
                except Exception as e:
                    print(f"Exception during client cleanup: {e}")


if __name__ == "__main__":
    unittest.main()
