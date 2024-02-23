#!/usr/bin/env python3

# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import time
import unittest

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient


class TestResponseStatistics(unittest.TestCase):
    def setUp(self):
        self._model_name = "set_by_test_case"
        self._min_infer_delay_ns = 0
        self._min_output_delay_ns = 0
        self._min_cancel_delay_ns = 0
        self._number_of_fail_responses = 0
        self._number_of_empty_responses = 0
        self._statistics_counts = []
        self._grpc_client = grpcclient.InferenceServerClient(
            "localhost:8001", verbose=True
        )
        self._http_client = httpclient.InferenceServerClient("localhost:8000")

    # Return a coupled (callback, response) pair for gRPC stream infer.
    def _generate_streaming_callback_and_response_pair(self):
        # [{"result": result, "error": error}, ...]
        response = []

        def callback(result, error):
            response.append({"result": result, "error": error})

        return callback, response

    # Send an infer request and return its responses. 'number_of_responses' is the sum
    # of success, fail and empty responses the model should return for this request.
    # 'cancel_at_response_size' will cancel the stream when the number of responses
    # received equals the size, set to None if cancellation is not required. This
    # function waits until all success and fail responses are received, or cancelled.
    def _stream_infer(self, number_of_responses, cancel_at_response_size=None):
        callback, responses = self._generate_streaming_callback_and_response_pair()
        self._grpc_client.start_stream(callback)
        input_data = np.array([number_of_responses], dtype=np.int32)
        inputs = [grpcclient.InferInput("IN", input_data.shape, "INT32")]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [grpcclient.InferRequestedOutput("OUT")]
        self._grpc_client.async_stream_infer(
            model_name=self._model_name, inputs=inputs, outputs=outputs
        )
        if cancel_at_response_size is None:
            # poll until all expected responses are received
            while len(responses) < (
                number_of_responses - self._number_of_empty_responses
            ):
                time.sleep(0.1)
            self._grpc_client.stop_stream(cancel_requests=False)
        else:
            # poll until cancellation response size is reached
            while len(responses) < cancel_at_response_size:
                time.sleep(0.1)
            self._grpc_client.stop_stream(cancel_requests=True)
        return responses

    # Update expected statistics counts for the response at 'current_index'.
    # 'number_of_responses' is the sum of success, fail and empty responses expected
    # from this inference request. 'cancel_at_index' is the index at which the request
    # should be cancelled.
    def _update_statistics_counts(
        self, current_index, number_of_responses, cancel_at_index
    ):
        if current_index >= len(self._statistics_counts):
            self._statistics_counts.append(
                {
                    "compute_infer": 0,
                    "compute_output": 0,
                    "success": 0,
                    "fail": 0,
                    "empty_response": 0,
                    "cancel": 0,
                }
            )
        if current_index == cancel_at_index:
            # cancel
            self._statistics_counts[current_index]["cancel"] += 1
        elif (
            current_index
            + self._number_of_fail_responses
            + self._number_of_empty_responses
            < number_of_responses
        ):
            # success
            self._statistics_counts[current_index]["compute_infer"] += 1
            self._statistics_counts[current_index]["compute_output"] += 1
            self._statistics_counts[current_index]["success"] += 1
        elif current_index + self._number_of_empty_responses < number_of_responses:
            # fail
            self._statistics_counts[current_index]["compute_infer"] += 1
            self._statistics_counts[current_index]["compute_output"] += 1
            self._statistics_counts[current_index]["fail"] += 1
        else:
            # empty
            self._statistics_counts[current_index]["compute_infer"] += 1
            self._statistics_counts[current_index]["empty_response"] += 1

    # Check the 'response_stats' at 'current_index' for 'stats_name' is valid.
    def _check_statistics_count_and_duration(
        self, response_stats, current_index, stats_name
    ):
        expected_count = self._statistics_counts[current_index][stats_name]
        if stats_name == "compute_infer" or stats_name == "empty_response":
            delay_ns = self._min_infer_delay_ns
        elif stats_name == "compute_output":
            delay_ns = self._min_output_delay_ns
        elif stats_name == "cancel":
            delay_ns = self._min_cancel_delay_ns
        else:  # success or fail
            delay_ns = self._min_infer_delay_ns + self._min_output_delay_ns
        if delay_ns == 0:
            upper_bound_ns = 10000000 * expected_count
            lower_bound_ns = 0
        else:
            upper_bound_ns = 1.1 * delay_ns * expected_count
            lower_bound_ns = 0.9 * delay_ns * expected_count
        stats = response_stats[str(current_index)][stats_name]
        self.assertEqual(stats["count"], expected_count)
        self.assertLessEqual(stats["ns"], upper_bound_ns)
        self.assertGreaterEqual(stats["ns"], lower_bound_ns)

    # Fetch and return the response statistics from both gRPC and HTTP endpoints, and
    # check they are equivalent before returning.
    def _get_response_statistics(self):
        # http response statistics
        statistics_http = self._http_client.get_inference_statistics(
            model_name=self._model_name
        )
        model_stats_http = statistics_http["model_stats"][0]
        self.assertEqual(model_stats_http["name"], self._model_name)
        response_stats_http = model_stats_http["response_stats"]
        # grpc response statistics
        statistics_grpc = self._grpc_client.get_inference_statistics(
            model_name=self._model_name, as_json=True
        )
        model_stats_grpc = statistics_grpc["model_stats"][0]
        self.assertEqual(model_stats_grpc["name"], self._model_name)
        response_stats_grpc = model_stats_grpc["response_stats"]
        # check equivalent between http and grpc statistics
        self.assertEqual(len(response_stats_http), len(response_stats_grpc))
        for idx, statistics_http in response_stats_http.items():
            self.assertIn(idx, response_stats_grpc)
            statistics_grpc = response_stats_grpc[idx]
            for name, stats_http in statistics_http.items():
                self.assertIn(name, statistics_grpc)
                stats_grpc = statistics_grpc[name]
                # normalize gRPC statistics to http
                stats_grpc["count"] = (
                    int(stats_grpc["count"]) if ("count" in stats_grpc) else 0
                )
                stats_grpc["ns"] = int(stats_grpc["ns"]) if ("ns" in stats_grpc) else 0
                # check equal
                self.assertEqual(stats_http, stats_grpc)
        return response_stats_http

    # Check the response statistics is valid for a given infer request, providing its
    # 'responses', expected 'number_of_responses' and 'cancel_at_index'.
    def _check_response_stats(
        self, responses, number_of_responses, cancel_at_index=None
    ):
        response_stats = self._get_response_statistics()
        self.assertGreaterEqual(len(response_stats), number_of_responses)
        for i in range(number_of_responses):
            self._update_statistics_counts(i, number_of_responses, cancel_at_index)
            self._check_statistics_count_and_duration(
                response_stats, i, "compute_infer"
            )
            self._check_statistics_count_and_duration(
                response_stats, i, "compute_output"
            )
            self._check_statistics_count_and_duration(response_stats, i, "success")
            self._check_statistics_count_and_duration(response_stats, i, "fail")
            self._check_statistics_count_and_duration(
                response_stats, i, "empty_response"
            )
            self._check_statistics_count_and_duration(response_stats, i, "cancel")

    # Test response statistics. The statistics must be valid over two or more infers.
    def test_response_statistics(self):
        self._model_name = "square_int32"
        self._min_infer_delay_ns = 400000000
        self._min_output_delay_ns = 200000000
        self._number_of_fail_responses = 2
        self._number_of_empty_responses = 1
        # Send a request that generates 4 responses.
        number_of_responses = 4
        responses = self._stream_infer(number_of_responses)
        self._check_response_stats(responses, number_of_responses)
        # Send a request that generates 6 responses, and make sure the statistics are
        # aggregated with the previous request.
        number_of_responses = 6
        responses = self._stream_infer(number_of_responses)
        self._check_response_stats(responses, number_of_responses)
        # Send a request that generates 3 responses, and make sure the statistics are
        # aggregated with the previous requests.
        number_of_responses = 3
        responses = self._stream_infer(number_of_responses)
        self._check_response_stats(responses, number_of_responses)

    # Test response statistics with cancellation.
    def test_response_statistics_cancel(self):
        self._model_name = "square_int32_slow"
        self._min_infer_delay_ns = 1200000000
        self._min_output_delay_ns = 800000000
        self._min_cancel_delay_ns = 400000000

        # Send a request that generates 4 responses.
        number_of_responses = 4
        responses = self._stream_infer(number_of_responses)
        self._check_response_stats(responses, number_of_responses)

        # Send a request that generates 4 responses, and cancel on the 3rd response.
        # Make sure the statistics are aggregated with the previous request.
        responses = self._stream_infer(number_of_responses=4, cancel_at_response_size=1)
        # There is an infer and output delay on the 1st and 2nd response, and a cancel
        # delay on the 3rd response.
        min_total_delay_ns = (
            self._min_infer_delay_ns + self._min_output_delay_ns
        ) * 2 + self._min_cancel_delay_ns
        # Make sure the inference and cancellation is completed before checking.
        time.sleep(min_total_delay_ns * 1.5 / 1000000000)
        # The request is cancelled when the 2nd response is computing, so the
        # cancellation should be received at the 3rd response (index 2), making a total
        # of 3 responses on the statistics.
        self._check_response_stats(responses, number_of_responses=3, cancel_at_index=2)


if __name__ == "__main__":
    unittest.main()
