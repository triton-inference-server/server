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

import concurrent.futures
import time
import unittest

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


class TestResponseStatistics(unittest.TestCase):
    def setUp(self):
        self._model_name = "square_int32"
        self._min_infer_delay_ns = 400000000
        self._min_output_delay_ns = 200000000
        self._number_of_fail_responses = 2
        self._number_of_empty_responses = 1
        self._statistics_counts = []
        self._grpc_client = grpcclient.InferenceServerClient(
            "localhost:8001", verbose=True
        )
        self._http_client = httpclient.InferenceServerClient("localhost:8000")

    def _generate_streaming_callback_and_response_pair(self):
        response = []  # [{"result": result, "error": error}, ...]

        def callback(result, error):
            response.append({"result": result, "error": error})

        return callback, response

    def _stream_infer(self, number_of_responses):
        callback, responses = self._generate_streaming_callback_and_response_pair()
        self._grpc_client.start_stream(callback)
        input_data = np.array([number_of_responses], dtype=np.int32)
        inputs = [grpcclient.InferInput("IN", input_data.shape, "INT32")]
        inputs[0].set_data_from_numpy(input_data)
        outputs = [grpcclient.InferRequestedOutput("OUT")]
        self._grpc_client.async_stream_infer(
            model_name=self._model_name, inputs=inputs, outputs=outputs
        )
        while len(responses) < (number_of_responses - self._number_of_empty_responses):
            time.sleep(0.1)  # poll until all expected responses are received
        self._grpc_client.stop_stream()
        return responses

    def _update_statistics_counts(self, current_index, number_of_responses):
        if current_index >= len(self._statistics_counts):
            self._statistics_counts.append(
                {
                    "compute_infer": 0,
                    "compute_output": 0,
                    "success": 0,
                    "fail": 0,
                    "empty_response": 0,
                }
            )
        if (
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

    def _check_statistics_count_and_duration(
        self, response_stats, current_index, stats_name
    ):
        expected_count = self._statistics_counts[current_index][stats_name]
        if stats_name == "compute_infer" or stats_name == "empty_response":
            delay_ns = self._min_infer_delay_ns
        elif stats_name == "compute_output":
            delay_ns = self._min_output_delay_ns
        else:  # success or fail
            delay_ns = self._min_infer_delay_ns + self._min_output_delay_ns
        upper_bound_ns = 1.01 * delay_ns * expected_count
        lower_bound_ns = 0.99 * delay_ns * expected_count
        stats = response_stats[str(current_index)][stats_name]
        self.assertEqual(stats["count"], expected_count)
        self.assertLessEqual(stats["ns"], upper_bound_ns)
        self.assertGreaterEqual(stats["ns"], lower_bound_ns)

    def _check_response_stats(self, responses, number_of_responses):
        statistics_grpc = self._grpc_client.get_inference_statistics(
            model_name=self._model_name, as_json=True
        )
        statistics_http = self._http_client.get_inference_statistics(
            model_name=self._model_name
        )
        # self.assertEqual(statistics_grpc, statistics_http)
        model_stats = statistics_http["model_stats"][0]
        self.assertEqual(model_stats["name"], self._model_name)
        response_stats = model_stats["response_stats"]
        self.assertGreaterEqual(len(response_stats), number_of_responses)
        for i in range(number_of_responses):
            self._update_statistics_counts(i, number_of_responses)
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

    def test_response_statistics(self):
        number_of_responses = 4
        responses = self._stream_infer(number_of_responses)
        self._check_response_stats(responses, number_of_responses)

        number_of_responses = 6
        responses = self._stream_infer(number_of_responses)
        self._check_response_stats(responses, number_of_responses)

        number_of_responses = 3
        responses = self._stream_infer(number_of_responses)
        self._check_response_stats(responses, number_of_responses)


if __name__ == "__main__":
    unittest.main()
