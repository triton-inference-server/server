#!/usr/bin/python
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

import os
import re
import sys
import time
import unittest
from functools import partial

import numpy as np
import requests
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

sys.path.append("../common")
import test_util as tu

MILLIS_PER_SEC = 1000


def get_histogram_metric_key(
    metric_family, model_name, model_version, metric_type, le=""
):
    if metric_type in ["count", "sum"]:
        return f'{metric_family}_{metric_type}{{model="{model_name}",version="{model_version}"}}'
    elif metric_type == "bucket":
        return f'{metric_family}_{metric_type}{{model="{model_name}",version="{model_version}",le="{le}"}}'
    else:
        return None


class TestHistogramMetrics(tu.TestResultCollector):
    def setUp(self):
        self.tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")

    def get_histogram_metrics(self, metric_family: str):
        r = requests.get(f"http://{self.tritonserver_ipaddr}:8002/metrics")
        r.raise_for_status()

        # Regular expression to match the pattern
        pattern = f"^{metric_family}.*"
        histogram_dict = {}

        # Find all matches in the text
        matches = re.findall(pattern, r.text, re.MULTILINE)

        for match in matches:
            key, value = match.rsplit(" ")
            histogram_dict[key] = int(value)

        return histogram_dict

    def async_stream_infer(self, model_name, inputs, outputs):
        triton_client = grpcclient.InferenceServerClient(url="localhost:8001")

        # Define the callback function. Note the last two parameters should be
        # result and error. InferenceServerClient would povide the results of an
        # inference as grpcclient.InferResult in result. For successful
        # inference, error will be None, otherwise it will be an object of
        # tritonclientutils.InferenceServerException holding the error details
        def callback(user_data, result, error):
            if error:
                user_data.append(error)
            else:
                user_data.append(result)

        # list to hold the results of inference.
        user_data = []

        # Inference call
        triton_client.start_stream(callback=partial(callback, user_data))
        triton_client.async_stream_infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
        )
        triton_client.stop_stream()

        # Wait until the results are available in user_data
        time_out = 10
        while (len(user_data) == 0) and time_out > 0:
            time_out = time_out - 1
            time.sleep(1)

        # Display and validate the available results
        if len(user_data) == 1:
            # Check for the errors
            if type(user_data[0]) == InferenceServerException:
                print(user_data[0])
                sys.exit(1)

    def test_ensemble_decoupled(self):
        ensemble_model_name = "ensemble"
        wait_secs = 1

        # Infer
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput("INPUT", [1], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput("OUTPUT"))

        # Create the data for the input tensor. Initialize to all ones.
        input_data = np.ones(shape=(1), dtype=np.float32) * wait_secs
        # Initialize the data
        inputs[0].set_data_from_numpy(input_data)

        # Send 3 requests to ensemble decoupled model
        for request_num in range(3):
            self.async_stream_infer(ensemble_model_name, inputs, outputs)

            # Checks metrics output
            first_response_family = "nv_inference_first_response_histogram_ms"
            decoupled_model_name = "async_execute_decouple"
            histogram_dict = self.get_histogram_metrics(first_response_family)

            ensemble_model_count = get_histogram_metric_key(
                first_response_family, ensemble_model_name, "1", "count"
            )
            ensemble_model_sum = get_histogram_metric_key(
                first_response_family, ensemble_model_name, "1", "sum"
            )
            self.assertIn(ensemble_model_count, histogram_dict)
            self.assertGreaterEqual(histogram_dict[ensemble_model_count], request_num)
            self.assertIn(ensemble_model_sum, histogram_dict)
            self.assertGreaterEqual(
                histogram_dict[ensemble_model_sum],
                2 * wait_secs * MILLIS_PER_SEC * request_num,
            )

            decoupled_model_count = get_histogram_metric_key(
                first_response_family, decoupled_model_name, "1", "count"
            )
            decoupled_model_sum = get_histogram_metric_key(
                first_response_family, decoupled_model_name, "1", "sum"
            )
            self.assertIn(decoupled_model_count, histogram_dict)
            self.assertGreaterEqual(histogram_dict[decoupled_model_count], request_num)
            self.assertIn(decoupled_model_sum, histogram_dict)
            self.assertGreaterEqual(
                histogram_dict[decoupled_model_sum],
                wait_secs * MILLIS_PER_SEC * request_num,
            )


if __name__ == "__main__":
    unittest.main()
