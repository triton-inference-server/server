# /usr/bin/python
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
import time
import unittest

import requests

_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")
MODEL_LOAD_TIME = "nv_model_load_duration_secs{model="


def get_model_load_times():
    r = requests.get(f"http://{_tritonserver_ipaddr}:8002/metrics")
    r.raise_for_status()
    # Initialize an empty dictionary to store the data
    model_data = {}
    lines = r.text.strip().split("\n")
    for line in lines:
        # Use regex to extract model name, version, and load time
        match = re.match(
            r"nv_model_load_duration_secs\{model=\"(.*?)\",version=\"(.*?)\"\} (.*)",
            line,
        )
        if match:
            model_name = match.group(1)
            model_version = match.group(2)
            load_time = float(match.group(3))
            # Store in dictionary
            if model_name not in model_data:
                model_data[model_name] = {}
            model_data[model_name][model_version] = load_time
    return model_data


def load_model_explicit(model_name, server_url="http://localhost:8000"):
    endpoint = f"{server_url}/v2/repository/models/{model_name}/load"
    response = requests.post(endpoint)
    try:
        self.assertEqual(response.status_code, 200)
        print(f"Model '{model_name}' loaded successfully.")
    except AssertionError:
        print(
            f"Failed to load model '{model_name}'. Status code: {response.status_code}"
        )
        print("Response:", response.text)


def unload_model_explicit(model_name, server_url="http://localhost:8000"):
    endpoint = f"{server_url}/v2/repository/models/{model_name}/unload"
    response = requests.post(endpoint)
    try:
        self.assertEqual(response.status_code, 200)
        print(f"Model '{model_name}' unloaded successfully.")
    except AssertionError:
        print(
            f"Failed to unload model '{model_name}'. Status code: {response.status_code}"
        )
        print("Response:", response.text)


class TestGeneralMetrics(unittest.TestCase):
    def setUp(self):
        self.model_name = "libtorch_float32_float32_float32"
        self.model_name_multiple_versions = "input_all_optional"

    def test_metrics_load_time(self):
        model_load_times = get_model_load_times()
        load_time = model_load_times.get(self.model_name, {}).get("1")

        self.assertIsNotNone(load_time, "Model Load time not found")

        dict_size = len(model_load_times)
        self.assertEqual(dict_size, 1, "Too many model_load_time entries found")

    def test_metrics_load_time_explicit_load(self):
        model_load_times = get_model_load_times()
        load_time = model_load_times.get(self.model_name, {}).get("1")

        self.assertIsNotNone(load_time, "Model Load time not found")

        dict_size = len(model_load_times)
        self.assertEqual(dict_size, 1, "Too many model_load_time entries found")

    def test_metrics_load_time_explicit_unload(self):
        model_load_times = get_model_load_times()
        load_time = model_load_times.get(self.model_name, {}).get("1")
        self.assertIsNone(load_time, "Model Load time found even after unload")

    def test_metrics_load_time_multiple_version_reload(self):
        # Part 0 check start condistion, metric should not be present
        model_load_times = get_model_load_times()
        load_time = model_load_times.get(self.model_name, {}).get("1")
        self.assertIsNone(load_time, "Model Load time found even before model load")

        # Part 1 load multiple versions of the same model and check if slow and fast models reflect the metric correctly
        load_model_explicit(self.model_name_multiple_versions)
        model_load_times = get_model_load_times()
        load_time_slow = model_load_times.get(
            self.model_name_multiple_versions, {}
        ).get("1")
        load_time_fast = model_load_times.get(
            self.model_name_multiple_versions, {}
        ).get("2")
        # Fail the test if load_time_slow is less than load_time_fast
        self.assertGreaterEqual(
            load_time_slow,
            load_time_fast,
            "Slow load time should be greater than or equal to fast load time",
        )
        # Fail the test if load_time_slow is less than 10 seconds as manual delay is 10 seconds
        self.assertGreaterEqual(
            load_time_slow,
            10,
            "Slow load time should be greater than or equal to fast load time",
        )
        # Fail the test if load_time_fast is greater than generous 2 seconds
        self.assertLess(
            load_time_fast,
            2,
            "Model taking too much time to load",
        )

        # Part 2 load multiple versions AGAIN and compare with prev values expect to be the same
        # as triton does not actually load the model again.
        load_model_explicit(self.model_name_multiple_versions)
        model_load_times_new = get_model_load_times()
        load_time_slow_new = model_load_times_new.get(
            self.model_name_multiple_versions, {}
        ).get("1")
        load_time_fast_new = model_load_times_new.get(
            self.model_name_multiple_versions, {}
        ).get("2")
        self.assertEqual(load_time_fast_new, load_time_fast)
        self.assertEqual(load_time_slow_new, load_time_slow)

        # Part 3 unload the model and expect the metrics to go away as model is not loaded now
        unload_model_explicit(self.model_name_multiple_versions)
        time.sleep(1)
        model_load_times_new = get_model_load_times()
        load_time_slow_new = model_load_times_new.get(
            self.model_name_multiple_versions, {}
        ).get("1")
        load_time_fast_new = model_load_times_new.get(
            self.model_name_multiple_versions, {}
        ).get("2")
        self.assertIsNone(load_time_slow_new, "Model Load time found even after unload")
        self.assertIsNone(load_time_fast_new, "Model Load time found even after unload")


if __name__ == "__main__":
    unittest.main()
