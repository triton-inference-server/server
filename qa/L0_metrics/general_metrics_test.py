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
import unittest

import requests

_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")
MODEL_LOAD_TIME = "nv_model_load_time{model="


def get_model_load_times():
    r = requests.get(f"http://{_tritonserver_ipaddr}:8002/metrics")
    r.raise_for_status()
    pattern = re.compile(rf'{MODEL_LOAD_TIME}"(.*?)".*?\ (\d+\.\d+)')
    model_load_times = {}
    matches = pattern.findall(r.text)
    for match in matches:
        model_name, load_time = match
        model_load_times[model_name] = float(load_time)
    return model_load_times


class TestGeneralMetrics(unittest.TestCase):
    def setUp(self):
        self.model_name = "libtorch_float32_float32_float32"

    def test_metrics_load_time(self):
        model_load_times = get_model_load_times()
        load_time = model_load_times.get(self.model_name)

        self.assertIsNotNone(load_time, "Model Load time not found")

        dict_size = len(model_load_times)
        self.assertEqual(dict_size, 1, "Too many model_load_time entries found")

    def test_metrics_load_time_explicit_load(self):
        model_load_times = get_model_load_times()
        load_time = model_load_times.get(self.model_name)

        self.assertIsNotNone(load_time, "Model Load time not found")

        dict_size = len(model_load_times)
        self.assertEqual(dict_size, 1, "Too many model_load_time entries found")

    def test_metrics_load_time_explicit_unload(self):
        r = requests.get(f"http://localhost:8000/v2/repository/models/")
        r.raise_for_status()
        print(r.text)
        model_load_times = get_model_load_times()
        load_time = model_load_times.get(self.model_name)

        self.assertIsNone(load_time, "Model Load time found even after unload")


if __name__ == "__main__":
    unittest.main()
