#!/usr/bin/env python
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tritonclient.http as tritonclient
from tritonclient.utils import InferenceServerException

import nvidia_smi


class MemoryUsageTest(unittest.TestCase):

    def setUp(self):
        nvidia_smi.nvmlInit()
        self.gpu_handle_ = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        self.client_ = tritonclient.InferenceServerClient(url="localhost:8000")

    def tearDown(self):
        nvidia_smi.nvmlShutdown()

    def report_used_gpu_memory(self):
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(self.gpu_handle_)
        return info.used

    def is_testing_backend(self, model_name, backend_name):
        return self.client_.get_model_config(
            model_name)["backend"] == backend_name

    def verify_recorded_usage(self, model_stat):
        recorded_gpu_usage = 0
        for usage in model_stat["memory_usage"]:
            if usage["type"] == "GPU":
                recorded_gpu_usage += usage["byte_size"]
        # unload and verify recorded usage
        before_total_usage = self.report_used_gpu_memory()
        self.client_.unload_model(model_stat["name"])
        # unload return after puttting model to unload process,
        # wait to be finished
        time.sleep(2)
        usage_delta = before_total_usage - self.report_used_gpu_memory()
        # check with tolerance as gpu usage obtained is overall usage
        self.assertTrue(
            usage_delta * 0.9 <= recorded_gpu_usage <= usage_delta * 1.1,
            msg=
            "For model {}, expect recorded usage to be in range [{}, {}], got {}"
            .format(model_stat["name"], usage_delta * 0.9, usage_delta * 1.1,
                    recorded_gpu_usage))

    def test_onnx(self):
        model_stats = self.client_.get_inference_statistics()["model_stats"]
        for model_stat in model_stats:
            if self.is_testing_backend(model_stat["name"], "onnxruntime"):
                self.verify_recorded_usage(model_stat)

    def test_plan(self):
        model_stats = self.client_.get_inference_statistics()["model_stats"]
        for model_stat in model_stats:
            if self.is_testing_backend(model_stat["name"], "tensorrt"):
                self.verify_recorded_usage(model_stat)


if __name__ == "__main__":
    unittest.main()
