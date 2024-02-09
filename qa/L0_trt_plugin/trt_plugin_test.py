#!/usr/bin/env python3

# Copyright 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import unittest

import numpy as np
import test_util as tu
import tritonclient.http as httpclient

# By default, find tritonserver on "localhost", but can be overridden
# with TRITONSERVER_IPADDR envvar
_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class PluginModelTest(tu.TestResultCollector):
    def _full_exact(self, model_name, plugin_name, shape):
        print(f"{_tritonserver_ipaddr}:8000")
        triton_client = httpclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8000")

        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput("INPUT0", list(shape), "FP32"))

        input0_data = np.ones(shape=shape).astype(np.float32)
        inputs[0].set_data_from_numpy(input0_data, binary_data=True)

        outputs.append(httpclient.InferRequestedOutput("OUTPUT0", binary_data=True))

        results = triton_client.infer(
            model_name + "_" + plugin_name, inputs, outputs=outputs
        )

        output0_data = results.as_numpy("OUTPUT0")
        tolerance_relative = 1e-6
        tolerance_absolute = 1e-7

        # Verify values of Clip, GELU, and Normalize
        if plugin_name == "CustomClipPlugin":
            # Clip data to minimum of .1, maximum of .5
            test_output = np.clip(input0_data, 0.1, 0.5)
            np.testing.assert_allclose(
                output0_data,
                test_output,
                rtol=tolerance_relative,
                atol=tolerance_absolute,
            )
        elif plugin_name == "CustomGeluPluginDynamic":
            # Add bias
            input0_data += 1
            # Calculate Gelu activation
            test_output = (input0_data * 0.5) * (
                1 + np.tanh((0.797885 * input0_data) + (0.035677 * (input0_data**3)))
            )
            np.testing.assert_allclose(
                output0_data,
                test_output,
                rtol=tolerance_relative,
                atol=tolerance_absolute,
            )
        elif plugin_name == "Normalize_TRT":
            # L2 norm is sqrt(sum([1]*16)))
            test_output = input0_data / np.sqrt(sum([1] * 16))
            np.testing.assert_allclose(
                output0_data,
                test_output,
                rtol=tolerance_relative,
                atol=tolerance_absolute,
            )
        else:
            self.fail("Unexpected plugin: " + plugin_name)

    def test_raw_fff_clip(self):
        for bs in (1, 8):
            self._full_exact(
                "plan_float32_float32_float32", "CustomClipPlugin", (bs, 16)
            )

    def test_raw_fff_gelu(self):
        self._full_exact(
            "plan_nobatch_float32_float32_float32",
            "CustomGeluPluginDynamic",
            (16, 1, 1),
        )

    def test_raw_fff_norm(self):
        # model that supports batching
        for bs in (1, 8):
            self._full_exact(
                "plan_float32_float32_float32", "Normalize_TRT", (bs, 16, 16, 16)
            )


if __name__ == "__main__":
    unittest.main()
