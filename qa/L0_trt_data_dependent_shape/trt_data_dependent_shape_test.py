#!/usr/bin/env python3

# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import unittest

import numpy as np
import test_util as tu
import tritonclient.http as client


class TrtDataDependentShapeTest(tu.TestResultCollector):
    def setUp(self):
        self.triton_client = client.InferenceServerClient(
            "localhost:8000", verbose=True
        )

    def test_fixed(self):
        model_name = "plan_nobatch_nonzero_fixed"
        input_np = np.arange(16, dtype=np.int32).reshape((4, 4))
        expected_output_np = np.nonzero(input_np)

        inputs = []
        inputs.append(client.InferInput("INPUT", [4, 4], "INT32"))
        inputs[-1].set_data_from_numpy(input_np)

        results = self.triton_client.infer(model_name=model_name, inputs=inputs)
        # Validate the results by comparing with precomputed values.
        output_np = results.as_numpy("OUTPUT")
        self.assertTrue(
            np.array_equal(output_np, expected_output_np),
            "OUTPUT expected: {}, got {}".format(expected_output_np, output_np),
        )

    def test_dynamic(self):
        model_name = "plan_nobatch_nonzero_dynamic"
        input_data = []
        for i in range(20 * 16):
            input_data.append(i if (i % 2) == 0 else 0)
        input_np = np.array(input_data, dtype=np.int32).reshape((20, 16))
        expected_output_np = np.nonzero(input_np)

        inputs = []
        inputs.append(client.InferInput("INPUT", [20, 16], "INT32"))
        inputs[-1].set_data_from_numpy(input_np)

        results = self.triton_client.infer(model_name=model_name, inputs=inputs)
        # Validate the results by comparing with precomputed values.
        output_np = results.as_numpy("OUTPUT")
        self.assertTrue(
            np.array_equal(output_np, expected_output_np),
            "OUTPUT expected: {}, got {}".format(expected_output_np, output_np),
        )


if __name__ == "__main__":
    unittest.main()
