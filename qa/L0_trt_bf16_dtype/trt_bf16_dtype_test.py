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


import sys

sys.path.append("../common")

import unittest

import numpy as np
import test_util as tu
import tritonclient.http as client


class TrtBF16DataTypeTest(tu.TestResultCollector):
    def setUp(self):
        self.triton_client = client.InferenceServerClient(
            "localhost:8000", verbose=True
        )

    def _infer_helper(self, model_name, shape):
        inputs = []
        outputs = []
        inputs.append(client.InferInput("INPUT0", shape, "BF16"))
        inputs.append(client.InferInput("INPUT1", shape, "BF16"))

        input0_data = np.ones(shape=shape).astype(np.float32)
        input1_data = np.ones(shape=shape).astype(np.float32)

        inputs[0].set_data_from_numpy(input0_data, binary_data=True)
        inputs[1].set_data_from_numpy(input1_data, binary_data=True)

        outputs.append(client.InferRequestedOutput("OUTPUT0", binary_data=True))
        outputs.append(client.InferRequestedOutput("OUTPUT1", binary_data=True))

        results = self.triton_client.infer(model_name, inputs, outputs=outputs)

        output0_data = results.as_numpy("OUTPUT0")
        output1_data = results.as_numpy("OUTPUT1")

        np.testing.assert_equal(
            output0_data,
            input0_data + input1_data,
            "Result output does not match the expected output",
        )
        np.testing.assert_equal(
            output1_data,
            input0_data - input1_data,
            "Result output does not match the expected output",
        )

    def test_fixed(self):
        for bs in [1, 4, 8]:
            self._infer_helper(
                "plan_bf16_bf16_bf16",
                [bs, 16],
            )

        self._infer_helper(
            "plan_nobatch_bf16_bf16_bf16",
            [16],
        )

    def test_dynamic(self):
        for bs in [1, 4, 8]:
            self._infer_helper(
                "plan_bf16_bf16_bf16",
                [bs, 16, 16],
            )

        self._infer_helper(
            "plan_nobatch_bf16_bf16_bf16",
            [16, 16],
        )


if __name__ == "__main__":
    unittest.main()
