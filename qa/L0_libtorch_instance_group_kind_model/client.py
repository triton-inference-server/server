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

import os
import sys

sys.path.append("../common")

import unittest

import numpy as np
import test_util as tu
import tritonclient.http as httpclient

# By default, find tritonserver on "localhost", but can be overridden
# with TRITONSERVER_IPADDR envvar
_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class InferTest(tu.TestResultCollector):
    def test_infer(self):
        try:
            triton_client = httpclient.InferenceServerClient(
                url=f"{_tritonserver_ipaddr}:8000"
            )
        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit(1)

        model_name = os.environ["MODEL_NAME"]

        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput("INPUT0", [1, 16], "FP32"))
        inputs.append(httpclient.InferInput("INPUT1", [1, 16], "FP32"))

        # Create the data for the two input tensors.
        input0_data = np.arange(start=0, stop=16, dtype=np.float32)
        input0_data = np.expand_dims(input0_data, axis=0)
        input1_data = np.arange(start=32, stop=48, dtype=np.float32)
        input1_data = np.expand_dims(input1_data, axis=0)

        # Initialize the data
        inputs[0].set_data_from_numpy(input0_data, binary_data=True)
        inputs[1].set_data_from_numpy(input1_data, binary_data=True)

        outputs.append(httpclient.InferRequestedOutput("OUTPUT__0", binary_data=True))
        outputs.append(httpclient.InferRequestedOutput("OUTPUT__1", binary_data=True))

        results = triton_client.infer(model_name, inputs, outputs=outputs)

        output0_data = results.as_numpy("OUTPUT__0")
        output1_data = results.as_numpy("OUTPUT__1")

        expected_output_0 = input0_data + input1_data
        expected_output_1 = input0_data - input1_data

        self.assertEqual(output0_data.shape, (1, 16))
        self.assertEqual(output1_data.shape, (1, 16))

        self.assertTrue(np.all(expected_output_0 == output0_data))
        self.assertTrue(np.all(expected_output_1 == output1_data))


if __name__ == "__main__":
    unittest.main()
