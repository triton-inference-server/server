#!/usr/bin/env python
# Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from tritonclient.utils import InferenceServerException

# By default, find tritonserver on "localhost", but can be overridden
# with TRITONSERVER_IPADDR envvar
_tritonserver_ipaddr = os.environ.get('TRITONSERVER_IPADDR', 'localhost')


class InferTest(tu.TestResultCollector):

    def test_infer(self):
        try:
            triton_client = httpclient.InferenceServerClient(
                url=f"{_tritonserver_ipaddr}:8000")
        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit(1)

        model_name = "libtorch_int32_int32_int32"

        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput('INPUT0', [1, 16], "INT32"))
        inputs.append(httpclient.InferInput('INPUT1', [1, 16], "INT32"))

        # Create the data for the two input tensors. Initialize the first
        # to unique integers and the second to all ones.
        input0_data = np.arange(start=0, stop=16, dtype=np.int32)
        input0_data = np.expand_dims(input0_data, axis=0)
        input1_data = np.full(shape=(1, 16), fill_value=-1, dtype=np.int32)

        # Initialize the data
        inputs[0].set_data_from_numpy(input0_data, binary_data=True)
        inputs[1].set_data_from_numpy(input1_data, binary_data=True)

        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT__0', binary_data=True))
        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT__1', binary_data=True))

        results = triton_client.infer(model_name, inputs, outputs=outputs)

        output0_data = results.as_numpy('OUTPUT__0')
        output1_data = results.as_numpy('OUTPUT__1')

        # Validate the results by comparing with precomputed values.
        for i in range(16):
            print(
                str(input0_data[0][i]) + " - " + str(input1_data[0][i]) +
                " = " + str(output0_data[0][i]))
            print(
                str(input0_data[0][i]) + " + " + str(input1_data[0][i]) +
                " = " + str(output1_data[0][i]))
            if (input0_data[0][i] - input1_data[0][i]) != output0_data[0][i]:
                print("sync infer error: incorrect difference")
                sys.exit(1)
            if (input0_data[0][i] + input1_data[0][i]) != output1_data[0][i]:
                print("sync infer error: incorrect sum")
                sys.exit(1)


if __name__ == '__main__':
    unittest.main()
