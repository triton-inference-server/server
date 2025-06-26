#!/usr/bin/env python3

# Copyright 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import test_util as tu
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException


class ClassificationParameterTest(tu.TestResultCollector):
    def setUp(self):
        self.protocol = os.environ.get("CLIENT_TYPE", "http")
        if self.protocol == "http":
            self.client = httpclient.InferenceServerClient("localhost:8000")
        else:
            self.client = grpcclient.InferenceServerClient("localhost:8001")

    def _prepare_io(self, input_data, dtype):
        if self.protocol == "http":
            inputs = [httpclient.InferInput("INPUT0", input_data.shape, dtype)]
            outputs = [httpclient.InferRequestedOutput(name="OUTPUT0", class_count=5)]
        else:
            inputs = [grpcclient.InferInput("INPUT0", input_data.shape, dtype)]
            outputs = [grpcclient.InferRequestedOutput(name="OUTPUT0", class_count=5)]
        inputs[0].set_data_from_numpy(input_data)
        return inputs, outputs

    def test_classificattion(self):
        shape = (1, 8)
        dtype = "FP32"
        model_name = "identity_fp32"
        input_data = np.ones(shape, dtype=np.float32)

        inputs, outputs = self._prepare_io(input_data, dtype)
        result = self.client.infer(
            model_name=model_name, inputs=inputs, outputs=outputs
        )
        output = result.get_output("OUTPUT0")
        if self.protocol == "http":
            output_dtype = output["datatype"]
        else:
            output_dtype = output.datatype

        self.assertEqual(output_dtype, "BYTES")

    def test_classificattion_incorrect_data_type(self):
        shape = (1, 8)
        model_name = "identity_bytes"
        dtype = "BYTES"
        input_data = np.array([["test"] * shape[1]], dtype=object)

        inputs, outputs = self._prepare_io(input_data, dtype)
        with self.assertRaises(InferenceServerException) as e:
            self.client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

        self.assertIn(
            "class result not available for output due to unsupported type 'BYTES'",
            str(e.exception),
        )


if __name__ == "__main__":
    unittest.main()
