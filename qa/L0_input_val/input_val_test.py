#!/usr/bin/env python
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import traceback
import unittest

import numpy as np
import requests
import test_util as tu
import tritonclient.grpc as tritongrpcclient
import tritonclient.http as tritonhttpclient
from tritonclient.utils import InferenceServerException


class InputValTest(tu.TestResultCollector):
    def test_input_validation_required_empty(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        infer_response = triton_client.infer("input_all_required", inputs=inputs)
        self.assertTrue(infer_response.has_error())
        self.assertIn(
            "expected 3 required inputs but got 0 inputs for model 'input_val_output'. Got inputs [], but missing ['INPUT0','INPUT1','INPUT2']",
            infer_response.error().message(),
        )

    def test_input_validation_required_missing(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        inputs.append(tritonhttpclient.InferInput("INPUT0", [1], "FP32"))
        inputs.append(tritonhttpclient.InferInput("INPUT1", [1], "FP32"))

        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.float32))
        inputs[1].set_data_from_numpy(np.arange(1, dtype=np.float32))

        infer_response = triton_client.infer("input_all_required", inputs=inputs)
        self.assertTrue(infer_response.has_error())
        self.assertIn(
            "expected 3 inputs but got 2 inputs for model 'input_all_required'. Got inputs ['INPUT0','INPUT1], but missing ['INPUT2']",
            infer_response.error().message(),
        )

    def test_input_validation_optional(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        inputs.append(tritonhttpclient.InferInput("INPUT0", [1], "FP32"))
        # Option Input is added, required is missing
        inputs.append(tritonhttpclient.InferInput("INPUT3", [1], "FP32"))

        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.float32))
        inputs[1].set_data_from_numpy(np.arange(1, dtype=np.float32))

        infer_response = triton_client.infer("input_optional", inputs=inputs)
        self.assertTrue(infer_response.has_error())
        self.assertIn(
            "expected number of inputs between 4 and 3 but got 2 inputs for model 'input_val_output'. Got inputs ['INPUT0','INPUT3'], but missing required inputs ['INPUT1','INPUT2']",
            infer_response.error().message(),
        )


if __name__ == "__main__":
    unittest.main()
