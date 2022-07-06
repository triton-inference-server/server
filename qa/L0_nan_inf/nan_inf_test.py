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

sys.path.append('../common')

import json
import unittest
import traceback

import requests
import numpy as np
import tritonclient.http as tritonhttpclient
import tritonclient.grpc as tritongrpcclient
from tritonclient.utils import InferenceServerException
import test_util as tu


class NanInfTest(tu.TestResultCollector):
    expected_output = np.array([np.nan, np.inf, np.NINF, 1, 2, 3],
                               dtype=np.float32)
    model_name = "nan_inf_output"

    def test_http_raw(self):
        payload = {
            "inputs": [{
                "name": "INPUT0",
                "datatype": "FP32",
                "shape": [1],
                "data": [1]
            }]
        }
        response = requests.post(
            "http://localhost:8000/v2/models/nan_inf_output/infer",
            data=json.dumps(payload))
        if not response.ok:
            self.assertTrue(False, "Response not OK: {}".format(response.text))

        try:
            print(response.json())
        except:
            self.assertTrue(
                False, "Response was not valid JSON:\n{}".format(response.text))

    def test_http(self):
        triton_client = tritonhttpclient.InferenceServerClient("localhost:8000")
        inputs = []
        inputs.append(tritonhttpclient.InferInput('INPUT0', [1], "FP32"))
        self.infer_helper(triton_client, inputs)

    def test_grpc(self):
        triton_client = tritongrpcclient.InferenceServerClient("localhost:8001")
        inputs = []
        inputs.append(tritongrpcclient.InferInput('INPUT0', [1], "FP32"))
        self.infer_helper(triton_client, inputs)

    def infer_helper(self, triton_client, inputs):
        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.float32))

        try:
            results = triton_client.infer(model_name=self.model_name,
                                          inputs=inputs)
            output0_data = results.as_numpy('OUTPUT0')
            # Verify output is as expected
            # Make sure nan's are equivalent when compared
            output_correct = np.array_equal(output0_data,
                                            self.expected_output,
                                            equal_nan=True)
            self.assertTrue(
                output_correct,
                "didn't get expected output0: {}".format(output0_data))
        except InferenceServerException as ex:
            self.assertTrue(False, ex.message())
        except:
            self.assertTrue(False, traceback.format_exc())


if __name__ == '__main__':
    unittest.main()
