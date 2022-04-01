# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import numpy as np
import requests as httpreq
import unittest
from builtins import range
import tritonhttpclient as httpclient
import test_util as tu

FLAGS = None


class SharedWeightsTest(tu.TestResultCollector):

    def _full_exact(self, model_name, request_concurrency, shape):

        # Run async requests to make sure backend handles concurrent requests
        # correctly.
        client = httpclient.InferenceServerClient(
            "localhost:8000", concurrency=request_concurrency)
        input_datas = []
        requests = []
        for i in range(request_concurrency):
            input_data = (16384 * np.random.randn(*shape)).astype(np.float32)
            input_datas.append(input_data)
            inputs = [
                httpclient.InferInput("INPUT__0", input_data.shape, "FP32")
            ]
            inputs[0].set_data_from_numpy(input_data)
            requests.append(client.async_infer(model_name, inputs))

        for i in range(request_concurrency):
            # Get the result from the initiated asynchronous inference request.
            # Note the call will block until the server responds.
            results = requests[i].get_result()

            output_data = results.as_numpy("OUTPUT__0")
            self.assertIsNotNone(output_data,
                                 "error: expected 'OUTPUT__0' to be found")
            np.testing.assert_allclose(output_data, input_datas[i])

    def test_pytorch_identity_model(self):
        model_name = "libtorch_nobatch_zero_1_float32"
        self._full_exact(model_name, 128, [8])


if __name__ == '__main__':
    unittest.main()
