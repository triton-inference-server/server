# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
import tritonhttpclient as httpclient
from tritonclientutils import InferenceServerException


class TFTRTOptimizationTest(tu.TestResultCollector):

    def setUp(self):
        self.input0_ = np.arange(start=0, stop=16,
                                 dtype=np.float32).reshape(1, 16)
        self.input1_ = np.ones(shape=16, dtype=np.float32).reshape(1, 16)
        self.expected_output0_ = self.input0_ + self.input1_
        self.expected_output1_ = self.input0_ - self.input1_

    def _addsub_infer(self, model_name):
        triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                         verbose=True)

        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput('INPUT0', [1, 16], "FP32"))
        inputs.append(httpclient.InferInput('INPUT1', [1, 16], "FP32"))

        # Initialize the data
        inputs[0].set_data_from_numpy(self.input0_, binary_data=True)
        inputs[1].set_data_from_numpy(self.input1_, binary_data=False)

        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT0', binary_data=True))
        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT1', binary_data=True))

        results = triton_client.infer(model_name, inputs, outputs=outputs)

        output0_data = results.as_numpy('OUTPUT0')
        output1_data = results.as_numpy('OUTPUT1')

        self.assertTrue(np.array_equal(self.expected_output0_, output0_data),
                        "incorrect sum")
        self.assertTrue(np.array_equal(self.expected_output1_, output1_data),
                        "incorrect difference")

    def test_graphdef(self):
        self._addsub_infer("graphdef_float32_float32_float32_trt")
        self._addsub_infer("graphdef_float32_float32_float32_param")

    def test_savedmodel(self):
        self._addsub_infer("savedmodel_float32_float32_float32_trt")
        self._addsub_infer("savedmodel_float32_float32_float32_param")


if __name__ == '__main__':
    unittest.main()
