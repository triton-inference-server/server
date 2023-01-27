# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
import os
import test_util as tu
import tritonhttpclient as httpclient


class TfGpuIoTest(tu.TestResultCollector):

    def _test_helper(self, model_name, batching_enabled, input_name,
                     output_name, shape):
        try:
            triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                             verbose=True)
            inputs = []
            outputs = []
            if batching_enabled:
                shape = [1] + shape
            inputs.append(httpclient.InferInput(input_name, shape, "FP32"))
            if batching_enabled:
                input_data = np.ones(shape=shape, dtype=np.float32)
            else:
                input_data = np.ones(shape=shape, dtype=np.float32)
            inputs[0].set_data_from_numpy(input_data, binary_data=True)

            outputs.append(
                httpclient.InferRequestedOutput(output_name, binary_data=True))
            results = triton_client.infer(model_name, inputs, outputs=outputs)
        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_sig_tag0(self):
        self._test_helper("sig_tag0", False, "INPUT", "OUTPUT", [
            16,
        ])

    def test_graphdef_zero_1_float32_def(self):
        self._test_helper("graphdef_zero_1_float32_def", True, "INPUT0",
                          "OUTPUT0", [
                              16384,
                          ])

    def test_graphdef_zero_1_float32_gpu(self):
        self._test_helper("graphdef_zero_1_float32_gpu", True, "INPUT0",
                          "OUTPUT0", [
                              16384,
                          ])

    def test_savedmodel_zero_1_float32_def(self):
        self._test_helper("savedmodel_zero_1_float32_def", True, "INPUT0",
                          "OUTPUT0", [
                              16384,
                          ])

    def test_savedmodel_zero_1_float32_gpu(self):
        self._test_helper("savedmodel_zero_1_float32_gpu", True, "INPUT0",
                          "OUTPUT0", [
                              16384,
                          ])


if __name__ == '__main__':
    unittest.main()
