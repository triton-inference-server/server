# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import tritonhttpclient
from tritonclientutils import *


class UnknownRankTest(tu.TestResultCollector):
    # helper function to generate requests to the server
    def infer_unknown(self, model_name, tensor_shape):
        print("About to run the test")
        input_data = np.random.random_sample(tensor_shape).astype(np.float32)
        client = tritonhttpclient.InferenceServerClient('localhost:8000')
        inputs = [
            tritonhttpclient.InferInput("INPUT", input_data.shape,
                                        np_to_triton_dtype(input_data.dtype))
        ]
        inputs[0].set_data_from_numpy(input_data)
        results = client.infer(model_name, inputs)
        self.assertTrue(np.array_equal(results.as_numpy('OUTPUT'), input_data))

    def test_success(self):
        model_name = "unknown_rank_success"
        tensor_shape = (1)
        try:
            self.infer_unknown(model_name, tensor_shape)
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_wrong_input(self):
        model_name = "unknown_rank_wrong_output"
        tensor_shape = (1, 2)
        try:
            self.infer_unknown(model_name, tensor_shape)
            self.fail(
                "Found success when expected failure with model given " \
                "wrong input tensor [1,2] for input [-1,1]."
            )
        except InferenceServerException as ex:
            self.assertIn(
                "unexpected shape for input \'INPUT\' for model " \
                "\'unknown_rank_wrong_output\'. Expected [1], got [1,2]",
                ex.message())


if __name__ == '__main__':
    unittest.main()
