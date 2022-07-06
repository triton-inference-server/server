# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

sys.path.append("../../common")

import test_util as tu
import tritonclient.http as httpclient
from tritonclient.utils import *
import numpy as np
import unittest


class RestartTest(tu.TestResultCollector):

    def _infer_helper(self, model_name, shape, data_type):
        with httpclient.InferenceServerClient("localhost:8000") as client:
            input_data_0 = np.array(np.random.randn(*shape), dtype=data_type)
            inputs = [
                httpclient.InferInput("INPUT0", shape,
                                      np_to_triton_dtype(input_data_0.dtype))
            ]
            inputs[0].set_data_from_numpy(input_data_0)
            result = client.infer(model_name, inputs)
            output0 = result.as_numpy('OUTPUT0')
            self.assertTrue(np.all(input_data_0 == output0))

    def test_restart(self):
        shape = [1, 16]
        model_name = 'restart'
        dtype = np.float32

        # Since the stub process has been killed, the first request
        # will return an exception.
        with self.assertRaises(InferenceServerException):
            self._infer_helper(model_name, shape, dtype)

        # The second request should work properly since the stub process should
        # have come alive.
        self._infer_helper(model_name, shape, dtype)

    def test_infer(self):
        shape = [1, 16]
        model_name = 'restart'
        dtype = np.float32
        self._infer_helper(model_name, shape, dtype)


if __name__ == '__main__':
    unittest.main()
