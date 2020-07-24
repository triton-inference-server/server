# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

from builtins import range
from future.utils import iteritems
import unittest
import numpy as np
import os
import test_util as tu

import tritonhttpclient as httpclient
from tritonclientutils import InferenceServerException


class PluginModelTest(tu.TestResCollector):

    def _full_exact(self, batch_size, model_name, plugin_name):
        triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                         verbose=True)

        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput('INPUT0', [batch_size, 16], "FP32"))

        input0_data = np.random.randn(batch_size, 16).astype(np.float32)
        inputs[0].set_data_from_numpy(input0_data, binary_data=True)

        outputs.append(
            httpclient.InferRequestedOutput('OUTPUT0', binary_data=True))

        results = triton_client.infer(model_name + '_' + plugin_name,
                                      inputs,
                                      outputs=outputs)

        output0_data = results.as_numpy('OUTPUT0')

        # Verify values of Leaky RELU (it uses 0.1 instead of the default 0.01)
        # and for CustomClipPlugin min_clip = 0.1, max_clip = 0.5
        for b in range(batch_size):
            if plugin_name == 'LReLU_TRT':
                test_input = np.where(input0_data > 0, input0_data,
                                      input0_data * 0.1)
                self.assertTrue(np.isclose(output0_data, test_input).all())
            else:
                # [TODO] Add test for CustomClip output
                test_input = np.clip(input0_data, 0.1, 0.5)

    def test_raw_fff_lrelu(self):
        # model that supports batching
        for bs in (1, 8):
            self._full_exact(bs, 'plan_float32_float32_float32', 'LReLU_TRT')

    # add test for CustomClipPlugin after model is fixed


if __name__ == '__main__':
    unittest.main()
