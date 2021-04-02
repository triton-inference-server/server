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


class TagSigdefTest(tu.TestResultCollector):
    def _test_helper(self, model_name, tag, sig_def):

        # {
        #     tag/sig_def: multiplier
        # }

        shape = [16]
        output_name = "OUTPUT"
        triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                         verbose=True)
        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput('INPUT', shape, "FP32"))

        input0_data = np.ones(shape=shape).astype(np.float32)
        inputs[0].set_data_from_numpy(input0_data, binary_data=True)

        outputs.append(
            httpclient.InferRequestedOutput(output_name, binary_data=True))

        results = triton_client.infer(model_name,
                                      inputs,
                                      outputs=outputs)

        output0_data = results.as_numpy(output_name)

        # if
        multiplier = 2
        test_output = input0_data * multiplier
        print(test_output)
        print(output0_data)
        self.assertTrue(np.isclose(output0_data, test_output).all())

    def test_default_tag(self):
        signature_def_name="testSigDef"
        tag_name="testTag"
        self._test_helper('sig_tag', tag_name, signature_def_name)


if __name__ == '__main__':
    unittest.main()
