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
    base_model_name = "sig_tag"
    base_tag = "serve"
    test_tag = "testTag"
    base_sig_def = "serving_default"
    test_sig_def = "testSigDef"
    dims = 16

    def _test_helper(self, modelVersion, tag, sig_def):
        shape = [self.dims]
        model_name = self.base_model_name + str(modelVersion)
        # The multiplier is defined during model creation. See server/qa/common/gen_tag_sigdef.py
        # for details
        multiplier = modelVersion + 1
        output_name = "OUTPUT"
        triton_client = httpclient.InferenceServerClient("localhost:8000",
                                                         verbose=True)
        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput('INPUT', shape, "FP32"))
        input_data = np.ones(shape=shape).astype(np.float32)
        inputs[0].set_data_from_numpy(input_data, binary_data=True)

        outputs.append(
            httpclient.InferRequestedOutput(output_name, binary_data=True))
        results = triton_client.infer(model_name, inputs, outputs=outputs)
        output_data = results.as_numpy(output_name)
        test_output = input_data * multiplier
        self.assertTrue(np.isclose(output_data, test_output).all())

    def test_default(self):
        self._test_helper(0, self.base_tag, self.base_sig_def)

    def test_sig_def(self):
        self._test_helper(1, self.base_tag, self.test_sig_def)

    def test_tag(self):
        self._test_helper(2, self.test_tag, self.base_sig_def)

    def test_tag_sig_def(self):
        self._test_helper(3, self.test_tag, self.test_sig_def)


if __name__ == '__main__':
    unittest.main()
