# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
from tensorrtserver.api import *

class TFTRTOptimizationTest(unittest.TestCase):
    def setUp(self):
        self.input0_ = np.arange(start=0, stop=16, dtype=np.float32)
        self.input1_ = np.ones(shape=16, dtype=np.float32)
        self.expected_output0_ = self.input0_ + self.input1_
        self.expected_output1_ = self.input0_ - self.input1_

    def _addsub_infer(self, model_name):
        infer_ctx = InferContext("localhost:8000", ProtocolType.HTTP, model_name)

        result = infer_ctx.run({ 'INPUT0' : (self.input0_,),
                                'INPUT1' : (self.input1_,) },
                            { 'OUTPUT0' : InferContext.ResultFormat.RAW,
                                'OUTPUT1' : InferContext.ResultFormat.RAW },
                            1)
        output0_data = result['OUTPUT0'][0]
        output1_data = result['OUTPUT1'][0]

        self.assertTrue(np.array_equal(self.expected_output0_, output0_data), "incorrect sum")
        self.assertTrue(np.array_equal(self.expected_output1_, output1_data), "incorrect difference")

    def test_graphdef(self):
        self._addsub_infer("graphdef_float32_float32_float32_trt")
        self._addsub_infer("graphdef_float32_float32_float32_param")

    def test_savedmodel(self):
        self._addsub_infer("savedmodel_float32_float32_float32_trt")
        self._addsub_infer("savedmodel_float32_float32_float32_param")


if __name__ == '__main__':
    unittest.main()
