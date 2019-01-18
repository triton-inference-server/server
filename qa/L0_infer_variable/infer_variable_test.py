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

from builtins import range
from future.utils import iteritems
import unittest
import numpy as np
import infer_util as iu
import test_util as tu
from tensorrtserver.api import *

np_dtype_string = np.dtype(object)

class InferVariableTest(unittest.TestCase):
    def _full_exact(self, req_raw, input_dtype, output0_dtype, output1_dtype,
                    input_shape, output0_shape, output1_shape, swap=False):

        if tu.validate_for_tf_model(input_dtype, output0_dtype, output1_dtype,
                                    input_shape, output0_shape, output1_shape):
            # model that supports batching
            for bs in (1, 8):
                iu.infer_exact(self, 'graphdef', input_shape, bs, req_raw,
                               input_dtype, output0_dtype, output1_dtype,
                               swap=swap, send_input_shape=True)
                iu.infer_exact(self, 'savedmodel', input_shape, bs, req_raw,
                               input_dtype, output0_dtype, output1_dtype,
                               swap=swap, send_input_shape=True)
            # model that does not support batching
            iu.infer_exact(self, 'graphdef_nobatch', input_shape, 1, req_raw,
                           input_dtype, output0_dtype, output1_dtype,
                           swap=swap, send_input_shape=True)
            iu.infer_exact(self, 'savedmodel_nobatch', input_shape, 1, req_raw,
                           input_dtype, output0_dtype, output1_dtype,
                           swap=swap, send_input_shape=True)


    def test_raw_fff(self):
        self._full_exact(True, np.float32, np.float32, np.float32, (16,), (16,), (16,))
    def test_raw_fii(self):
        self._full_exact(True, np.float32, np.int32, np.int32, (2,8), (2,8), (2,8))
    def test_raw_fll(self):
        self._full_exact(True, np.float32, np.int64, np.int64, (8,4), (8,4), (8,4))
    def test_raw_fil(self):
        self._full_exact(True, np.float32, np.int32, np.int64, (2,8,2), (2,8,2), (2,8,2))
    def test_raw_ooo(self):
        self._full_exact(True, np_dtype_string, np_dtype_string, np_dtype_string, (16,), (16,), (16,))
    def test_raw_oii(self):
        self._full_exact(True, np_dtype_string, np.int32, np.int32, (2,8), (2,8), (2,8))
    def test_raw_ooi(self):
        self._full_exact(True, np_dtype_string, np_dtype_string, np.int32, (8,4), (8,4), (8,4))
    def test_raw_oio(self):
        self._full_exact(True, np_dtype_string, np.int32, np_dtype_string, (2,8,2), (2,8,2), (2,8,2))

    def test_class_fff(self):
        self._full_exact(False, np.float32, np.float32, np.float32, (16,), (16,), (16,))
    def test_class_fii(self):
        self._full_exact(False, np.float32, np.int32, np.int32, (2,8), (2,8), (2,8))
    def test_class_fll(self):
        self._full_exact(False, np.float32, np.int64, np.int64, (8,4), (8,4), (8,4))
    def test_class_fil(self):
        self._full_exact(False, np.float32, np.int32, np.int64, (2,8,2), (2,8,2), (2,8,2))


if __name__ == '__main__':
    unittest.main()
