# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
import infer_util as iu
import test_util as tu
from tritonclientutils import *


class TrtCudaGraphTest(tu.TestResCollector):

    def setUp(self):
        self.dtype_ = np.float32
        self.model_name_ = 'plan'

    def _check_infer(self, tensor_shape, batch_size=1):
        try:
            iu.infer_exact(self,
                           self.model_name_, (batch_size,) + tensor_shape,
                           batch_size,
                           self.dtype_,
                           self.dtype_,
                           self.dtype_,
                           model_version=1,
                           use_http_json_tensors=False,
                           use_grpc=False,
                           use_streaming=False)
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_fixed_shape(self):
        tensor_shape = (16,)
        self._check_infer(tensor_shape)
        # Inference that should not have CUDA graph captured
        self._check_infer(tensor_shape, 5)

    def test_dynamic_shape(self):
        tensor_shape = (20,)
        self._check_infer(tensor_shape)
        # Inference that should not have CUDA graph captured
        self._check_infer(tensor_shape, 5)


if __name__ == '__main__':
    unittest.main()
