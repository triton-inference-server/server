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
import os
import shutil
import time
import unittest
import numpy as np
import infer_util as iu
import test_util as tu
from tensorrtserver.api import *
import tensorrtserver.api.server_status_pb2 as server_status

class TrtDynamicShapeTest(unittest.TestCase):
    def setUp(self):
        self.dtype_ = np.float32
        self.model_name_ = 'plan'

    def test_load_specific_optimization_profile(self):
        # Only OP 5 should be available, which only allow batch size 8
        tensor_shape = (1,)
        try:
            iu.infer_exact(self, self.model_name_, tensor_shape, 1,
                            self.dtype_, self.dtype_, self.dtype_)
        except InferenceServerException as ex:
            self.assertEqual("inference:0", ex.server_id())
            self.assertTrue(
              "model expected the shape of dimension 0 to be between 6 and 8 but received 1" in ex.message())

        try:
            iu.infer_exact(self, self.model_name_, tensor_shape, 8,
                            self.dtype_, self.dtype_, self.dtype_)
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_load_default_optimization_profile(self):
        # Only default OP (OP 0) has max tensor shape 33
        tensor_shape = (33,)

        try:
            iu.infer_exact(self, self.model_name_, tensor_shape, 8,
                            self.dtype_, self.dtype_, self.dtype_)
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        over_tensor_shape = (34,)
        try:
            iu.infer_exact(self, self.model_name_, over_tensor_shape, 8,
                            self.dtype_, self.dtype_, self.dtype_)
        except InferenceServerException as ex:
            self.assertEqual("inference:0", ex.server_id())
            self.assertTrue(
                    "model expected the shape of dimension 1 to be between 1 and 33 but received 34" in ex.message())

    def test_select_optimization_profile(self):
        # Different profile has different optimized input shape
        batch_size = 4
        tensor_shape = (16,)
        try:
            iu.infer_exact(self, self.model_name_, tensor_shape, batch_size,
                            self.dtype_, self.dtype_, self.dtype_)
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_load_wrong_optimization_profile(self):
        try:
            model_name = tu.get_model_name(self.model_name_, self.dtype_, self.dtype_, self.dtype_)
            ctx = ServerStatusContext("localhost:8000", ProtocolType.HTTP, model_name, True)
            ss = ctx.get_server_status()
            self.assertEqual(os.environ["TENSORRT_SERVER_VERSION"], ss.version)
            self.assertEqual("inference:0", ss.id)
            self.assertEqual(len(ss.model_status), 1)
            self.assertTrue(model_name in ss.model_status,
                            "expected status for model " + model_name)
            self.assertTrue(1 in ss.model_status[model_name].version_status,
                            "expected status for version 1 of model " + model_name)
            v = ss.model_status[model_name].version_status[1]
            self.assertEqual(v.ready_state, server_status.MODEL_UNAVAILABLE)
            self.assertNotEqual(len(v.ready_state_reason.message), 0,
                        "expected non-empty message for load failure")
        except InferenceServerException as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

        

if __name__ == '__main__':
    unittest.main()
