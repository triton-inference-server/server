# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

class InferTest(unittest.TestCase):
    def _full_exact(self, req_raw, input_dtype, output0_dtype, output1_dtype, swap):
        input_size = 16

        if tu.validate_for_tf_model(input_dtype, output0_dtype, output1_dtype,
                                    (input_size,), (input_size,), (input_size,)):
            # model that supports batching
            for bs in (1, 8):
                iu.infer_exact(self, 'graphdef', (input_size,), bs, req_raw,
                               input_dtype, output0_dtype, output1_dtype, swap=swap)
                iu.infer_exact(self, 'savedmodel', (input_size,), bs, req_raw,
                               input_dtype, output0_dtype, output1_dtype, swap=swap)
            # model that does not support batching
            iu.infer_exact(self, 'graphdef_nobatch', (input_size,), 1, req_raw,
                           input_dtype, output0_dtype, output1_dtype, swap=swap)
            iu.infer_exact(self, 'savedmodel_nobatch', (input_size,), 1, req_raw,
                           input_dtype, output0_dtype, output1_dtype, swap=swap)

        if tu.validate_for_c2_model(input_dtype, output0_dtype, output1_dtype,
                                    (input_size,), (input_size,), (input_size,)):
            # model that supports batching
            for bs in (1, 8):
                iu.infer_exact(self, 'netdef', (input_size,), bs, req_raw,
                               input_dtype, output0_dtype, output1_dtype, swap=swap)
            # model that does not support batching
            iu.infer_exact(self, 'netdef_nobatch', (input_size,), 1, req_raw,
                           input_dtype, output0_dtype, output1_dtype, swap=swap)

        if tu.validate_for_trt_model(input_dtype, output0_dtype, output1_dtype,
                                    (input_size,1,1), (input_size,1,1), (input_size,1,1)):
            # model that supports batching
            for bs in (1, 8):
                iu.infer_exact(self, 'plan', (input_size, 1, 1), bs, req_raw,
                               input_dtype, output0_dtype, output1_dtype, swap=swap)
            # model that does not support batching
            iu.infer_exact(self, 'plan_nobatch', (input_size, 1, 1), 1, req_raw,
                           input_dtype, output0_dtype, output1_dtype, swap=swap)

        # the custom model is src/custom/addsub... it does not swap
        # the inputs so always set to False
        if tu.validate_for_custom_model(input_dtype, output0_dtype, output1_dtype,
                                        (input_size,), (input_size,), (input_size,)):
            # model that supports batching
            for bs in (1, 8):
                iu.infer_exact(self, 'custom', (input_size,), bs, req_raw,
                               input_dtype, output0_dtype, output1_dtype, swap=False)
            # model that does not support batching
            iu.infer_exact(self, 'custom_nobatch', (input_size,), 1, req_raw,
                           input_dtype, output0_dtype, output1_dtype, swap=False)

    def test_raw_bbb(self):
        self._full_exact(True, np.int8, np.int8, np.int8, swap=True)
    def test_raw_sss(self):
        self._full_exact(True, np.int16, np.int16, np.int16, swap=True)
    def test_raw_iii(self):
        self._full_exact(True, np.int32, np.int32, np.int32, swap=True)
    def test_raw_lll(self):
        self._full_exact(True, np.int64, np.int64, np.int64, swap=False)
    def test_raw_hhh(self):
        self._full_exact(True, np.float16, np.float16, np.float16, swap=False)
    def test_raw_fff(self):
        self._full_exact(True, np.float32, np.float32, np.float32, swap=True)
    def test_raw_hff(self):
        self._full_exact(True, np.float16, np.float32, np.float32, swap=False)
    def test_raw_bii(self):
        self._full_exact(True, np.int8, np.int32, np.int32, swap=False)
    def test_raw_ibb(self):
        self._full_exact(True, np.int32, np.int8, np.int8, swap=False)
    def test_raw_ibs(self):
        self._full_exact(True, np.int32, np.int8, np.int16, swap=False)
    def test_raw_iff(self):
        self._full_exact(True, np.int32, np.float32, np.float32, swap=False)
    def test_raw_fii(self):
        self._full_exact(True, np.float32, np.int32, np.int32, swap=False)
    def test_raw_ihs(self):
        self._full_exact(True, np.int32, np.float16, np.int16, swap=False)

    def test_raw_oii(self):
        self._full_exact(True, np_dtype_string, np.int32, np.int32, swap=False)
    def test_raw_ooo(self):
        self._full_exact(True, np_dtype_string, np_dtype_string, np_dtype_string, swap=False)
    def test_raw_oio(self):
        self._full_exact(True, np_dtype_string, np.int32, np_dtype_string, swap=False)
    def test_raw_ooi(self):
        self._full_exact(True, np_dtype_string, np_dtype_string, np.int32, swap=False)
    def test_raw_ioo(self):
        self._full_exact(True, np.int32, np_dtype_string, np_dtype_string, swap=False)
    def test_raw_iio(self):
        self._full_exact(True, np.int32, np.int32, np_dtype_string, swap=False)
    def test_raw_ioi(self):
        self._full_exact(True, np.int32, np_dtype_string, np.int32, swap=False)

    def test_class_bbb(self):
        self._full_exact(False, np.int8, np.int8, np.int8, swap=True)
    def test_class_sss(self):
        self._full_exact(False, np.int16, np.int16, np.int16, swap=True)
    def test_class_iii(self):
        self._full_exact(False, np.int32, np.int32, np.int32, swap=True)
    def test_class_lll(self):
        self._full_exact(False, np.int64, np.int64, np.int64, swap=False)
    def test_class_fff(self):
        self._full_exact(False, np.float32, np.float32, np.float32, swap=True)
    def test_class_iff(self):
        self._full_exact(False, np.int32, np.float32, np.float32, swap=False)

    def test_raw_version_latest_1(self):
        input_size = 16
        tensor_shape = (input_size,)

        # There are 3 versions of graphdef_int8_int8_int8 but
        # only version 3 should be available
        for platform in ('graphdef', 'savedmodel'):
            try:
                iu.infer_exact(self, platform, tensor_shape, 1, True,
                               np.int8, np.int8, np.int8,
                               model_version=1, swap=False)
            except InferenceServerException as ex:
                self.assertEqual("inference:0", ex.server_id())
                self.assertTrue(
                    ex.message().startswith("Inference request for unknown model"))

            try:
                iu.infer_exact(self, platform, tensor_shape, 1, True,
                               np.int8, np.int8, np.int8,
                               model_version=2, swap=True)
            except InferenceServerException as ex:
                self.assertEqual("inference:0", ex.server_id())
                self.assertTrue(
                    ex.message().startswith("Inference request for unknown model"))

            iu.infer_exact(self, platform, tensor_shape, 1, True,
                           np.int8, np.int8, np.int8,
                           model_version=3, swap=True)

    def test_raw_version_latest_2(self):
        input_size = 16
        tensor_shape = (input_size,)

        # There are 3 versions of graphdef_int16_int16_int16 but only
        # versions 2 and 3 should be available
        for platform in ('graphdef', 'savedmodel'):
            try:
                iu.infer_exact(self, platform, tensor_shape, 1, True,
                               np.int16, np.int16, np.int16,
                               model_version=1, swap=False)
            except InferenceServerException as ex:
                self.assertEqual("inference:0", ex.server_id())
                self.assertTrue(
                    ex.message().startswith("Inference request for unknown model"))

            iu.infer_exact(self, platform, tensor_shape, 1, True,
                           np.int16, np.int16, np.int16,
                           model_version=2, swap=True)
            iu.infer_exact(self, platform, tensor_shape, 1, True,
                           np.int16, np.int16, np.int16,
                           model_version=3, swap=True)

    def test_raw_version_all(self):
        input_size = 16
        tensor_shape = (input_size,)

        # There are 3 versions of *_int32_int32_int32 and all should
        # be available.
        for platform in ('graphdef', 'savedmodel', 'netdef'):
            iu.infer_exact(self, platform, tensor_shape, 1, True,
                           np.int32, np.int32, np.int32,
                           model_version=1, swap=False)
            iu.infer_exact(self, platform, tensor_shape, 1, True,
                           np.int32, np.int32, np.int32,
                           model_version=2, swap=True)
            iu.infer_exact(self, platform, tensor_shape, 1, True,
                           np.int32, np.int32, np.int32,
                           model_version=3, swap=True)

    def test_raw_version_specific_1(self):
        input_size = 16
        tensor_shape = (input_size,)

        # There are 3 versions of *_float16_float16_float16 but only
        # version 1 should be available.
        for platform in ('graphdef', 'savedmodel'):
            iu.infer_exact(self, platform, tensor_shape, 1, True,
                           np.float16, np.float16, np.float16,
                           model_version=1, swap=False)

            try:
                iu.infer_exact(self, platform, tensor_shape, 1, True,
                               np.float16, np.float16, np.float16,
                               model_version=2, swap=True)
            except InferenceServerException as ex:
                self.assertEqual("inference:0", ex.server_id())
                self.assertTrue(
                    ex.message().startswith("Inference request for unknown model"))

            try:
                iu.infer_exact(self, platform, tensor_shape, 1, True,
                               np.float16, np.float16, np.float16,
                               model_version=3, swap=True)
            except InferenceServerException as ex:
                self.assertEqual("inference:0", ex.server_id())
                self.assertTrue(
                    ex.message().startswith("Inference request for unknown model"))

    def test_raw_version_specific_1_3(self):
        input_size = 16

        # There are 3 versions of *_float32_float32_float32 but only
        # versions 1 and 3 should be available.
        for platform in ('graphdef', 'savedmodel', 'netdef', 'plan'):
            tensor_shape = (input_size, 1, 1) if platform == 'plan' else (input_size,)
            iu.infer_exact(self, platform, tensor_shape, 1, True,
                           np.float32, np.float32, np.float32,
                           model_version=1, swap=False)

            try:
                iu.infer_exact(self, platform, tensor_shape, 1, True,
                               np.float32, np.float32, np.float32,
                               model_version=2, swap=True)
            except InferenceServerException as ex:
                self.assertEqual("inference:0", ex.server_id())
                self.assertTrue(
                    ex.message().startswith("Inference request for unknown model"))

            iu.infer_exact(self, platform, tensor_shape, 1, True,
                           np.float32, np.float32, np.float32,
                           model_version=3, swap=True)


if __name__ == '__main__':
    unittest.main()
