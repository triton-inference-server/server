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
import os
if "TEST_SHARED_MEMORY" in os.environ:
    TEST_SHARED_MEMORY=int(os.environ["TEST_SHARED_MEMORY"])
else:
    TEST_SHARED_MEMORY=0

CPU_ONLY = (os.environ.get('TENSORRT_SERVER_CPU_ONLY') is not None)

np_dtype_string = np.dtype(object)

class InferTest(unittest.TestCase):
    def _full_exact(self, input_dtype, output0_dtype, output1_dtype,
                    output0_raw, output1_raw, swap):
        def _infer_exact_helper(tester, pf, tensor_shape, batch_size,
                input_dtype, output0_dtype, output1_dtype,
                output0_raw=True, output1_raw=True,
                model_version=None, swap=False,
                outputs=("OUTPUT0", "OUTPUT1"), use_http=True, use_grpc=True,
                skip_request_id_check=False, use_streaming=True,
                correlation_id=0):
            for bs in (1, batch_size):
                # model that does not support batching
                if bs == 1:
                    iu.infer_exact(tester, pf + "_nobatch", tensor_shape, bs,
                                    input_dtype, output0_dtype, output1_dtype,
                                    output0_raw, output1_raw,
                                    model_version, swap,
                                    outputs, use_http, use_grpc,
                                    skip_request_id_check, use_streaming,
                                    correlation_id)
                # model that supports batching
                iu.infer_exact(tester, pf, tensor_shape, bs,
                               input_dtype, output0_dtype, output1_dtype,
                               output0_raw, output1_raw,
                               model_version, swap, outputs, use_http, use_grpc,
                               skip_request_id_check, use_streaming,
                               correlation_id)

        input_size = 16

        all_ensemble_prefix = ["simple_", "sequence_", "fan_"]
        ensemble_prefix = [""]
        for prefix in all_ensemble_prefix:
            if tu.validate_for_ensemble_model(prefix,
                                    input_dtype, output0_dtype, output1_dtype,
                                    (input_size,), (input_size,), (input_size,)):
                ensemble_prefix.append(prefix)

        if tu.validate_for_tf_model(input_dtype, output0_dtype, output1_dtype,
                                    (input_size,), (input_size,), (input_size,)):
            for prefix in ensemble_prefix:
                for pf in ["graphdef", "savedmodel"]:
                    _infer_exact_helper(self, prefix + pf, (input_size,), 8,
                                    input_dtype, output0_dtype, output1_dtype,
                                    output0_raw=output0_raw, output1_raw=output1_raw, swap=swap)


        if tu.validate_for_c2_model(input_dtype, output0_dtype, output1_dtype,
                                    (input_size,), (input_size,), (input_size,)):
            for prefix in ensemble_prefix:
                _infer_exact_helper(self, prefix + 'netdef', (input_size,), 8,
                                input_dtype, output0_dtype, output1_dtype,
                                output0_raw=output0_raw, output1_raw=output1_raw, swap=swap)

        if not CPU_ONLY and tu.validate_for_trt_model(input_dtype, output0_dtype, output1_dtype,
                                    (input_size,1,1), (input_size,1,1), (input_size,1,1)):
            for prefix in ensemble_prefix:
                _infer_exact_helper(self, prefix + 'plan', (input_size, 1, 1), 8,
                                input_dtype, output0_dtype, output1_dtype,
                                output0_raw=output0_raw, output1_raw=output1_raw, swap=swap)

        # the custom model is src/custom/addsub... it does not swap
        # the inputs so always set to False
        if tu.validate_for_custom_model(input_dtype, output0_dtype, output1_dtype,
                                        (input_size,), (input_size,), (input_size,)):
            # No basic ensemble models are created against custom models
            _infer_exact_helper(self, 'custom', (input_size,), 8,
                            input_dtype, output0_dtype, output1_dtype,
                            output0_raw=output0_raw, output1_raw=output1_raw, swap=False)

        if tu.validate_for_onnx_model(input_dtype, output0_dtype, output1_dtype,
                                    (input_size,), (input_size,), (input_size,)):
            # No basic ensemble models are created against onnx models for now [TODO]
            _infer_exact_helper(self, 'onnx', (input_size,), 8,
                            input_dtype, output0_dtype, output1_dtype,
                            output0_raw=output0_raw, output1_raw=output1_raw, swap=swap)

        if tu.validate_for_libtorch_model(input_dtype, output0_dtype, output1_dtype,
                                    (input_size,), (input_size,), (input_size,)):
            # No basic ensemble models are created wuth libtorch models for now [TODO]
            _infer_exact_helper(self, 'libtorch', (input_size,), 8,
                            input_dtype, output0_dtype, output1_dtype,
                            output0_raw=output0_raw, output1_raw=output1_raw, swap=swap)

    def test_raw_bbb(self):
        self._full_exact(np.int8, np.int8, np.int8,
                         output0_raw=True, output1_raw=True, swap=True)
    def test_raw_sss(self):
        self._full_exact(np.int16, np.int16, np.int16,
                         output0_raw=True, output1_raw=True, swap=True)
    def test_raw_iii(self):
        self._full_exact(np.int32, np.int32, np.int32,
                         output0_raw=True, output1_raw=True, swap=True)
    def test_raw_lll(self):
        self._full_exact(np.int64, np.int64, np.int64,
                         output0_raw=True, output1_raw=True, swap=False)
    def test_raw_hhh(self):
        self._full_exact(np.float16, np.float16, np.float16,
                         output0_raw=True, output1_raw=True, swap=False)
    def test_raw_fff(self):
        self._full_exact(np.float32, np.float32, np.float32,
                         output0_raw=True, output1_raw=True, swap=True)
    def test_raw_hff(self):
        self._full_exact(np.float16, np.float32, np.float32,
                         output0_raw=True, output1_raw=True, swap=False)
    def test_raw_bii(self):
        self._full_exact(np.int8, np.int32, np.int32,
                         output0_raw=True, output1_raw=True, swap=False)
    def test_raw_ibb(self):
        self._full_exact(np.int32, np.int8, np.int8,
                         output0_raw=True, output1_raw=True, swap=False)
    def test_raw_ibs(self):
        self._full_exact(np.int32, np.int8, np.int16,
                         output0_raw=True, output1_raw=True, swap=False)
    def test_raw_iff(self):
        self._full_exact(np.int32, np.float32, np.float32,
                         output0_raw=True, output1_raw=True, swap=False)
    def test_raw_fii(self):
        self._full_exact(np.float32, np.int32, np.int32,
                         output0_raw=True, output1_raw=True, swap=False)
    def test_raw_ihs(self):
        self._full_exact(np.int32, np.float16, np.int16,
                         output0_raw=True, output1_raw=True, swap=False)

    if not TEST_SHARED_MEMORY:
        def test_raw_oii(self):
            self._full_exact(np_dtype_string, np.int32, np.int32,
                             output0_raw=True, output1_raw=True, swap=False)
        def test_raw_ooo(self):
            self._full_exact(np_dtype_string, np_dtype_string, np_dtype_string,
                             output0_raw=True, output1_raw=True, swap=False)
        def test_raw_oio(self):
            self._full_exact(np_dtype_string, np.int32, np_dtype_string,
                             output0_raw=True, output1_raw=True, swap=False)
        def test_raw_ooi(self):
            self._full_exact(np_dtype_string, np_dtype_string, np.int32,
                             output0_raw=True, output1_raw=True, swap=False)
        def test_raw_ioo(self):
            self._full_exact(np.int32, np_dtype_string, np_dtype_string,
                             output0_raw=True, output1_raw=True, swap=False)
        def test_raw_iio(self):
            self._full_exact(np.int32, np.int32, np_dtype_string,
                             output0_raw=True, output1_raw=True, swap=False)
        def test_raw_ioi(self):
            self._full_exact(np.int32, np_dtype_string, np.int32,
                             output0_raw=True, output1_raw=True, swap=False)

        def test_class_bbb(self):
            self._full_exact(np.int8, np.int8, np.int8,
                             output0_raw=False, output1_raw=False, swap=True)
        def test_class_sss(self):
            self._full_exact(np.int16, np.int16, np.int16,
                             output0_raw=False, output1_raw=False, swap=True)
        def test_class_iii(self):
            self._full_exact(np.int32, np.int32, np.int32,
                             output0_raw=False, output1_raw=False, swap=True)
        def test_class_lll(self):
            self._full_exact(np.int64, np.int64, np.int64,
                             output0_raw=False, output1_raw=False, swap=False)
        def test_class_fff(self):
            self._full_exact(np.float32, np.float32, np.float32,
                             output0_raw=False, output1_raw=False, swap=True)
        def test_class_iff(self):
            self._full_exact(np.int32, np.float32, np.float32,
                             output0_raw=False, output1_raw=False, swap=False)

        def test_mix_bbb(self):
            self._full_exact(np.int8, np.int8, np.int8,
                             output0_raw=True, output1_raw=False, swap=True)
        def test_mix_sss(self):
            self._full_exact(np.int16, np.int16, np.int16,
                             output0_raw=False, output1_raw=True, swap=True)
        def test_mix_iii(self):
            self._full_exact(np.int32, np.int32, np.int32,
                             output0_raw=True, output1_raw=False, swap=True)
        def test_mix_lll(self):
            self._full_exact(np.int64, np.int64, np.int64,
                             output0_raw=False, output1_raw=True, swap=False)
        def test_mix_fff(self):
            self._full_exact(np.float32, np.float32, np.float32,
                             output0_raw=True, output1_raw=False, swap=True)
        def test_mix_iff(self):
            self._full_exact(np.int32, np.float32, np.float32,
                             output0_raw=False, output1_raw=True, swap=False)

        def test_raw_version_latest_1(self):
            input_size = 16
            tensor_shape = (input_size,)

            # There are 3 versions of graphdef_int8_int8_int8 but
            # only version 3 should be available
            for platform in ('graphdef', 'savedmodel'):
                try:
                    iu.infer_exact(self, platform, tensor_shape, 1,
                                   np.int8, np.int8, np.int8,
                                   model_version=1, swap=False)
                except InferenceServerException as ex:
                    self.assertEqual("inference:0", ex.server_id())
                    self.assertTrue(
                        ex.message().startswith("Inference request for unknown model"))

                try:
                    iu.infer_exact(self, platform, tensor_shape, 1,
                                   np.int8, np.int8, np.int8,
                                   model_version=2, swap=True)
                except InferenceServerException as ex:
                    self.assertEqual("inference:0", ex.server_id())
                    self.assertTrue(
                        ex.message().startswith("Inference request for unknown model"))

                iu.infer_exact(self, platform, tensor_shape, 1,
                               np.int8, np.int8, np.int8,
                               model_version=3, swap=True)

        def test_raw_version_latest_2(self):
            input_size = 16
            tensor_shape = (input_size,)

            # There are 3 versions of graphdef_int16_int16_int16 but only
            # versions 2 and 3 should be available
            for platform in ('graphdef', 'savedmodel'):
                try:
                    iu.infer_exact(self, platform, tensor_shape, 1,
                                   np.int16, np.int16, np.int16,
                                   model_version=1, swap=False)
                except InferenceServerException as ex:
                    self.assertEqual("inference:0", ex.server_id())
                    self.assertTrue(
                        ex.message().startswith("Inference request for unknown model"))

                iu.infer_exact(self, platform, tensor_shape, 1,
                               np.int16, np.int16, np.int16,
                               model_version=2, swap=True)
                iu.infer_exact(self, platform, tensor_shape, 1,
                               np.int16, np.int16, np.int16,
                               model_version=3, swap=True)

        def test_raw_version_all(self):
            input_size = 16
            tensor_shape = (input_size,)

            # There are 3 versions of *_int32_int32_int32 and all should
            # be available.
            for platform in ('graphdef', 'savedmodel', 'netdef'):
                iu.infer_exact(self, platform, tensor_shape, 1,
                               np.int32, np.int32, np.int32,
                               model_version=1, swap=False)
                iu.infer_exact(self, platform, tensor_shape, 1,
                               np.int32, np.int32, np.int32,
                               model_version=2, swap=True)
                iu.infer_exact(self, platform, tensor_shape, 1,
                               np.int32, np.int32, np.int32,
                               model_version=3, swap=True)

        def test_raw_version_specific_1(self):
            input_size = 16
            tensor_shape = (input_size,)

            # There are 3 versions of *_float16_float16_float16 but only
            # version 1 should be available.
            for platform in ('graphdef', 'savedmodel'):
                iu.infer_exact(self, platform, tensor_shape, 1,
                               np.float16, np.float16, np.float16,
                               model_version=1, swap=False)

                try:
                    iu.infer_exact(self, platform, tensor_shape, 1,
                                   np.float16, np.float16, np.float16,
                                   model_version=2, swap=True)
                except InferenceServerException as ex:
                    self.assertEqual("inference:0", ex.server_id())
                    self.assertTrue(
                        ex.message().startswith("Inference request for unknown model"))

                try:
                    iu.infer_exact(self, platform, tensor_shape, 1,
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
                if platform == 'plan' and CPU_ONLY:
                    continue
                tensor_shape = (input_size, 1, 1) if platform == 'plan' else (input_size,)
                iu.infer_exact(self, platform, tensor_shape, 1,
                               np.float32, np.float32, np.float32,
                               model_version=1, swap=False)

                try:
                    iu.infer_exact(self, platform, tensor_shape, 1,
                                   np.float32, np.float32, np.float32,
                                   model_version=2, swap=True)
                except InferenceServerException as ex:
                    self.assertEqual("inference:0", ex.server_id())
                    self.assertTrue(
                        ex.message().startswith("Inference request for unknown model"))

                iu.infer_exact(self, platform, tensor_shape, 1,
                               np.float32, np.float32, np.float32,
                               model_version=3, swap=True)

    def test_ensemble_mix_platform(self):
        # Skip on CPU only machine as TensorRT model is used in this ensemble
        if CPU_ONLY:
            return
        for bs in (1, 8):
            iu.infer_exact(self, "mix_platform", (16,), bs,
                np.float32, np.float32, np.float32)

    def test_ensemble_mix_type(self):
        for bs in (1, 8):
            iu.infer_exact(self, "mix_type", (16,), bs,
                np.int32, np.float32, np.float32)

    def test_ensemble_mix_ensemble(self):
        for bs in (1, 8):
            iu.infer_exact(self, "mix_ensemble", (16,), bs,
                np.int32, np.float32, np.float32)

    def test_ensemble_mix_batch_nobatch(self):
        base_names = ["batch_to_nobatch", "nobatch_to_batch"]
        for name in base_names:
            for bs in (1, 8):
                iu.infer_exact(self, name, (16,), bs,
                    np.float32, np.float32, np.float32)
            iu.infer_exact(self, name + "_nobatch", (8, 16,), 1,
                np.float32, np.float32, np.float32)

        # batch -> nobatch -> batch
        for bs in (1, 8):
            iu.infer_exact(self, "mix_nobatch_batch", (16,), bs,
                np.float32, np.float32, np.float32)

    if not TEST_SHARED_MEMORY:
        def test_ensemble_label_lookup(self):
            # Ensemble needs to look up label from the actual model
            for bs in (1, 8):
                iu.infer_exact(self, "mix_platform", (16,), bs,
                    np.float32, np.float32, np.float32, output0_raw=False, output1_raw=False)

            # Label from the actual model will be passed along the nested ensemble
            for bs in (1, 8):
                iu.infer_exact(self, "mix_ensemble", (16,), bs,
                    np.int32, np.float32, np.float32, output0_raw=False, output1_raw=False)

            # If label file is provided, it will use the provided label file directly
            try:
                iu.infer_exact(self, "wrong_label", (16,), 1,
                    np.int32, np.float32, np.float32, output0_raw=False, output1_raw=False)
            except AssertionError:
                # Sanity check that infer_exact failed since this ensemble is provided
                # with unexpected labels
                pass

            for bs in (1, 8):
                iu.infer_exact(self, "label_override", (16,), bs,
                    np.int32, np.float32, np.float32, output0_raw=False, output1_raw=False)


if __name__ == '__main__':
    unittest.main()
