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

import unittest
import numpy as np
import infer_util as iu
import test_util as tu
import os

from tritonclientutils import *

TEST_SYSTEM_SHARED_MEMORY = bool(
    int(os.environ.get('TEST_SYSTEM_SHARED_MEMORY', 0)))
TEST_CUDA_SHARED_MEMORY = bool(int(os.environ.get('TEST_CUDA_SHARED_MEMORY',
                                                  0)))
CPU_ONLY = (os.environ.get('TRITON_SERVER_CPU_ONLY') is not None)
USE_GRPC = (os.environ.get('USE_GRPC', 1) != "0")
USE_HTTP = (os.environ.get('USE_HTTP', 1) != "0")
BACKENDS = os.environ.get(
    'BACKENDS', "graphdef savedmodel onnx libtorch plan custom python")
ENSEMBLES = bool(int(os.environ.get('ENSEMBLES', 1)))

np_dtype_string = np.dtype(object)


class InferTest(tu.TestResultCollector):

    def _full_exact(self, input_dtype, output0_dtype, output1_dtype,
                    output0_raw, output1_raw, swap):

        def _infer_exact_helper(tester,
                                pf,
                                tensor_shape,
                                batch_size,
                                input_dtype,
                                output0_dtype,
                                output1_dtype,
                                output0_raw=True,
                                output1_raw=True,
                                model_version=None,
                                swap=False,
                                outputs=("OUTPUT0", "OUTPUT1"),
                                use_http=USE_HTTP,
                                use_grpc=USE_GRPC,
                                use_http_json_tensors=True,
                                skip_request_id_check=True,
                                use_streaming=True,
                                correlation_id=0):
            for bs in (1, batch_size):
                # model that does not support batching
                if bs == 1:
                    iu.infer_exact(
                        tester,
                        pf + "_nobatch",
                        tensor_shape,
                        bs,
                        input_dtype,
                        output0_dtype,
                        output1_dtype,
                        output0_raw,
                        output1_raw,
                        model_version,
                        swap,
                        outputs,
                        USE_HTTP,
                        USE_GRPC,
                        use_http_json_tensors,
                        skip_request_id_check,
                        use_streaming,
                        correlation_id,
                        use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                        use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)
                # model that supports batching
                iu.infer_exact(
                    tester,
                    pf, (bs,) + tensor_shape,
                    bs,
                    input_dtype,
                    output0_dtype,
                    output1_dtype,
                    output0_raw,
                    output1_raw,
                    model_version,
                    swap,
                    outputs,
                    USE_HTTP,
                    USE_GRPC,
                    use_http_json_tensors,
                    skip_request_id_check,
                    use_streaming,
                    correlation_id,
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

        input_size = 16

        all_ensemble_prefix = ["simple_", "sequence_", "fan_"]
        ensemble_prefix = [""]
        if ENSEMBLES and ("custom" in BACKENDS):
            for prefix in all_ensemble_prefix:
                if tu.validate_for_ensemble_model(prefix, input_dtype,
                                                  output0_dtype, output1_dtype,
                                                  (input_size,), (input_size,),
                                                  (input_size,)):
                    ensemble_prefix.append(prefix)

        if tu.validate_for_tf_model(input_dtype, output0_dtype, output1_dtype,
                                    (input_size,), (input_size,),
                                    (input_size,)):
            for prefix in ensemble_prefix:
                for pf in ["graphdef", "savedmodel"]:
                    if pf in BACKENDS:
                        _infer_exact_helper(self,
                                            prefix + pf, (input_size,),
                                            8,
                                            input_dtype,
                                            output0_dtype,
                                            output1_dtype,
                                            output0_raw=output0_raw,
                                            output1_raw=output1_raw,
                                            swap=swap)

        if not CPU_ONLY and tu.validate_for_trt_model(
                input_dtype, output0_dtype, output1_dtype, (input_size, 1, 1),
            (input_size, 1, 1), (input_size, 1, 1)):
            for prefix in ensemble_prefix:
                if 'plan' in BACKENDS:
                    if input_dtype == np.int8:
                        _infer_exact_helper(self,
                                            prefix + 'plan', (input_size, 1, 1),
                                            8,
                                            input_dtype,
                                            output0_dtype,
                                            output1_dtype,
                                            output0_raw=output0_raw,
                                            output1_raw=output1_raw,
                                            swap=swap)
                    else:
                        _infer_exact_helper(self,
                                            prefix + 'plan', (input_size,),
                                            8,
                                            input_dtype,
                                            output0_dtype,
                                            output1_dtype,
                                            output0_raw=output0_raw,
                                            output1_raw=output1_raw,
                                            swap=swap)

        # the custom model is src/custom/addsub... it does not swap
        # the inputs so always set to False
        if tu.validate_for_custom_model(input_dtype, output0_dtype,
                                        output1_dtype, (input_size,),
                                        (input_size,), (input_size,)):
            # No basic ensemble models are created against custom models
            if 'custom' in BACKENDS:
                _infer_exact_helper(self,
                                    'custom', (input_size,),
                                    8,
                                    input_dtype,
                                    output0_dtype,
                                    output1_dtype,
                                    output0_raw=output0_raw,
                                    output1_raw=output1_raw,
                                    swap=False)

        if tu.validate_for_onnx_model(input_dtype, output0_dtype, output1_dtype,
                                      (input_size,), (input_size,),
                                      (input_size,)):
            for prefix in ensemble_prefix:
                if 'onnx' in BACKENDS:
                    _infer_exact_helper(self,
                                        prefix + 'onnx', (input_size,),
                                        8,
                                        input_dtype,
                                        output0_dtype,
                                        output1_dtype,
                                        output0_raw=output0_raw,
                                        output1_raw=output1_raw,
                                        swap=swap)

        if tu.validate_for_libtorch_model(input_dtype, output0_dtype,
                                          output1_dtype, (input_size,),
                                          (input_size,), (input_size,)):
            for prefix in ensemble_prefix:
                if 'libtorch' in BACKENDS:
                    _infer_exact_helper(self,
                                        prefix + 'libtorch', (input_size,),
                                        8,
                                        input_dtype,
                                        output0_dtype,
                                        output1_dtype,
                                        output0_raw=output0_raw,
                                        output1_raw=output1_raw,
                                        swap=swap)

        if prefix == "":
            if 'python' in BACKENDS:
                _infer_exact_helper(self,
                                    prefix + 'python', (input_size,),
                                    8,
                                    input_dtype,
                                    output0_dtype,
                                    output1_dtype,
                                    output0_raw=output0_raw,
                                    output1_raw=output1_raw,
                                    swap=swap)

    def test_raw_bbb(self):
        self._full_exact(np.int8,
                         np.int8,
                         np.int8,
                         output0_raw=True,
                         output1_raw=True,
                         swap=True)

    def test_raw_sss(self):
        self._full_exact(np.int16,
                         np.int16,
                         np.int16,
                         output0_raw=True,
                         output1_raw=True,
                         swap=True)

    def test_raw_iii(self):
        self._full_exact(np.int32,
                         np.int32,
                         np.int32,
                         output0_raw=True,
                         output1_raw=True,
                         swap=True)

    def test_raw_lll(self):
        self._full_exact(np.int64,
                         np.int64,
                         np.int64,
                         output0_raw=True,
                         output1_raw=True,
                         swap=False)

    def test_raw_hhh(self):
        self._full_exact(np.float16,
                         np.float16,
                         np.float16,
                         output0_raw=True,
                         output1_raw=True,
                         swap=False)

    def test_raw_fff(self):
        self._full_exact(np.float32,
                         np.float32,
                         np.float32,
                         output0_raw=True,
                         output1_raw=True,
                         swap=True)

    def test_raw_hff(self):
        self._full_exact(np.float16,
                         np.float32,
                         np.float32,
                         output0_raw=True,
                         output1_raw=True,
                         swap=False)

    def test_raw_bii(self):
        self._full_exact(np.int8,
                         np.int32,
                         np.int32,
                         output0_raw=True,
                         output1_raw=True,
                         swap=False)

    def test_raw_ibb(self):
        self._full_exact(np.int32,
                         np.int8,
                         np.int8,
                         output0_raw=True,
                         output1_raw=True,
                         swap=False)

    def test_raw_ibs(self):
        self._full_exact(np.int32,
                         np.int8,
                         np.int16,
                         output0_raw=True,
                         output1_raw=True,
                         swap=False)

    def test_raw_iff(self):
        self._full_exact(np.int32,
                         np.float32,
                         np.float32,
                         output0_raw=True,
                         output1_raw=True,
                         swap=False)

    def test_raw_fii(self):
        self._full_exact(np.float32,
                         np.int32,
                         np.int32,
                         output0_raw=True,
                         output1_raw=True,
                         swap=False)

    def test_raw_ihs(self):
        self._full_exact(np.int32,
                         np.float16,
                         np.int16,
                         output0_raw=True,
                         output1_raw=True,
                         swap=False)

    def test_raw_ooo(self):
        self._full_exact(np_dtype_string,
                         np_dtype_string,
                         np_dtype_string,
                         output0_raw=True,
                         output1_raw=True,
                         swap=False)

    def test_raw_oii(self):
        self._full_exact(np_dtype_string,
                         np.int32,
                         np.int32,
                         output0_raw=True,
                         output1_raw=True,
                         swap=False)

    def test_raw_oio(self):
        self._full_exact(np_dtype_string,
                         np.int32,
                         np_dtype_string,
                         output0_raw=True,
                         output1_raw=True,
                         swap=False)

    def test_raw_ooi(self):
        self._full_exact(np_dtype_string,
                         np_dtype_string,
                         np.int32,
                         output0_raw=True,
                         output1_raw=True,
                         swap=False)

    def test_raw_ioo(self):
        self._full_exact(np.int32,
                         np_dtype_string,
                         np_dtype_string,
                         output0_raw=True,
                         output1_raw=True,
                         swap=False)

    def test_raw_iio(self):
        self._full_exact(np.int32,
                         np.int32,
                         np_dtype_string,
                         output0_raw=True,
                         output1_raw=True,
                         swap=False)

    def test_raw_ioi(self):
        self._full_exact(np.int32,
                         np_dtype_string,
                         np.int32,
                         output0_raw=True,
                         output1_raw=True,
                         swap=False)

    # shared memory does not support class output
    if not (TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY):

        def test_class_bbb(self):
            self._full_exact(np.int8,
                             np.int8,
                             np.int8,
                             output0_raw=False,
                             output1_raw=False,
                             swap=True)

        def test_class_sss(self):
            self._full_exact(np.int16,
                             np.int16,
                             np.int16,
                             output0_raw=False,
                             output1_raw=False,
                             swap=True)

        def test_class_iii(self):
            self._full_exact(np.int32,
                             np.int32,
                             np.int32,
                             output0_raw=False,
                             output1_raw=False,
                             swap=True)

        def test_class_lll(self):
            self._full_exact(np.int64,
                             np.int64,
                             np.int64,
                             output0_raw=False,
                             output1_raw=False,
                             swap=False)

        def test_class_fff(self):
            self._full_exact(np.float32,
                             np.float32,
                             np.float32,
                             output0_raw=False,
                             output1_raw=False,
                             swap=True)

        def test_class_iff(self):
            self._full_exact(np.int32,
                             np.float32,
                             np.float32,
                             output0_raw=False,
                             output1_raw=False,
                             swap=False)

        def test_mix_bbb(self):
            self._full_exact(np.int8,
                             np.int8,
                             np.int8,
                             output0_raw=True,
                             output1_raw=False,
                             swap=True)

        def test_mix_sss(self):
            self._full_exact(np.int16,
                             np.int16,
                             np.int16,
                             output0_raw=False,
                             output1_raw=True,
                             swap=True)

        def test_mix_iii(self):
            self._full_exact(np.int32,
                             np.int32,
                             np.int32,
                             output0_raw=True,
                             output1_raw=False,
                             swap=True)

        def test_mix_lll(self):
            self._full_exact(np.int64,
                             np.int64,
                             np.int64,
                             output0_raw=False,
                             output1_raw=True,
                             swap=False)

        def test_mix_fff(self):
            self._full_exact(np.float32,
                             np.float32,
                             np.float32,
                             output0_raw=True,
                             output1_raw=False,
                             swap=True)

        def test_mix_iff(self):
            self._full_exact(np.int32,
                             np.float32,
                             np.float32,
                             output0_raw=False,
                             output1_raw=True,
                             swap=False)

    def test_raw_version_latest_1(self):
        input_size = 16
        tensor_shape = (1, input_size)

        # There are 3 versions of graphdef_int8_int8_int8 but
        # only version 3 should be available
        for platform in ('graphdef', 'savedmodel'):
            if platform not in BACKENDS:
                continue
            try:
                iu.infer_exact(
                    self,
                    platform,
                    tensor_shape,
                    1,
                    np.int8,
                    np.int8,
                    np.int8,
                    model_version=1,
                    swap=False,
                    use_http=USE_HTTP,
                    use_grpc=USE_GRPC,
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)
            except InferenceServerException as ex:
                self.assertTrue(
                    ex.message().startswith("Request for unknown model"))

            try:
                iu.infer_exact(
                    self,
                    platform,
                    tensor_shape,
                    1,
                    np.int8,
                    np.int8,
                    np.int8,
                    model_version=2,
                    swap=True,
                    use_http=USE_HTTP,
                    use_grpc=USE_GRPC,
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)
            except InferenceServerException as ex:
                self.assertTrue(
                    ex.message().startswith("Request for unknown model"))

            iu.infer_exact(self,
                           platform,
                           tensor_shape,
                           1,
                           np.int8,
                           np.int8,
                           np.int8,
                           model_version=3,
                           swap=True,
                           use_http=USE_HTTP,
                           use_grpc=USE_GRPC,
                           use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                           use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

    def test_raw_version_latest_2(self):
        input_size = 16
        tensor_shape = (1, input_size)

        # There are 3 versions of graphdef_int16_int16_int16 but only
        # versions 2 and 3 should be available
        for platform in ('graphdef', 'savedmodel'):
            if platform not in BACKENDS:
                continue
            try:
                iu.infer_exact(
                    self,
                    platform,
                    tensor_shape,
                    1,
                    np.int16,
                    np.int16,
                    np.int16,
                    model_version=1,
                    swap=False,
                    use_http=USE_HTTP,
                    use_grpc=USE_GRPC,
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)
            except InferenceServerException as ex:
                self.assertTrue(
                    ex.message().startswith("Request for unknown model"))

            iu.infer_exact(self,
                           platform,
                           tensor_shape,
                           1,
                           np.int16,
                           np.int16,
                           np.int16,
                           model_version=2,
                           swap=True,
                           use_http=USE_HTTP,
                           use_grpc=USE_GRPC,
                           use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                           use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)
            iu.infer_exact(self,
                           platform,
                           tensor_shape,
                           1,
                           np.int16,
                           np.int16,
                           np.int16,
                           model_version=3,
                           swap=True,
                           use_http=USE_HTTP,
                           use_grpc=USE_GRPC,
                           use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                           use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

    def test_raw_version_all(self):
        input_size = 16
        tensor_shape = (1, input_size)

        # There are 3 versions of *_int32_int32_int32 and all should
        # be available.
        for platform in ('graphdef', 'savedmodel'):
            if platform not in BACKENDS:
                continue
            iu.infer_exact(self,
                           platform,
                           tensor_shape,
                           1,
                           np.int32,
                           np.int32,
                           np.int32,
                           model_version=1,
                           swap=False,
                           use_http=USE_HTTP,
                           use_grpc=USE_GRPC,
                           use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                           use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)
            iu.infer_exact(self,
                           platform,
                           tensor_shape,
                           1,
                           np.int32,
                           np.int32,
                           np.int32,
                           model_version=2,
                           swap=True,
                           use_http=USE_HTTP,
                           use_grpc=USE_GRPC,
                           use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                           use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)
            iu.infer_exact(self,
                           platform,
                           tensor_shape,
                           1,
                           np.int32,
                           np.int32,
                           np.int32,
                           model_version=3,
                           swap=True,
                           use_http=USE_HTTP,
                           use_grpc=USE_GRPC,
                           use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                           use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

    def test_raw_version_specific_1(self):
        input_size = 16
        tensor_shape = (1, input_size)

        # There are 3 versions of *_float16_float16_float16 but only
        # version 1 should be available.
        for platform in ('graphdef', 'savedmodel'):
            if platform not in BACKENDS:
                continue
            iu.infer_exact(self,
                           platform,
                           tensor_shape,
                           1,
                           np.float16,
                           np.float16,
                           np.float16,
                           model_version=1,
                           swap=False,
                           use_http=USE_HTTP,
                           use_grpc=USE_GRPC,
                           use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                           use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

            try:
                iu.infer_exact(
                    self,
                    platform,
                    tensor_shape,
                    1,
                    np.float16,
                    np.float16,
                    np.float16,
                    model_version=2,
                    swap=True,
                    use_http=USE_HTTP,
                    use_grpc=USE_GRPC,
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)
            except InferenceServerException as ex:
                self.assertTrue(
                    ex.message().startswith("Request for unknown model"))

            try:
                iu.infer_exact(
                    self,
                    platform,
                    tensor_shape,
                    1,
                    np.float16,
                    np.float16,
                    np.float16,
                    model_version=3,
                    swap=True,
                    use_http=USE_HTTP,
                    use_grpc=USE_GRPC,
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)
            except InferenceServerException as ex:
                self.assertTrue(
                    ex.message().startswith("Request for unknown model"))

    def test_raw_version_specific_1_3(self):
        input_size = 16

        # There are 3 versions of *_float32_float32_float32 but only
        # versions 1 and 3 should be available.
        for platform in ('graphdef', 'savedmodel', 'plan'):
            if platform == 'plan' and CPU_ONLY:
                continue
            if platform not in BACKENDS:
                continue
            tensor_shape = (1, input_size)
            iu.infer_exact(self,
                           platform,
                           tensor_shape,
                           1,
                           np.float32,
                           np.float32,
                           np.float32,
                           model_version=1,
                           swap=False,
                           use_http=USE_HTTP,
                           use_grpc=USE_GRPC,
                           use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                           use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

            try:
                iu.infer_exact(
                    self,
                    platform,
                    tensor_shape,
                    1,
                    np.float32,
                    np.float32,
                    np.float32,
                    model_version=2,
                    swap=True,
                    use_http=USE_HTTP,
                    use_grpc=USE_GRPC,
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)
            except InferenceServerException as ex:
                self.assertTrue(
                    ex.message().startswith("Request for unknown model"))

            iu.infer_exact(self,
                           platform,
                           tensor_shape,
                           1,
                           np.float32,
                           np.float32,
                           np.float32,
                           model_version=3,
                           swap=True,
                           use_http=USE_HTTP,
                           use_grpc=USE_GRPC,
                           use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                           use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

    if ENSEMBLES:
        if all(x in BACKENDS for x in ['graphdef', 'savedmodel']):

            def test_ensemble_mix_platform(self):
                # Skip on CPU only machine as TensorRT model is used in this ensemble
                if CPU_ONLY:
                    return
                for bs in (1, 8):
                    iu.infer_exact(
                        self,
                        "mix_platform", (bs, 16),
                        bs,
                        np.float32,
                        np.float32,
                        np.float32,
                        use_http=USE_HTTP,
                        use_grpc=USE_GRPC,
                        use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                        use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

        if "graphdef" in BACKENDS:

            def test_ensemble_mix_type(self):
                for bs in (1, 8):
                    iu.infer_exact(
                        self,
                        "mix_type", (bs, 16),
                        bs,
                        np.int32,
                        np.float32,
                        np.float32,
                        use_http=USE_HTTP,
                        use_grpc=USE_GRPC,
                        use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                        use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

        if all(x in BACKENDS for x in ['graphdef', 'savedmodel']):

            def test_ensemble_mix_ensemble(self):
                for bs in (1, 8):
                    iu.infer_exact(
                        self,
                        "mix_ensemble", (bs, 16),
                        bs,
                        np.int32,
                        np.float32,
                        np.float32,
                        use_http=USE_HTTP,
                        use_grpc=USE_GRPC,
                        use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                        use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

        if all(x in BACKENDS for x in ['graphdef', 'custom']):

            def test_ensemble_mix_batch_nobatch(self):
                base_names = ["batch_to_nobatch", "nobatch_to_batch"]
                for name in base_names:
                    for bs in (1, 8):
                        iu.infer_exact(
                            self,
                            name, (bs, 16),
                            bs,
                            np.float32,
                            np.float32,
                            np.float32,
                            use_http=USE_HTTP,
                            use_grpc=USE_GRPC,
                            use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                            use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)
                    iu.infer_exact(
                        self,
                        name + "_nobatch", (8, 16),
                        1,
                        np.float32,
                        np.float32,
                        np.float32,
                        use_http=USE_HTTP,
                        use_grpc=USE_GRPC,
                        use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                        use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

                # batch -> nobatch -> batch
                for bs in (1, 8):
                    iu.infer_exact(
                        self,
                        "mix_nobatch_batch", (bs, 16),
                        bs,
                        np.float32,
                        np.float32,
                        np.float32,
                        use_http=USE_HTTP,
                        use_grpc=USE_GRPC,
                        use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                        use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

        if not (TEST_SYSTEM_SHARED_MEMORY or TEST_CUDA_SHARED_MEMORY):

            def test_ensemble_label_lookup(self):
                if all(x in BACKENDS for x in ['graphdef', 'savedmodel']):
                    # Ensemble needs to look up label from the actual model
                    for bs in (1, 8):
                        iu.infer_exact(
                            self,
                            "mix_platform", (bs, 16),
                            bs,
                            np.float32,
                            np.float32,
                            np.float32,
                            output0_raw=False,
                            output1_raw=False,
                            use_http=USE_HTTP,
                            use_grpc=USE_GRPC,
                            use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                            use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

                if all(x in BACKENDS for x in ['graphdef', 'savedmodel']):
                    # Label from the actual model will be passed along the nested ensemble
                    for bs in (1, 8):
                        iu.infer_exact(
                            self,
                            "mix_ensemble", (bs, 16),
                            bs,
                            np.int32,
                            np.float32,
                            np.float32,
                            output0_raw=False,
                            output1_raw=False,
                            use_http=USE_HTTP,
                            use_grpc=USE_GRPC,
                            use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                            use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

                if "graphdef" in BACKENDS:
                    # If label file is provided, it will use the provided label file directly
                    try:
                        iu.infer_exact(
                            self,
                            "wrong_label", (1, 16),
                            1,
                            np.int32,
                            np.float32,
                            np.float32,
                            output0_raw=False,
                            output1_raw=False,
                            use_http=USE_HTTP,
                            use_grpc=USE_GRPC,
                            use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                            use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)
                    except AssertionError:
                        # Sanity check that infer_exact failed since this ensemble is provided
                        # with unexpected labels
                        pass

                if "graphdef" in BACKENDS:
                    for bs in (1, 8):
                        iu.infer_exact(
                            self,
                            "label_override", (bs, 16),
                            bs,
                            np.int32,
                            np.float32,
                            np.float32,
                            output0_raw=False,
                            output1_raw=False,
                            use_http=USE_HTTP,
                            use_grpc=USE_GRPC,
                            use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                            use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)


if __name__ == '__main__':
    unittest.main()
