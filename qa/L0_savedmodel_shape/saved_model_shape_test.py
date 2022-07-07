# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
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
import os

np_dtype_string = np.dtype(object)


class SavedModelShapeTest(tu.TestResultCollector):

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
                                use_http=True,
                                use_grpc=True,
                                skip_request_id_check=False,
                                use_streaming=True,
                                correlation_id=0):
            for bs in (1, batch_size):
                # model that does not support batching
                if bs == 1:
                    iu.infer_exact(tester,
                                   "savedmodel_nobatch",
                                   tensor_shape,
                                   bs,
                                   input_dtype,
                                   output0_dtype,
                                   output1_dtype,
                                   output0_raw=output0_raw,
                                   output1_raw=output1_raw,
                                   model_version=model_version,
                                   swap=swap,
                                   outputs=outputs,
                                   use_http=use_http,
                                   use_grpc=use_grpc,
                                   skip_request_id_check=skip_request_id_check,
                                   use_streaming=use_streaming,
                                   correlation_id=correlation_id)
                # model that supports batching
                iu.infer_exact(tester,
                               "savedmodel", (bs,) + tensor_shape,
                               bs,
                               input_dtype,
                               output0_dtype,
                               output1_dtype,
                               output0_raw=output0_raw,
                               output1_raw=output1_raw,
                               model_version=model_version,
                               swap=swap,
                               outputs=outputs,
                               use_http=use_http,
                               use_grpc=use_grpc,
                               skip_request_id_check=skip_request_id_check,
                               use_streaming=use_streaming,
                               correlation_id=correlation_id)

        input_size = 16

        if tu.validate_for_tf_model(input_dtype, output0_dtype, output1_dtype,
                                    (input_size,), (input_size,),
                                    (input_size,)):
            _infer_exact_helper(self,
                                "savedmodel", (input_size,),
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


if __name__ == '__main__':
    unittest.main()
