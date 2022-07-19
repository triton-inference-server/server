# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Allow caller to setup specific set of backends to test
DEFAULT_BACKENDS="graphdef savedmodel plan onnx libtorch"
TEST_BACKENDS = os.environ.get("BACKENDS", DEFAULT_BACKENDS).split()

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
                                use_http=True,
                                use_grpc=True,
                                skip_request_id_check=False,
                                use_streaming=True,
                                correlation_id=0):
            for bs in (1, batch_size):
                iu.infer_exact(tester,
                               pf, (bs,) + tensor_shape,
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
            for pf in ["graphdef", "savedmodel"]:
                if pf in TEST_BACKENDS:
                    _infer_exact_helper(self,
                                        pf, (input_size,),
                                        8,
                                        input_dtype,
                                        output0_dtype,
                                        output1_dtype,
                                        output0_raw=output0_raw,
                                        output1_raw=output1_raw,
                                        swap=swap)

        if tu.validate_for_trt_model(input_dtype, output0_dtype, output1_dtype,
                                     (input_size, 1, 1), (input_size, 1, 1),
                                     (input_size, 1, 1)):
            if "plan" in TEST_BACKENDS:
                if input_dtype == np.int8:
                    shape = (input_size, 1, 1)
                else:
                    shape = (input_size,)
                _infer_exact_helper(self,
                                    'plan', shape,
                                    8,
                                    input_dtype,
                                    output0_dtype,
                                    output1_dtype,
                                    output0_raw=output0_raw,
                                    output1_raw=output1_raw,
                                    swap=swap)

        if tu.validate_for_onnx_model(input_dtype, output0_dtype, output1_dtype,
                                      (input_size,), (input_size,),
                                      (input_size,)):
            if "onnx" in TEST_BACKENDS:
                _infer_exact_helper(self,
                                    'onnx', (input_size,),
                                    8,
                                    input_dtype,
                                    output0_dtype,
                                    output1_dtype,
                                    output0_raw=output0_raw,
                                    output1_raw=output1_raw,
                                    swap=swap)

        # Skip for batched string I/O
        if tu.validate_for_libtorch_model(input_dtype, output0_dtype,
                                          output1_dtype, (input_size,),
                                          (input_size,), (input_size,), 8):
            if "libtorch" in TEST_BACKENDS:
                _infer_exact_helper(self,
                                    'libtorch', (input_size,),
                                    8,
                                    input_dtype,
                                    output0_dtype,
                                    output1_dtype,
                                    output0_raw=output0_raw,
                                    output1_raw=output1_raw,
                                    swap=swap)

    def test_raw_fff(self):
        self._full_exact(np.float32,
                         np.float32,
                         np.float32,
                         output0_raw=True,
                         output1_raw=True,
                         swap=True)

    def test_raw_ooo(self):
        self._full_exact(np_dtype_string,
                         np_dtype_string,
                         np_dtype_string,
                         output0_raw=True,
                         output1_raw=True,
                         swap=False)

    def test_class_fff(self):
        self._full_exact(np.float32,
                         np.float32,
                         np.float32,
                         output0_raw=False,
                         output1_raw=False,
                         swap=True)


if __name__ == '__main__':
    unittest.main()
