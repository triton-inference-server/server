# Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TEST_SYSTEM_SHARED_MEMORY = bool(
    int(os.environ.get('TEST_SYSTEM_SHARED_MEMORY', 0)))
TEST_CUDA_SHARED_MEMORY = bool(int(os.environ.get('TEST_CUDA_SHARED_MEMORY',
                                                  0)))


class InferReshapeTest(tu.TestResultCollector):

    def _full_reshape(self,
                      dtype,
                      input_shapes,
                      output_shapes=None,
                      no_batch=True):
        # 'shapes' is list of shapes, one for each input.
        if output_shapes is None:
            output_shapes = input_shapes

        # For validation assume any shape can be used...
        if tu.validate_for_tf_model(dtype, dtype, dtype, input_shapes[0],
                                    input_shapes[0], input_shapes[0]):
            # model that supports batching
            for bs in (1, 8):
                full_shapes = [[
                    bs,
                ] + input_shape for input_shape in input_shapes]
                full_output_shapes = [[
                    bs,
                ] + output_shape for output_shape in output_shapes]
                iu.infer_zero(
                    self,
                    'graphdef',
                    bs,
                    dtype,
                    full_shapes,
                    full_output_shapes,
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)
                iu.infer_zero(
                    self,
                    'savedmodel',
                    bs,
                    dtype,
                    full_shapes,
                    full_output_shapes,
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)
            # model that does not support batching
            if no_batch:
                iu.infer_zero(
                    self,
                    'graphdef_nobatch',
                    1,
                    dtype,
                    input_shapes,
                    output_shapes,
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)
                iu.infer_zero(
                    self,
                    'savedmodel_nobatch',
                    1,
                    dtype,
                    input_shapes,
                    output_shapes,
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

        if tu.validate_for_onnx_model(dtype, dtype, dtype, input_shapes[0],
                                      input_shapes[0], input_shapes[0]):
            # model that supports batching
            for bs in (1, 8):
                full_shapes = [[
                    bs,
                ] + input_shape for input_shape in input_shapes]
                full_output_shapes = [[
                    bs,
                ] + output_shape for output_shape in output_shapes]
                iu.infer_zero(
                    self,
                    'onnx',
                    bs,
                    dtype,
                    full_shapes,
                    full_output_shapes,
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)
            # model that does not support batching
            if no_batch:
                iu.infer_zero(
                    self,
                    'onnx_nobatch',
                    1,
                    dtype,
                    input_shapes,
                    output_shapes,
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

        if tu.validate_for_libtorch_model(dtype,
                                          dtype,
                                          dtype,
                                          input_shapes[0],
                                          input_shapes[0],
                                          input_shapes[0],
                                          reshape=True):
            # skip variable size reshape on libtorch for now,
            # see "gen_qa_reshape_model.py" for detail
            if dtype != np.int32:
                # model that does not support batching
                # skip for libtorch string I/O
                if no_batch and (dtype != np_dtype_string):
                    iu.infer_zero(
                        self,
                        'libtorch_nobatch',
                        1,
                        dtype,
                        input_shapes,
                        output_shapes,
                        use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                        use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

                # model that supports batching
                for bs in (1, 8):
                    full_shapes = [[
                        bs,
                    ] + input_shape for input_shape in input_shapes]
                    full_output_shapes = [[
                        bs,
                    ] + output_shape for output_shape in output_shapes]
                    iu.infer_zero(
                        self,
                        'libtorch',
                        bs,
                        dtype,
                        full_shapes,
                        full_output_shapes,
                        use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                        use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

        for name in ["simple_reshape", "sequence_reshape", "fan_reshape"]:
            # [TODO] Skip variable size reshape on ensemble for now.
            # Need rework on how ensemble for reshape are generated
            if dtype == np.int32:
                break
            if tu.validate_for_ensemble_model(name, dtype, dtype, dtype,
                                              input_shapes[0], input_shapes[0],
                                              input_shapes[0]):
                # model that supports batching
                for bs in (1, 8):
                    full_shapes = [[
                        bs,
                    ] + input_shape for input_shape in input_shapes]
                    full_output_shapes = [[
                        bs,
                    ] + output_shape for output_shape in output_shapes]
                    iu.infer_zero(
                        self,
                        name,
                        bs,
                        dtype,
                        full_shapes,
                        full_output_shapes,
                        use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                        use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)
                # model that does not support batching
                if no_batch:
                    iu.infer_zero(
                        self,
                        name + '_nobatch',
                        1,
                        dtype,
                        input_shapes,
                        output_shapes,
                        use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                        use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

    def _trt_reshape(self,
                     dtype,
                     input_shapes,
                     output_shapes=None,
                     no_batch=True):
        # 'shapes' is list of shapes, one for each input.
        if output_shapes is None:
            output_shapes = input_shapes

        if tu.validate_for_trt_model(dtype, dtype, dtype, input_shapes[0],
                                     input_shapes[0], input_shapes[0]):
            # model that supports batching
            for bs in (1, 8):
                full_shapes = [[
                    bs,
                ] + input_shape for input_shape in input_shapes]
                full_output_shapes = [[
                    bs,
                ] + output_shape for output_shape in output_shapes]
                iu.infer_zero(
                    self,
                    'plan',
                    bs,
                    dtype,
                    full_shapes,
                    full_output_shapes,
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)
            # model that does not support batching
            if no_batch:
                iu.infer_zero(
                    self,
                    'plan_nobatch',
                    1,
                    dtype,
                    input_shapes,
                    output_shapes,
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY)

    def test_ff1(self):
        self._full_reshape(np.float32, input_shapes=([1],), no_batch=False)

    def test_ff2(self):
        self._full_reshape(np.float32, input_shapes=([1], [8]), no_batch=False)
        self._trt_reshape(np.float32, input_shapes=([1], [8]))

    def test_ff3(self):
        self._full_reshape(np.float32, input_shapes=([4, 4], [2], [2, 2, 3]))

    def test_ff4(self):
        self._full_reshape(np.float32,
                           input_shapes=([4, 4], [2], [2, 2, 3], [1]),
                           output_shapes=([16], [1, 2], [3, 2, 2], [1]))
        self._trt_reshape(np.float32,
                          input_shapes=([4, 4], [2], [2, 2, 3], [1]),
                          output_shapes=([2, 2, 4], [1, 2, 1], [3, 2,
                                                                2], [1, 1, 1]))

    def test_ii1(self):
        self._full_reshape(np.int32, input_shapes=([2, 4, 5, 6],))

    def test_ii2(self):
        self._full_reshape(np.int32,
                           input_shapes=([4, 1], [2]),
                           output_shapes=([1, 4], [1, 2]))

    def test_ii3(self):
        self._full_reshape(np.int32, input_shapes=([1, 4, 1], [8], [2, 2, 3]))

    def test_oo1(self):
        self._full_reshape(np.object_, input_shapes=([1],), no_batch=False)


if __name__ == '__main__':
    unittest.main()
