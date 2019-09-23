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
import test_util as tu

class LargePayLoadTest(unittest.TestCase):
    def setUp(self):
        self.data_type_ = np.float32
        # n GB divided by element size
        self.input_size_ = 6 * (1024 * 1024 * 1024) / np.dtype(self.data_type_).itemsize
        self.protocols_ = ((ProtocolType.HTTP, 'localhost:8000'),
                        (ProtocolType.GRPC, 'localhost:8001'))

    def _test_helper(self, ctx, tensor_shape, small_tensor_shape,
                     input_name='INPUT0', output_name='OUTPUT0'):
        try:
            in0 = np.random.random(tensor_shape).astype(self.data_type_)
            results = ctx.run({ input_name : (in0,)},
                            { output_name : InferContext.ResultFormat.RAW},
                            1)
            # if the inference is completed, examine results to ensure that
            # the framework and protocol do support large payload
            self.assertTrue(np.array_equal(in0, results[output_name][0]), "output is different from input")

        except InferenceServerException as ex:
            # if the inference failed, inference server should return error
            # gracefully. In addition to this, send a small payload to
            # verify if the server is still functional
            sin0 = np.random.random(small_tensor_shape).astype(self.data_type_)
            results = ctx.run({ input_name : (sin0,)},
                            { output_name : InferContext.ResultFormat.RAW},
                            1)
            self.assertTrue(np.array_equal(sin0, results[output_name][0]), "output is different from input")

    def test_graphdef(self):
        tensor_shape = (self.input_size_,)
        small_tensor_shape = (1,)

        # graphdef_nobatch_zero_1_float32 is identity model with input shape [-1]
        for protocol, url in self.protocols_:
            model_name = tu.get_zero_model_name("graphdef_nobatch", 1, self.data_type_)
            ctx = InferContext(url, protocol, model_name, None, True)
            self._test_helper(ctx, tensor_shape, small_tensor_shape)            

    def test_savedmodel(self):
        tensor_shape = (self.input_size_,)
        small_tensor_shape = (1,)

        # savedmodel_nobatch_zero_1_float32 is identity model with input shape [-1]
        for protocol, url in self.protocols_:
            model_name = tu.get_zero_model_name("savedmodel_nobatch", 1, self.data_type_)
            ctx = InferContext(url, protocol, model_name, None, True)
            self._test_helper(ctx, tensor_shape, small_tensor_shape)

    def test_netdef(self):
        tensor_shape = (self.input_size_,)
        small_tensor_shape = (1,)

        # netdef_nobatch_zero_1_float32 is identity model with input shape [-1]
        for protocol, url in self.protocols_:
            model_name = tu.get_zero_model_name("netdef_nobatch", 1, self.data_type_)
            ctx = InferContext(url, protocol, model_name, None, True)
            self._test_helper(ctx, tensor_shape, small_tensor_shape)

    def test_onnx(self):
        tensor_shape = (self.input_size_,)
        small_tensor_shape = (1,)

        # onnx_nobatch_zero_1_float32 is identity model with input shape [-1]
        for protocol, url in self.protocols_:
            model_name = tu.get_zero_model_name("onnx_nobatch", 1, self.data_type_)
            ctx = InferContext(url, protocol, model_name, None, True)
            self._test_helper(ctx, tensor_shape, small_tensor_shape)

    def test_plan(self):
        tensor_shape = (self.input_size_,)
        small_tensor_shape = (1,)

        # plan_nobatch_zero_1_float32 is identity model with input shape [-1]
        for protocol, url in self.protocols_:
            model_name = tu.get_zero_model_name("plan_nobatch", 1, self.data_type_)
            ctx = InferContext(url, protocol, model_name, None, True)
            self._test_helper(ctx, tensor_shape, small_tensor_shape)

    def test_libtorch(self):
        tensor_shape = (self.input_size_,)
        small_tensor_shape = (1,)

        # libtorch_nobatch_zero_1_float32 is identity model with input shape [-1]
        for protocol, url in self.protocols_:
            model_name = tu.get_zero_model_name("libtorch_nobatch", 1, self.data_type_)
            ctx = InferContext(url, protocol, model_name, None, True)
            self._test_helper(ctx, tensor_shape, small_tensor_shape,
                              'INPUT__0', 'OUTPUT__0')

    def test_custom(self):
        tensor_shape = (self.input_size_,)
        small_tensor_shape = (1,)

        # custom_zero_1_float32 is identity model with input shape [-1]
        for protocol, url in self.protocols_:
            model_name = tu.get_zero_model_name("custom", 1, self.data_type_)
            ctx = InferContext(url, protocol, model_name, None, True)
            self._test_helper(ctx, tensor_shape, small_tensor_shape)


if __name__ == '__main__':
    unittest.main()
