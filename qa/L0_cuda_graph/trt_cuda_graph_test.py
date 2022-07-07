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


class TrtCudaGraphTest(tu.TestResultCollector):

    def setUp(self):
        self.dtype_ = np.float32
        self.dtype_str_ = "FP32"
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

    def _erroneous_infer(self, tensor_shape, batch_size):
        import tritonhttpclient
        item_size = batch_size
        for dim in tensor_shape:
            item_size *= dim
        full_shape = (batch_size,) + tensor_shape
        input_np = np.arange(item_size, dtype=self.dtype_).reshape(full_shape)
        expected_output0_np = input_np + input_np
        expected_output1_np = input_np - input_np

        inputs = []
        inputs.append(
            tritonhttpclient.InferInput('INPUT0', full_shape, self.dtype_str_))
        inputs[-1].set_data_from_numpy(input_np)
        inputs.append(
            tritonhttpclient.InferInput('INPUT1', full_shape, self.dtype_str_))
        inputs[-1].set_data_from_numpy(input_np)
        outputs = []
        outputs.append(
            tritonhttpclient.InferRequestedOutput('OUTPUT0', binary_data=True))
        outputs.append(
            tritonhttpclient.InferRequestedOutput('OUTPUT1', binary_data=True))

        model_name = tu.get_model_name(self.model_name_, self.dtype_,
                                       self.dtype_, self.dtype_)
        results = tritonhttpclient.InferenceServerClient(
            "localhost:8000", verbose=True).infer(model_name=model_name,
                                                  inputs=inputs,
                                                  outputs=outputs)
        # Validate the results by comparing with precomputed values.
        output0_np = results.as_numpy('OUTPUT0')
        output1_np = results.as_numpy('OUTPUT1')
        self.assertFalse(np.array_equal(output0_np, expected_output0_np),
                         "expects OUTPUT0 is not correct")
        self.assertFalse(np.array_equal(output1_np, expected_output1_np),
                         "expects OUTPUT1 is not correct")

    def test_fixed_shape(self):
        tensor_shape = (16,)
        self._check_infer(tensor_shape)
        # Inference that should not have CUDA graph captured
        self._check_infer(tensor_shape, 5)

    def test_dynamic_shape(self):
        tensor_shape = (16,)
        self._check_infer(tensor_shape)
        # Inference that should not have CUDA graph captured
        self._check_infer((20,))
        self._check_infer(tensor_shape, 5)

    def test_range_fixed_shape(self):
        tensor_shape = (16,)
        # Inferences that are in range of captured CUDA graph,
        # model should tolerate difference in batch size
        self._check_infer(tensor_shape, 4)
        self._check_infer(tensor_shape, 2)
        # Inferences that shouldn't use CUDA graph
        self._check_infer(tensor_shape, 1)
        self._check_infer(tensor_shape, 8)

    def test_range_dynamic_shape(self):
        # Inferences that are in range of captured CUDA graph,
        # model should tolerate difference in batch size
        self._check_infer((16,), 4)
        self._check_infer((16,), 2)
        # Inference should return dummy result
        # because the input shape is different
        self._erroneous_infer((10,), 3)

        # Inferences that shouldn't use CUDA graph
        self._check_infer((7,), 3)
        self._check_infer((16,), 1)
        self._check_infer((16,), 8)
        self._check_infer((30,), 4)


if __name__ == '__main__':
    unittest.main()
