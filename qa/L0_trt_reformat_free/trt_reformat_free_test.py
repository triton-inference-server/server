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
import tritonhttpclient
import tritonclient.utils.shared_memory as shm
from tritonclientutils import InferenceServerException


class TrtReformatFreeTest(tu.TestResultCollector):
    def div_up(a, b):
        return (a + b - 1) // b

    def reformat(format, tensor_np):
        if format == trt.TensorFormat.CHW2:
            shape = list(tensor_np.shape) + [2]
            shape[-4] = div_up(shape[-4], 2)
            reformatted_tensor_np = np.empty(shape, tensor_np.dtype)
            if len(tensor_np.shape) == 3:
                batch = [(tensor_np, reformatted_tensor_np)]
            elif len(tensor_np.shape) == 4:
                batch = [(tensor_np[idx], reformatted_tensor_np[idx]) for idx in tensor_np.shape[0]]
            else:
                raise ValueError("Unexpected numpy shape {} for testing reformat-free input".format(tensor_np.shape))
            for (tensor, reformatted_tensor) in batch:
                for c in range(tensor_np.shape[0]):
                    for h in range(tensor_np.shape[1]):
                        for w in range(tensor_np.shape[2]):
                            reformatted_tensor[c//2][h][w][c%2] = tensor_np[c][h][w]
        return reformatted_tensor_np

    def add_reformat_free_data_as_shared_memory(self, name, tensor, tensor_np):
        byte_size = tensor_np.size * tensor_np.dtype.itemsize
        self.shm_handles.append(shm.create_shared_memory_region(name, name, byte_size))
        # Put data values into shared memory
        shm.set_shared_memory_region(byte_size, [tensor_np])
        # Register shared memory with Triton Server
        self.triton_client.register_system_shared_memory(name, name, byte_size)
        # Set the parameters to use data from shared memory
        tensor.set_shared_memory(name, byte_size)

    def setUp(self):
        self.shm_handles = []
        self.triton_client = tritonhttpclient.InferenceServerClient("localhost:8000",
                                                             verbose=True)

    def test_nobatch_chw2_input(self):
        model_name = "plan_nobatch_CHW2_LINEAR_float16_float16_float16"
        input_np = np.arange(26, dtype=np.float16).reshape((13, 2, 1))
        expected_output0_np = input_np + input_np
        expected_output1_np = input_np - input_np
        reformatted_input_np = reformat(trt.TensorFormat.CHW2, input_np)

        # Note that the tensor metadata used for inference input is different
        # from what is used as data
        inputs = []
        inputs.append(httpclient.InferInput('INPUT0', [13, 2, 1], "FP16"))
        self.add_reformat_free_data_as_shared_memory("input0", inputs[-1], reformatted_input_np)
        inputs.append(httpclient.InferInput('INPUT1', [13, 2, 1], "FP16"))
        self.add_reformat_free_data_as_shared_memory("input1", inputs[-1], reformatted_input_np)

        outputs = []
        outputs.append(httpclient.InferRequestedOutput('OUTPUT0', binary_data=True))
        outputs.append(httpclient.InferRequestedOutput('OUTPUT1', binary_data=True))

        results = self.triton_client.infer(model_name=model_name,
                                    inputs=inputs,
                                    outputs=outputs)
        # Validate the results by comparing with precomputed values.
        output0_np = results.as_numpy('OUTPUT0')
        output1_np = results.as_numpy('OUTPUT1')
        self.assertTrue(
                    np.array_equal(output0_np, expected_output0_np),
                    "OUTPUT0 expected: {}, got {}".format(
                        expected_output0_np, output0_np))
        self.assertTrue(
                    np.array_equal(output1_np, expected_output1_np),
                    "OUTPUT0 expected: {}, got {}".format(
                        expected_output1_np, output1_np))


    def test_chw2_input(self):
        model_name = "plan_CHW2_LINEAR_float16_float16_float16"
        for bs in [1, 8]:
            input_np = np.arange(26 * bs, dtype=np.float16).reshape((bs, 13, 2, 1))
            expected_output0_np = input_np + input_np
            expected_output1_np = input_np - input_np
            reformatted_input_np = reformat(trt.TensorFormat.CHW2, input_np)

            # Note that the tensor metadata used for inference input is different
            # from what is used as data
            inputs = []
            inputs.append(httpclient.InferInput('INPUT0', [bs, 13, 2, 1], "FP16"))
            self.add_reformat_free_data_as_shared_memory("input0", inputs[-1], reformatted_input_np)
            inputs.append(httpclient.InferInput('INPUT1', [bs, 13, 2, 1], "FP16"))
            self.add_reformat_free_data_as_shared_memory("input1", inputs[-1], reformatted_input_np)

            outputs = []
            outputs.append(httpclient.InferRequestedOutput('OUTPUT0', binary_data=True))
            outputs.append(httpclient.InferRequestedOutput('OUTPUT1', binary_data=True))

            results = self.triton_client.infer(model_name=model_name,
                                        inputs=inputs,
                                        outputs=outputs)
            # Validate the results by comparing with precomputed values.
            output0_np = results.as_numpy('OUTPUT0')
            output1_np = results.as_numpy('OUTPUT1')
            self.assertTrue(
                        np.array_equal(output0_np, expected_output0_np),
                        "OUTPUT0 expected: {}, got {}".format(
                            expected_output0_np, output0_np))
            self.assertTrue(
                        np.array_equal(output1_np, expected_output1_np),
                        "OUTPUT0 expected: {}, got {}".format(
                            expected_output1_np, output1_np))


if __name__ == '__main__':
    unittest.main()
