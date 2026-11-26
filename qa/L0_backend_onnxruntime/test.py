#!/usr/bin/env python3
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import unittest

import ml_dtypes
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient


class BFloat16Test(unittest.TestCase):
    def setUp(self):
        self.protocol = os.environ.get("CLIENT_TYPE", "http")
        if self.protocol == "http":
            self.client_ = httpclient.InferenceServerClient("localhost:8000")
            self.client_module_ = httpclient
        else:
            self.client_ = grpcclient.InferenceServerClient("localhost:8001")
            self.client_module_ = grpcclient
        self.model_name_ = "onnx_bf16_bf16_bf16"
        # Model dims are [-1, 16]: dynamic batch dim plus inner dim of 16.
        self.inner_dim_ = 16

    def _infer_bf16(self, input0_data, input1_data):
        """Helper to run BF16 inference and return the output numpy arrays."""
        input0 = self.client_module_.InferInput(
            "INPUT0", list(input0_data.shape), "BF16"
        )
        input1 = self.client_module_.InferInput(
            "INPUT1", list(input1_data.shape), "BF16"
        )
        input0.set_data_from_numpy(input0_data)
        input1.set_data_from_numpy(input1_data)

        results = self.client_.infer(self.model_name_, [input0, input1])
        return results.as_numpy("OUTPUT0"), results.as_numpy("OUTPUT1")

    def test_bf16_add_sub_variants(self):
        """Run BF16 add/sub across multiple cases batched in a single request:
        zeros, negatives, large, small, cancellation, and identical."""
        cases = [
            (0.0, 0.0),
            (-1.5, 3.5),
            (100.0, 200.0),
            (1e-2, 1e-2),
            (1.0, -1.0),
            (2.0, 2.0),
        ]
        batch_size = len(cases)
        input0_data = np.empty((batch_size, self.inner_dim_), dtype=ml_dtypes.bfloat16)
        input1_data = np.empty((batch_size, self.inner_dim_), dtype=ml_dtypes.bfloat16)
        for i, (v0, v1) in enumerate(cases):
            input0_data[i, :] = v0
            input1_data[i, :] = v1

        output0, output1 = self._infer_bf16(input0_data, input1_data)
        self.assertEqual(output0.dtype, ml_dtypes.bfloat16)
        self.assertEqual(output1.dtype, ml_dtypes.bfloat16)
        np.testing.assert_array_equal(output0, input0_data + input1_data)
        np.testing.assert_array_equal(output1, input0_data - input1_data)


if __name__ == "__main__":
    unittest.main()
