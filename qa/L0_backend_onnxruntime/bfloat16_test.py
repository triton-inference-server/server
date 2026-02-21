#!/usr/bin/env python
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
import sys
import unittest

import ml_dtypes
import numpy as np
import pytest
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

# Client type can be passed as first arg (e.g. python bfloat16_test.py http) or via CLIENT_TYPE env.
if len(sys.argv) >= 2 and sys.argv[1] in ("http", "grpc"):
    os.environ["CLIENT_TYPE"] = sys.argv[1]
    del sys.argv[1]


class BFloat16Test(unittest.TestCase):
    def setUp(self):
        self.protocol = os.environ.get("CLIENT_TYPE", "http")
        if self.protocol == "http":
            self.client_ = httpclient.InferenceServerClient("localhost:8000")
        else:
            self.client_ = grpcclient.InferenceServerClient("localhost:8001")
        self.model_name_ = "add_bf16"

    def _assert_allclose_bf16(self, actual, desired, **kwargs):
        """Compare bfloat16 arrays by converting to float32 for the check.

        We cannot use np.testing.assert_allclose(actual, desired) directly:
        isclose() does result_type(y, 1.) and raises DTypePromotionError for
        bfloat16 in NumPy 2. The error message is misleading—it says
        "Float16DType and bfloat16" even when both arrays are bfloat16; the
        real conflict is bfloat16 vs the scalar 1.0 (float64) used inside
        isclose. Converting to float32 only for the comparison avoids this.
        """
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float32),
            np.asarray(desired, dtype=np.float32),
            **kwargs,
        )

    def _infer_bf16(self, input0_data, input1_data):
        """Helper to run BF16 inference and return the output numpy array."""
        if self.protocol == "http":
            input0 = httpclient.InferInput("INPUT0", [5, 5], "BF16")
            input1 = httpclient.InferInput("INPUT1", [5, 5], "BF16")
        else:
            input0 = grpcclient.InferInput("INPUT0", [5, 5], "BF16")
            input1 = grpcclient.InferInput("INPUT1", [5, 5], "BF16")
        input0.set_data_from_numpy(input0_data)
        input1.set_data_from_numpy(input1_data)

        results = self.client_.infer(self.model_name_, [input0, input1])
        return results.as_numpy("OUTPUT")

    @pytest.mark.parametrize(
        "input0_val,input1_val,expected_val",
        [
            (0.0, 0.0, 0.0),  # zeros
            (-1.5, 3.5, 2.0),  # negatives / mixed
            (100.0, 200.0, 300.0),  # large
            (1e-2, 1e-2, 2e-2),  # small (near underflow)
            (1.0, -1.0, 0.0),  # cancellation
            (2.0, 2.0, 4.0),  # identical inputs
        ],
    )
    def test_bf16_add_variants(self, input0_val, input1_val, expected_val):
        """Run BF16 add for one case: zeros, negatives, large, small, cancellation, or identical."""
        shape = (5, 5)
        output = self._infer_bf16(
            np.full(shape, input0_val, dtype=ml_dtypes.bfloat16),
            np.full(shape, input1_val, dtype=ml_dtypes.bfloat16),
        )
        self.assertEqual(output.dtype, ml_dtypes.bfloat16)
        self._assert_allclose_bf16(
            output, np.full(shape, expected_val, dtype=ml_dtypes.bfloat16)
        )


if __name__ == "__main__":
    unittest.main()
