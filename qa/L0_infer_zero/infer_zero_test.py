#!/usr/bin/env python3

# Copyright 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import unittest

import infer_util as iu
import numpy as np
import test_util as tu

np_dtype_string = np.dtype(object)

TEST_SYSTEM_SHARED_MEMORY = bool(int(os.environ.get("TEST_SYSTEM_SHARED_MEMORY", 0)))
TEST_CUDA_SHARED_MEMORY = bool(int(os.environ.get("TEST_CUDA_SHARED_MEMORY", 0)))
BACKENDS = os.environ.get("BACKENDS", "graphdef savedmodel onnx libtorch")
VALIDATION_FNS = {
    "onnx": tu.validate_for_onnx_model,
    "graphdef": tu.validate_for_tf_model,
    "savedmodel": tu.validate_for_tf_model,
    "libtorch": tu.validate_for_libtorch_model,
}


class InferZeroTest(tu.TestResultCollector):
    def _full_zero(self, dtype, shapes):
        # 'shapes' is list of shapes, one for each input.
        for backend in BACKENDS.split(" "):
            # object models do not exist right now for PyTorch
            if backend == "libtorch" and dtype == "object":
                return

            if not VALIDATION_FNS[backend](
                dtype, dtype, dtype, shapes[0], shapes[0], shapes[0]
            ):
                return

            for bs in (1, 8):
                batch_shapes = [
                    [
                        bs,
                    ]
                    + shape
                    for shape in shapes
                ]
                iu.infer_zero(
                    self,
                    backend,
                    bs,
                    dtype,
                    batch_shapes,
                    batch_shapes,
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY,
                )

            # model that does not support batching
            iu.infer_zero(
                self,
                f"{backend}_nobatch",
                1,
                dtype,
                shapes,
                shapes,
                use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY,
            )

        for name in ["simple_zero", "sequence_zero", "fan_zero"]:
            if tu.validate_for_ensemble_model(
                name, dtype, dtype, dtype, shapes[0], shapes[0], shapes[0]
            ):
                # model that supports batching
                for bs in (1, 8):
                    batch_shapes = [
                        [
                            bs,
                        ]
                        + shape
                        for shape in shapes
                    ]
                    iu.infer_zero(
                        self,
                        name,
                        bs,
                        dtype,
                        batch_shapes,
                        batch_shapes,
                        use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                        use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY,
                    )
                # model that does not support batching
                iu.infer_zero(
                    self,
                    name + "_nobatch",
                    1,
                    dtype,
                    shapes,
                    shapes,
                    use_system_shared_memory=TEST_SYSTEM_SHARED_MEMORY,
                    use_cuda_shared_memory=TEST_CUDA_SHARED_MEMORY,
                )

    def test_ff1_sanity(self):
        self._full_zero(
            np.float32,
            (
                [
                    1,
                ],
            ),
        )

    def test_ff1(self):
        self._full_zero(
            np.float32,
            (
                [
                    0,
                ],
            ),
        )

    def test_ff3_sanity(self):
        self._full_zero(
            np.float32,
            (
                [
                    1,
                ],
                [
                    2,
                ],
                [
                    1,
                ],
            ),
        )

    def test_ff3_0(self):
        self._full_zero(
            np.float32,
            (
                [
                    0,
                ],
                [
                    0,
                ],
                [
                    0,
                ],
            ),
        )

    def test_ff3_1(self):
        self._full_zero(
            np.float32,
            (
                [
                    0,
                ],
                [
                    0,
                ],
                [
                    1,
                ],
            ),
        )

    def test_ff3_2(self):
        self._full_zero(
            np.float32,
            (
                [
                    0,
                ],
                [
                    1,
                ],
                [
                    0,
                ],
            ),
        )

    def test_ff3_3(self):
        self._full_zero(
            np.float32,
            (
                [
                    1,
                ],
                [
                    0,
                ],
                [
                    0,
                ],
            ),
        )

    def test_ff3_4(self):
        self._full_zero(
            np.float32,
            (
                [
                    1,
                ],
                [
                    0,
                ],
                [
                    1,
                ],
            ),
        )

    def test_hh1_sanity(self):
        self._full_zero(np.float16, ([2, 2],))

    def test_hh1_0(self):
        self._full_zero(np.float16, ([1, 0],))

    def test_hh1_1(self):
        self._full_zero(np.float16, ([0, 1],))

    def test_hh1_2(self):
        self._full_zero(np.float16, ([0, 0],))

    def test_hh3_sanity(self):
        self._full_zero(np.float16, ([2, 2], [2, 2], [1, 1]))

    def test_hh3_0(self):
        self._full_zero(np.float16, ([0, 0], [0, 0], [0, 0]))

    def test_hh3_1(self):
        self._full_zero(np.float16, ([0, 1], [0, 1], [2, 3]))

    def test_hh3_2(self):
        self._full_zero(np.float16, ([1, 0], [1, 3], [0, 1]))

    def test_hh3_3(self):
        self._full_zero(np.float16, ([1, 1], [3, 0], [0, 0]))

    def test_hh3_4(self):
        self._full_zero(np.float16, ([1, 1], [0, 6], [2, 2]))

    def test_oo1_sanity(self):
        self._full_zero(
            np_dtype_string,
            (
                [
                    2,
                ],
            ),
        )

    def test_oo1(self):
        self._full_zero(
            np_dtype_string,
            (
                [
                    0,
                ],
            ),
        )

    def test_oo3_sanity(self):
        self._full_zero(np_dtype_string, ([2, 2], [2, 2], [1, 1]))

    def test_oo3_0(self):
        self._full_zero(np_dtype_string, ([0, 0], [0, 0], [0, 0]))

    def test_oo3_1(self):
        self._full_zero(np_dtype_string, ([0, 1], [0, 1], [2, 3]))

    def test_oo3_2(self):
        self._full_zero(np_dtype_string, ([1, 0], [1, 3], [0, 1]))

    def test_oo3_3(self):
        self._full_zero(np_dtype_string, ([1, 1], [3, 0], [0, 0]))

    def test_oo3_4(self):
        self._full_zero(np_dtype_string, ([1, 1], [0, 6], [2, 2]))

    def test_bb1_sanity(self):
        self._full_zero(
            bool,
            (
                [
                    10,
                ],
            ),
        )

    def test_bb1_0(self):
        self._full_zero(
            bool,
            (
                [
                    0,
                ],
            ),
        )


if __name__ == "__main__":
    unittest.main()
