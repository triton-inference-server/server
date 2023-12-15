#!/usr/bin/env python3

# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import infer_util as iu
import numpy as np
import test_util as tu

TENSOR_SIZE = 16384


class TfGpuIoTest(tu.TestResultCollector):
    def _test_helper(
        self,
        model_name,
        shape,
        override_input_names=[],
        override_output_names=[],
        batching_enabled=False,
    ):
        try:
            bs = 1
            if batching_enabled:
                shape = [
                    [
                        bs,
                    ]
                    + shape
                ]
            iu.infer_zero(
                self,
                "graphdef",
                bs,
                np.float32,
                shape,
                shape,
                override_model_name=model_name,
                override_input_names=override_input_names,
                override_output_names=override_output_names,
            )

        except Exception as ex:
            self.assertTrue(False, "unexpected error {}".format(ex))

    def test_sig_tag0(self):
        self._test_helper(
            "sig_tag0",
            [16],
            override_input_names=["INPUT"],
            override_output_names=["OUTPUT"],
        )

    def test_graphdef_zero_1_float32_def(self):
        self._test_helper(
            "graphdef_zero_1_float32_def", [TENSOR_SIZE], batching_enabled=True
        )

    def test_graphdef_zero_1_float32_gpu(self):
        self._test_helper(
            "graphdef_zero_1_float32_gpu", [TENSOR_SIZE], batching_enabled=True
        )

    def test_savedmodel_zero_1_float32_def(self):
        self._test_helper(
            "savedmodel_zero_1_float32_def", [TENSOR_SIZE], batching_enabled=True
        )

    def test_savedmodel_zero_1_float32_gpu(self):
        self._test_helper(
            "savedmodel_zero_1_float32_gpu", [TENSOR_SIZE], batching_enabled=True
        )


if __name__ == "__main__":
    unittest.main()
