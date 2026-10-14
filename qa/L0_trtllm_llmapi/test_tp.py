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

"""Tensor-parallel serving; LLMAPI_TP selects the tensor_parallel_size.

Needs LLMAPI_TP GPUs. With fewer, the test is reported as skipped, or fails when
REQUIRE_GPUS=1, so a single-GPU run cannot pass for a multi-GPU matrix.
"""

import os
import sys
import unittest

sys.path.append("../common")

import llmapi_test_util as util  # noqa: E402
from test_util import TestResultCollector  # noqa: E402

PROMPT = "The capital of France is Paris. The capital of Germany is"


class LlmapiTensorParallelTest(TestResultCollector):
    def test_generate(self):
        tp = int(os.environ.get("LLMAPI_TP", "2"))
        util.require_gpus(self, tp)
        result = util.grpc_infer(
            PROMPT, sampling_param_max_tokens=16, sampling_param_temperature=0.0
        )
        self.assertIn("Berlin", util.output_text(result))


if __name__ == "__main__":
    unittest.main()
