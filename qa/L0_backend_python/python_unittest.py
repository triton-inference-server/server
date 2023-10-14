#!/usr/bin/env python3

# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

sys.path.append("../../common")

import os
import unittest

import shm_util
import test_util as tu
import tritonclient.grpc as grpcclient
from tritonclient.utils import *


class PythonUnittest(tu.TestResultCollector):
    def setUp(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()

    def _run_unittest(self, model_name):
        with grpcclient.InferenceServerClient("localhost:8001") as client:
            # No input is required
            result = client.infer(model_name, [], client_timeout=240)
            output0 = result.as_numpy("OUTPUT0")

            # The model returns 1 if the tests were successfully passed.
            # Otherwise, it will return 0.
            self.assertEqual(
                output0, [1], f"python_unittest failed for model {model_name}"
            )

    def test_python_unittest(self):
        model_name = os.environ["MODEL_NAME"]
        bls_kind = os.environ.get("BLS_KIND", "non_decoupled")

        # Stress test memory growth test more on nightly runs
        nightly = int(os.environ.get("TRITON_NIGHTLY", "0"))
        repetitions = 5 if nightly else 1

        # Warmup run, let it do any upfront shm initializations required
        self._run_unittest(model_name)

        debug_str = f"{model_name} {bls_kind}"
        with self._shm_leak_detector.Probe(debug_str=debug_str) as shm_probe:
            for i in range(repetitions):
                print(f"[{debug_str}] Test iteration: {i}")
                self._run_unittest(model_name)


if __name__ == "__main__":
    unittest.main()
