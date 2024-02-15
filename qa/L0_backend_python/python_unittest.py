#!/usr/bin/env python3

# Copyright 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

# By default, find tritonserver on "localhost", but for windows tests
# we overwrite the IP address with the TRITONSERVER_IPADDR envvar
_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class PythonUnittest(unittest.TestCase):
    def setUp(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()

    def _run_unittest(self, model_name):
        with grpcclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8001") as client:
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
        with self._shm_leak_detector.Probe() as shm_probe:
            self._run_unittest(model_name)


if __name__ == "__main__":
    unittest.main()
