# Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import test_util as tu
import shm_util
import unittest
import tritonclient.grpc as grpcclient
from tritonclient.utils import *
import os


class PythonUnittest(tu.TestResultCollector):

    def setUp(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()

    def _run_unittest(self, model_name):
        with grpcclient.InferenceServerClient("localhost:8001") as client:
            # No input is required
            result = client.infer(model_name, [], client_timeout=120)
            output0 = result.as_numpy('OUTPUT0')

            # The model returns 1 if the tests were sucessfully passed.
            # Otherwise, it will return 0.
            self.assertEqual(output0, [1])

    def test_python_unittest(self):
        model_name = os.environ['MODEL_NAME']

        if model_name == 'bls' or model_name == 'bls_memory' or model_name == 'bls_memory_async':
            # For these tests, the memory region size will be grown. Because of
            # this we need to use the shared memory probe only on the later
            # call so that the probe can detect the leak correctly.
            self._run_unittest(model_name)

            # [FIXME] See DLIS-3684
            self._run_unittest(model_name)
            with self._shm_leak_detector.Probe() as shm_probe:
                self._run_unittest(model_name)
        else:
            with self._shm_leak_detector.Probe() as shm_probe:
                self._run_unittest(model_name)


if __name__ == '__main__':
    unittest.main()
