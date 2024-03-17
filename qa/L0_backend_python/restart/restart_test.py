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

import os
import sys

sys.path.append("../../common")

import unittest

import numpy as np
import shm_util
import tritonclient.http as httpclient
from tritonclient.utils import *

# By default, find tritonserver on "localhost", but for windows tests
# we overwrite the IP address with the TRITONSERVER_IPADDR envvar
_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class RestartTest(unittest.TestCase):
    def setUp(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()

    def _infer_helper(self, model_name, shape, data_type):
        with httpclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8000") as client:
            input_data_0 = np.array(np.random.randn(*shape), dtype=data_type)
            inputs = [
                httpclient.InferInput(
                    "INPUT0", shape, np_to_triton_dtype(input_data_0.dtype)
                )
            ]
            inputs[0].set_data_from_numpy(input_data_0)
            result = client.infer(model_name, inputs)
            output0 = result.as_numpy("OUTPUT0")
            self.assertTrue(np.all(input_data_0 == output0))

    def test_restart(self):
        shape = [1, 16]
        model_name = "restart"
        dtype = np.float32

        # Since the stub process has been killed, the first request
        # will return an exception.
        with self.assertRaises(InferenceServerException):
            # FIXME: No leak check here as the unhealthy stub error likely causes issues.
            # tritonclient.utils.InferenceServerException: [400] Failed to
            # process the request(s) for model instance 'restart_0_0',
            # message: Stub process 'restart_0_0' is not healthy.
            # [restart] Shared memory leak detected: 1007216 (current) > 1007056 (prev).
            self._infer_helper(model_name, shape, dtype)

        # The second request should work properly since the stub process should
        # have come alive.
        with self._shm_leak_detector.Probe() as shm_probe:
            self._infer_helper(model_name, shape, dtype)

    def test_infer(self):
        shape = [1, 16]
        model_name = "restart"
        dtype = np.float32
        with self._shm_leak_detector.Probe() as shm_probe:
            self._infer_helper(model_name, shape, dtype)


if __name__ == "__main__":
    unittest.main()
