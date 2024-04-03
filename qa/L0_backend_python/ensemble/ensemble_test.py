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


class EnsembleTest(unittest.TestCase):
    def setUp(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()

    def infer(self, model_name):
        shape = [16]
        with self._shm_leak_detector.Probe() as shm_probe:
            with httpclient.InferenceServerClient(
                f"{_tritonserver_ipaddr}:8000"
            ) as client:
                input_data_0 = np.random.random(shape).astype(np.float32)
                input_data_1 = np.random.random(shape).astype(np.float32)
                inputs = [
                    httpclient.InferInput(
                        "INPUT0",
                        input_data_0.shape,
                        np_to_triton_dtype(input_data_0.dtype),
                    ),
                    httpclient.InferInput(
                        "INPUT1",
                        input_data_1.shape,
                        np_to_triton_dtype(input_data_1.dtype),
                    ),
                ]
                inputs[0].set_data_from_numpy(input_data_0)
                inputs[1].set_data_from_numpy(input_data_1)
                result = client.infer(model_name, inputs)
                output0 = result.as_numpy("OUTPUT0")
                output1 = result.as_numpy("OUTPUT1")
                self.assertIsNotNone(output0)
                self.assertIsNotNone(output1)

                # Set a big enough tolerance to reduce intermittence. May be
                # better to test integer outputs in the future for consistency.
                self.assertTrue(np.allclose(output0, 2 * input_data_0, atol=1e-06))
                self.assertTrue(np.allclose(output1, 2 * input_data_1, atol=1e-06))

    def test_ensemble(self):
        model_name = "ensemble"
        self.infer(model_name)

    def test_ensemble_gpu(self):
        model_name = "ensemble_gpu"
        self.infer(model_name)


if __name__ == "__main__":
    unittest.main()
