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

import itertools
import queue
import unittest
from functools import partial

import numpy as np
import shm_util
import tritonclient.grpc as grpcclient
from tritonclient.utils import *

TRIAL = os.getenv("TRIAL")

# By default, find tritonserver on "localhost", but for windows tests
# we overwrite the IP address with the TRITONSERVER_IPADDR envvar
_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class IOTest(unittest.TestCase):
    def setUp(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()
        self._client = grpcclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8001")

    def _run_ensemble_test(self, model_name):
        user_data = UserData()
        input0 = np.random.random([1000]).astype(np.float32)
        # Use context manager to close client stream if any early exit occurs
        with grpcclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8001") as client:
            client.start_stream(callback=partial(callback, user_data))
            # Each pair represents whether the corresponding model is in GPU or not.
            gpu_flags = [(True, False), (True, False), (True, False)]
            # Create iterable of all possible combinations of each model gpu location
            # ex: (True, True, True), (True, True, False), (True, False, True), ...
            combinations = itertools.product(*gpu_flags)
            for model_1_in_gpu, model_2_in_gpu, model_3_in_gpu in combinations:
                gpu_output = np.asarray(
                    [model_1_in_gpu, model_2_in_gpu, model_3_in_gpu], dtype=bool
                )
                inputs = [
                    grpcclient.InferInput(
                        "INPUT0", input0.shape, np_to_triton_dtype(input0.dtype)
                    ),
                    grpcclient.InferInput(
                        "GPU_OUTPUT",
                        gpu_output.shape,
                        np_to_triton_dtype(gpu_output.dtype),
                    ),
                ]
                inputs[0].set_data_from_numpy(input0)
                inputs[1].set_data_from_numpy(gpu_output)
                client.async_stream_infer(model_name=model_name, inputs=inputs)
                if TRIAL == "default":
                    result = user_data._completed_requests.get()
                    output0 = result.as_numpy("OUTPUT0")
                    self.assertIsNotNone(output0)
                    self.assertTrue(np.all(output0 == input0))
                else:
                    response_repeat = 2
                    for _ in range(response_repeat):
                        result = user_data._completed_requests.get()
                        output0 = result.as_numpy("OUTPUT0")
                        self.assertIsNotNone(output0)
                        self.assertTrue(np.all(output0 == input0))

    def test_ensemble_io(self):
        model_name = "ensemble_io"

        # FIXME: This test detects a decrease of 80 bytes, which fails inequality check:
        # [ensemble_io] Shared memory leak detected: 1006976 (current) != 1007056 (prev).
        # so Probe was modified to check for growth instead of inequality.
        with self._shm_leak_detector.Probe():
            self._run_ensemble_test(model_name)

    def test_empty_gpu_output(self):
        model_name = "dlpack_empty_output"
        with self._shm_leak_detector.Probe():
            input_data = np.array([[1.0]], dtype=np.float32)
            inputs = [
                grpcclient.InferInput(
                    "INPUT", input_data.shape, np_to_triton_dtype(input_data.dtype)
                )
            ]
            inputs[0].set_data_from_numpy(input_data)
            result = self._client.infer(model_name, inputs)
            output = result.as_numpy("OUTPUT")
            self.assertIsNotNone(output)
            self.assertEqual(output.size, 0)

    def test_variable_gpu_output(self):
        model_name = "variable_gpu_output"
        with self._shm_leak_detector.Probe():
            # Input is not important in this test
            input_data = np.array([[1.0]], dtype=np.float32)
            inputs = [
                grpcclient.InferInput(
                    "INPUT", input_data.shape, np_to_triton_dtype(input_data.dtype)
                )
            ]
            inputs[0].set_data_from_numpy(input_data)
            user_data = UserData()

            # The test sends five requests to the model and the model returns five
            # responses with different GPU output shapes
            num_requests = 5
            for _ in range(num_requests):
                _ = self._client.async_infer(
                    model_name=model_name,
                    inputs=inputs,
                    callback=partial(callback, user_data),
                )

            for i in range(num_requests):
                result = user_data._completed_requests.get()
                if result is InferenceServerException:
                    self.assertTrue(False, result)
                output = result.as_numpy("OUTPUT")
                self.assertIsNotNone(output)
                self.assertEqual(output.size, i + 1)
                np.testing.assert_almost_equal(output, np.ones(i + 1) * (i + 1))


if __name__ == "__main__":
    unittest.main()
