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

from functools import partial
import test_util as tu
import shm_util
import tritonclient.grpc as grpcclient
from tritonclient.utils import *
import numpy as np
import unittest
import queue
import os

TRIAL = os.getenv('TRIAL')


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class IOTest(tu.TestResultCollector):

    def setUp(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()

    def _run_test(self):
        model_name = "ensemble_io"
        user_data = UserData()
        with grpcclient.InferenceServerClient("localhost:8001") as client:
            input0 = np.random.random([1000]).astype(np.float32)
            client.start_stream(callback=partial(callback, user_data))
            for model_1_in_gpu in [True, False]:
                for model_2_in_gpu in [True, False]:
                    for model_3_in_gpu in [True, False]:
                        gpu_output = np.asarray(
                            [model_1_in_gpu, model_2_in_gpu, model_3_in_gpu],
                            dtype=bool)
                        inputs = [
                            grpcclient.InferInput(
                                "INPUT0", input0.shape,
                                np_to_triton_dtype(input0.dtype)),
                            grpcclient.InferInput(
                                "GPU_OUTPUT", gpu_output.shape,
                                np_to_triton_dtype(gpu_output.dtype))
                        ]
                        inputs[0].set_data_from_numpy(input0)
                        inputs[1].set_data_from_numpy(gpu_output)
                        client.async_stream_infer(model_name=model_name,
                                                  inputs=inputs)
                        if TRIAL == 'default':
                            result = user_data._completed_requests.get()
                            output0 = result.as_numpy('OUTPUT0')
                            self.assertIsNotNone(output0)
                            self.assertTrue(np.all(output0 == input0))
                        else:
                            response_repeat = 2
                            for _ in range(response_repeat):
                                result = user_data._completed_requests.get()
                                output0 = result.as_numpy('OUTPUT0')
                                self.assertIsNotNone(output0)
                                self.assertTrue(np.all(output0 == input0))

    def test_ensemble_io(self):
        # Only run the shared memory leak detection with the default trial
        if TRIAL == 'default':
            with self._shm_leak_detector.Probe():
                self._run_test()
        else:
            self._run_test()


if __name__ == '__main__':
    unittest.main()
