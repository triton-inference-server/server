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
import tritonclient.http as httpclient
from tritonclient.utils import *
import numpy as np
import unittest
import shm_util


class ExplicitModelTest(tu.TestResultCollector):

    def setUp(self):
        self._shm_leak_detector = shm_util.ShmLeakDetector()

    def send_identity_request(self, client, model_name):
        inputs = []
        inputs.append(httpclient.InferInput('INPUT0', [1, 16], "FP32"))
        input0_data = np.arange(start=0, stop=16, dtype=np.float32)
        input0_data = np.expand_dims(input0_data, axis=0)
        inputs[0].set_data_from_numpy(input0_data)

        with self._shm_leak_detector.Probe() as shm_probe:
            result = client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=[httpclient.InferRequestedOutput('OUTPUT0')])
        output_numpy = result.as_numpy('OUTPUT0')
        self.assertTrue(np.all(input0_data == output_numpy))

    def test_model_reload(self):
        model_name = "identity_fp32"
        ensemble_model_name = 'simple_' + "identity_fp32"
        with httpclient.InferenceServerClient("localhost:8000") as client:
            for _ in range(5):
                self.assertFalse(client.is_model_ready(model_name))
                # Load the model before the ensemble model to make sure reloading the
                # model works properly in Python backend.
                client.load_model(model_name)
                client.load_model(ensemble_model_name)
                self.assertTrue(client.is_model_ready(model_name))
                self.assertTrue(client.is_model_ready(ensemble_model_name))
                self.send_identity_request(client, model_name)
                self.send_identity_request(client, ensemble_model_name)
                client.unload_model(ensemble_model_name)
                client.unload_model(model_name)
                self.assertFalse(client.is_model_ready(model_name))
                self.assertFalse(client.is_model_ready(ensemble_model_name))


if __name__ == '__main__':
    unittest.main()
