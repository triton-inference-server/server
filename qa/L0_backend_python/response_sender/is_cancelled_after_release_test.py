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

import os
import time
import unittest

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

# By default, find tritonserver on "localhost", but for windows tests
# we overwrite the IP address with the TRITONSERVER_IPADDR envvar
_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class IsCancelledAfterReleaseTest(unittest.TestCase):
    def setUp(self):
        self._client = grpcclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8001")

    def tearDown(self):
        self._client.close()

    def _infer(self, model_name, value):
        inputs = [grpcclient.InferInput("INPUT0", [1, 1], "FP32")]
        input_np = np.array([[value]], np.float32)
        inputs[0].set_data_from_numpy(input_np)
        if not model_name.endswith("_decoupled"):
            try:
                result = self._client.infer(model_name, inputs)
            except InferenceServerException as error:
                return None, error
            return result.as_numpy("OUTPUT0"), None

        responses = []  # [(result, error), ...]
        self._client.start_stream(
            callback=lambda result, error: responses.append((result, error))
        )
        self._client.async_stream_infer(model_name, inputs)
        self._client.stop_stream()
        self.assertEqual(len(responses), 1)
        result, error = responses[0]
        return (None if error else result.as_numpy("OUTPUT0")), error

    def _assert_echo(self, model_name, value):
        output_np, error = self._infer(model_name, value)
        self.assertIsNone(error)
        self.assertTrue(np.allclose(output_np, [[value]]))

    def _test_released_requests(self, model_name):
        # Each request checks the previous one, which every fifth request
        # makes execute fail on.
        for i in range(1, 51):
            if i % 5 == 0:
                _, error = self._infer(model_name, -1.0)
                self.assertIsNotNone(error)
                self.assertIn("Negative input", error.message())
            else:
                self._assert_echo(model_name, float(i))
        self.assertTrue(self._client.is_server_live())

        time.sleep(1)  # make sure the logs are written before checking
        with open(os.environ["SERVER_LOG"]) as f:
            self.assertNotIn("Request reported as cancelled", f.read())

    def test_response_returned(self):
        self._test_released_requests("is_cancelled_after_release")

    def test_final_response_sent(self):
        self._test_released_requests("is_cancelled_after_release_decoupled")


if __name__ == "__main__":
    unittest.main()
