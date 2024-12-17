# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# By default, find tritonserver on "localhost", but for windows tests
# we overwrite the IP address with the TRITONSERVER_IPADDR envvar
_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class ResponseSenderTest(unittest.TestCase):
    def _generate_streaming_callback_and_responses_pair(self):
        responses = []  # [{"result": result, "error": error}, ...]

        def callback(result, error):
            responses.append({"result": result, "error": error})

        return callback, responses

    def test_respond_after_complete_final(self):
        with open(os.environ["SERVER_LOG"]) as f:
            server_log = f.read()
        self.assertNotIn("Test Passed", server_log)

        model_name = "response_sender_complete_final"
        shape = [1, 1]
        inputs = [grpcclient.InferInput("INPUT0", shape, "FP32")]
        input0_np = np.array([[123.45]], np.float32)
        inputs[0].set_data_from_numpy(input0_np)

        callback, responses = self._generate_streaming_callback_and_responses_pair()
        with grpcclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8001") as client:
            client.start_stream(callback)
            client.async_stream_infer(model_name, inputs)
            client.stop_stream()

        self.assertEqual(len(responses), 1)
        for response in responses:
            output0_np = response["result"].as_numpy(name="OUTPUT0")
            self.assertTrue(np.allclose(input0_np, output0_np))
            self.assertIsNone(response["error"])

        time.sleep(1)  # make sure the logs are written before checking
        with open(os.environ["SERVER_LOG"]) as f:
            server_log = f.read()
        self.assertNotIn("Unexpected request length", server_log)
        self.assertNotIn("Expected exception not raised", server_log)
        self.assertNotIn("Test FAILED", server_log)
        self.assertIn("Test Passed", server_log)


if __name__ == "__main__":
    unittest.main()
