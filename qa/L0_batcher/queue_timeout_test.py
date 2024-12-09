#!/usr/bin/env python3

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

import concurrent.futures
import os
import time
import unittest

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

# By default, find tritonserver on "localhost", but for windows tests
# we overwrite the IP address with the TRITONSERVER_IPADDR envvar
_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class TestMaxQueueDelayTimeout(unittest.TestCase):
    def setUp(self):
        # Initialize client
        self._triton = grpcclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8001")

    def _get_inputs(self, batch_size):
        self.assertIsInstance(batch_size, int)
        self.assertGreater(batch_size, 0)
        shape = [batch_size, 8]
        inputs = [grpcclient.InferInput("INPUT0", shape, "FP32")]
        inputs[0].set_data_from_numpy(np.ones(shape, dtype=np.float32))
        return inputs

    def _generate_callback_and_response_pair(self):
        response = {"responded": False, "result": None, "error": None}

        def callback(result, error):
            response["responded"] = True
            response["result"] = result
            response["error"] = error

        return callback, response

    # Test queued requests on dynamic batch scheduler can be cancelled
    def test_default_queue_policy_timeout_prompt_response(self):
        model_name = "dynamic_batch"
        with concurrent.futures.ThreadPoolExecutor() as pool:
            # Saturate the slots on the model
            saturate_thread = pool.submit(
                self._triton.infer, model_name, self._get_inputs(batch_size=1)
            )
            time.sleep(2)  # ensure the slots are filled
            # The next request should be queued
            callback, response = self._generate_callback_and_response_pair()
            self._triton.async_infer(
                model_name, self._get_inputs(batch_size=1), callback
            )
            time.sleep(2)  # ensure the request is queued
            # Check if the request has timed-out
            time.sleep(2)  # ensure the timeout period has expired
            self.assertTrue(response["responded"])
            self.assertEqual(response["result"], None)
            self.assertIsInstance(response["error"], InferenceServerException)
            self.assertEqual(response["error"].status(), "StatusCode.UNAVAILABLE")
            self.assertEqual(response["error"].message(), "Request timeout expired")
            # Join saturating thread
            saturate_thread.result()


if __name__ == "__main__":
    unittest.main()
