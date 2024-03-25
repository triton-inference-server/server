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


class ConcurrencyTest(unittest.TestCase):
    def setUp(self):
        # Initialize client
        self._triton = grpcclient.InferenceServerClient("localhost:8001")

    def _generate_streaming_callback_and_response_pair(self):
        response = []  # [{"result": result, "error": error}, ...]

        def callback(result, error):
            response.append({"result": result, "error": error})

        return callback, response

    # Helper for testing concurrent execution
    def _concurrent_execute_requests(self, batch_size, number_of_requests):
        model_name = "async_execute_decouple"
        delay_secs = 4
        shape = [batch_size, 1]
        inputs = [grpcclient.InferInput("WAIT_SECONDS", shape, "FP32")]
        inputs[0].set_data_from_numpy(np.full(shape, delay_secs, dtype=np.float32))

        callback, response = self._generate_streaming_callback_and_response_pair()
        self._triton.start_stream(callback)
        for i in range(number_of_requests):
            self._triton.async_stream_infer(model_name, inputs)

        # 2s for sending requests for processing and 2s for returning results.
        wait_secs = 2 + delay_secs + 2
        time.sleep(wait_secs)
        # Ensure the sleep is shorter than sequential processing delay.
        sequential_min_delay = wait_secs * batch_size * number_of_requests
        self.assertLessEqual(wait_secs, sequential_min_delay)

        # If executed sequentially, the results are not available yet, so concurrent
        # execution is observed from seeing the correct responses.
        self.assertEqual(len(response), number_of_requests)
        for res in response:
            self.assertEqual(res["result"].as_numpy("DUMMY_OUT").shape[0], batch_size)
            self.assertIsNone(res["error"])

        self._triton.stop_stream()

    # Test batched requests are executed concurrently
    def test_concurrent_execute_single_request(self):
        self._concurrent_execute_requests(batch_size=4, number_of_requests=1)

    # Test multiple requests are executed concurrently
    def test_concurrent_execute_multi_request(self):
        self._concurrent_execute_requests(batch_size=1, number_of_requests=4)

    # Test model exception handling
    def test_model_raise_exception(self):
        model_name = "async_execute_decouple"
        delay_secs = -1  # model will raise exception
        shape = [1, 1]
        inputs = [grpcclient.InferInput("WAIT_SECONDS", shape, "FP32")]
        inputs[0].set_data_from_numpy(np.full(shape, delay_secs, dtype=np.float32))

        with open(os.environ["SERVER_LOG"]) as f:
            server_log = f.read()
        self.assertNotIn("ValueError: wait_secs cannot be negative", server_log)

        callback, response = self._generate_streaming_callback_and_response_pair()
        self._triton.start_stream(callback)
        self._triton.async_stream_infer(model_name, inputs)
        time.sleep(2)
        self._triton.stop_stream()

        with open(os.environ["SERVER_LOG"]) as f:
            server_log = f.read()
        self.assertIn("ValueError: wait_secs cannot be negative", server_log)


if __name__ == "__main__":
    unittest.main()
