# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
import queue
import unittest
from functools import partial

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

OUTPUT_NUM_ELEMENTS = int(os.getenv("OUTPUT_NUM_ELEMENTS", 1))


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error, timeout=100)
    else:
        user_data._completed_requests.put(result, timeout=100)


class TestTritonInference(unittest.TestCase):
    def setUp(self):
        self.triton_client = grpcclient.InferenceServerClient(url="localhost:8001")

    def tearDown(self):
        self.triton_client.stop_stream()

    def test_inference(self):
        model_name = "repeat_int32"
        num_responses = 256
        in_data = np.random.randint(0, 1000, num_responses, dtype=np.int32)
        delay_data = np.zeros(num_responses, dtype=np.uint32)
        wait_data = np.zeros(1, dtype=np.uint32)
        user_data = UserData()

        inputs = [
            grpcclient.InferInput("IN", [num_responses], "INT32"),
            grpcclient.InferInput("DELAY", [num_responses], "UINT32"),
            grpcclient.InferInput("WAIT", [1], "UINT32"),
        ]
        outputs = [
            grpcclient.InferRequestedOutput("OUT"),
            grpcclient.InferRequestedOutput("IDX"),
        ]

        inputs[0].set_data_from_numpy(in_data)
        inputs[1].set_data_from_numpy(delay_data)
        inputs[2].set_data_from_numpy(wait_data)

        self.triton_client.start_stream(callback=partial(callback, user_data))
        self.triton_client.async_stream_infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
        )

        recv_count = 0
        while recv_count < num_responses:
            data_item = user_data._completed_requests.get()

            if isinstance(data_item, InferenceServerException):
                self.fail(f"InferenceServerException: {data_item}")
            try:
                response_idx = data_item.as_numpy("IDX")[0]
                response_data = data_item.as_numpy("OUT")
                expected_data = in_data[response_idx]

                self.assertEqual(
                    response_data[0],
                    expected_data,
                    f"Validation failed at index {response_idx} - response_data[0]: {response_data[0]}, expected_data: {expected_data}",
                )
                self.assertEqual(
                    response_data.size,
                    OUTPUT_NUM_ELEMENTS,
                    f"Validation failed - response_data.size: {response_data.size}, OUTPUT_NUM_ELEMENTS: {OUTPUT_NUM_ELEMENTS}",
                )

            except Exception as e:
                self.fail(f"Error processing response: {str(e)}")
            recv_count += 1

        self.assertEqual(
            user_data._completed_requests.qsize(),
            0,
            "Did not receive the expected number of responses.",
        )


if __name__ == "__main__":
    unittest.main()
