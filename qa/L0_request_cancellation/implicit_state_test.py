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

import os
import time
import unittest

import numpy as np
import tritonclient.grpc as grpcclient


class TestImplicitState(unittest.TestCase):
    def _get_inputs(self, delay_itrs):
        shape = [1, 1]
        inputs = [grpcclient.InferInput("DELAY_ITRS__0", shape, "INT64")]
        inputs[0].set_data_from_numpy(np.array([[delay_itrs]], np.int64))
        return inputs

    def _generate_streaming_callback_and_response_pair(self):
        response = []  # [{"result": result, "error": error}, ...]

        def callback(result, error):
            response.append({"result": result, "error": error})

        return callback, response

    def _sequence_state_model_infer(self, num_reqs, seq_ids, delay_itrs, cancel_reqs):
        model_name = "sequence_state"
        callback, response = self._generate_streaming_callback_and_response_pair()
        with grpcclient.InferenceServerClient("localhost:8001") as client:
            client.start_stream(callback)
            seq_start = True
            for req_id in range(num_reqs):
                for seq_id in seq_ids:
                    client.async_stream_infer(
                        model_name,
                        self._get_inputs(delay_itrs),
                        sequence_id=seq_id,
                        sequence_start=seq_start,
                    )
                    time.sleep(0.1)
                seq_start = False
            client.stop_stream(cancel_requests=cancel_reqs)
        return response

    # Test timeout is reset for a sequence slot after its sequence is cancelled
    def test_state_reset_after_cancel(self):
        sequence_timeout = 6  # secs
        # Start sequence 1 and cancel it
        num_reqs = 10
        response = self._sequence_state_model_infer(
            num_reqs, seq_ids=[1], delay_itrs=5000000, cancel_reqs=True
        )
        self.assertLess(
            len(response),
            num_reqs,
            "Precondition not met - sequence completed before cancellation",
        )
        # Wait for sequence 1 to timeout
        time.sleep(sequence_timeout + 2)
        # Start sequence 2 and 3
        self._sequence_state_model_infer(
            num_reqs=4, seq_ids=[2, 3], delay_itrs=0, cancel_reqs=False
        )
        # Check for any unexpected sequence state mixing
        with open(os.environ["SERVER_LOG"]) as f:
            server_log = f.read()
        self.assertNotIn("[MODEL ERROR] Invalid sequence state", server_log)


if __name__ == "__main__":
    unittest.main()
