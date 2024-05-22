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


class ResponseSenderTest(unittest.TestCase):
    def _get_inputs(
        self,
        number_of_response_before_return,
        send_complete_final_flag_before_return,
        return_a_response,
        number_of_response_after_return,
        send_complete_final_flag_after_return,
    ):
        shape = [1, 1]
        inputs = [
            grpcclient.InferInput("NUMBER_OF_RESPONSE_BEFORE_RETURN", shape, "UINT8"),
            grpcclient.InferInput(
                "SEND_COMPLETE_FINAL_FLAG_BEFORE_RETURN", shape, "BOOL"
            ),
            grpcclient.InferInput("RETURN_A_RESPONSE", shape, "BOOL"),
            grpcclient.InferInput("NUMBER_OF_RESPONSE_AFTER_RETURN", shape, "UINT8"),
            grpcclient.InferInput(
                "SEND_COMPLETE_FINAL_FLAG_AFTER_RETURN", shape, "BOOL"
            ),
        ]
        inputs[0].set_data_from_numpy(
            np.array([[number_of_response_before_return]], np.uint8)
        )
        inputs[1].set_data_from_numpy(
            np.array([[send_complete_final_flag_before_return]], bool)
        )
        inputs[2].set_data_from_numpy(np.array([[return_a_response]], bool))
        inputs[3].set_data_from_numpy(
            np.array([[number_of_response_after_return]], np.uint8)
        )
        inputs[4].set_data_from_numpy(
            np.array([[send_complete_final_flag_after_return]], bool)
        )
        return inputs

    def _generate_streaming_callback_and_responses_pair(self):
        responses = []  # [{"result": result, "error": error}, ...]

        def callback(result, error):
            responses.append({"result": result, "error": error})

        return callback, responses

    def _infer(
        self,
        model_name,
        number_of_response_before_return,
        send_complete_final_flag_before_return,
        return_a_response,
        number_of_response_after_return,
        send_complete_final_flag_after_return,
    ):
        inputs = self._get_inputs(
            number_of_response_before_return,
            send_complete_final_flag_before_return,
            return_a_response,
            number_of_response_after_return,
            send_complete_final_flag_after_return,
        )
        callback, responses = self._generate_streaming_callback_and_responses_pair()
        with grpcclient.InferenceServerClient("localhost:8001") as client:
            client.start_stream(callback)
            client.async_stream_infer(model_name, inputs)
            time.sleep(2)  # to collect all responses
            client.stop_stream()
        return responses

    # Decoupled model send response final flag before request return.
    def test_decoupled_zero_response_pre_return(self):
        responses = self._infer(
            model_name="response_sender_decoupled",
            number_of_response_before_return=0,
            send_complete_final_flag_before_return=True,
            return_a_response=False,
            number_of_response_after_return=0,
            send_complete_final_flag_after_return=False,
        )
        self.assertEqual(len(responses), 0)

    # Decoupled model send response final flag after request return.
    def test_decoupled_zero_response_post_return(self):
        responses = self._infer(
            model_name="response_sender_decoupled",
            number_of_response_before_return=0,
            send_complete_final_flag_before_return=False,
            return_a_response=False,
            number_of_response_after_return=0,
            send_complete_final_flag_after_return=True,
        )
        self.assertEqual(len(responses), 0)

    # Decoupled model send 1 response before request return.
    def test_decopled_one_response_pre_return(self):
        number_of_response = 1
        responses = self._infer(
            model_name="response_sender_decoupled",
            number_of_response_before_return=number_of_response,
            send_complete_final_flag_before_return=True,
            return_a_response=False,
            number_of_response_after_return=0,
            send_complete_final_flag_after_return=False,
        )
        self.assertEqual(len(responses), number_of_response)

    # Decoupled model send 1 response after request return.
    def test_decopled_one_response_post_return(self):
        number_of_response = 1
        responses = self._infer(
            model_name="response_sender_decoupled",
            number_of_response_before_return=0,
            send_complete_final_flag_before_return=False,
            return_a_response=False,
            number_of_response_after_return=number_of_response,
            send_complete_final_flag_after_return=True,
        )
        self.assertEqual(len(responses), number_of_response)

    # Decoupled model send 2 response before request return.
    def test_decopled_two_response_pre_return(self):
        number_of_response = 2
        responses = self._infer(
            model_name="response_sender_decoupled",
            number_of_response_before_return=number_of_response,
            send_complete_final_flag_before_return=True,
            return_a_response=False,
            number_of_response_after_return=0,
            send_complete_final_flag_after_return=False,
        )
        self.assertEqual(len(responses), number_of_response)

    # Decoupled model send 2 response after request return.
    def test_decopled_two_response_post_return(self):
        number_of_response = 2
        responses = self._infer(
            model_name="response_sender_decoupled",
            number_of_response_before_return=0,
            send_complete_final_flag_before_return=False,
            return_a_response=False,
            number_of_response_after_return=number_of_response,
            send_complete_final_flag_after_return=True,
        )
        self.assertEqual(len(responses), number_of_response)


if __name__ == "__main__":
    unittest.main()
