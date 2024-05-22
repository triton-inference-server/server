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


import threading
import time

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def _get_instructions_from_request(self, request):
        instr = {}
        return_a_response_np = pb_utils.get_input_tensor_by_name(
            request, "RETURN_A_RESPONSE"
        ).as_numpy()
        instr["batch_size"] = return_a_response_np.shape[0]
        instr["return_a_response"] = bool(return_a_response_np.sum())
        instr["number_of_pre_return_response"] = (
            pb_utils.get_input_tensor_by_name(
                request, "NUMBER_OF_RESPONSE_BEFORE_RETURN"
            )
            .as_numpy()
            .sum()
        )
        instr["number_of_post_return_response"] = (
            pb_utils.get_input_tensor_by_name(
                request, "NUMBER_OF_RESPONSE_AFTER_RETURN"
            )
            .as_numpy()
            .sum()
        )
        instr["send_complete_final_flag_pre_return"] = bool(
            pb_utils.get_input_tensor_by_name(
                request, "SEND_COMPLETE_FINAL_FLAG_BEFORE_RETURN"
            )
            .as_numpy()
            .sum()
        )
        instr["send_complete_final_flag_post_return"] = bool(
            pb_utils.get_input_tensor_by_name(
                request, "SEND_COMPLETE_FINAL_FLAG_AFTER_RETURN"
            )
            .as_numpy()
            .sum()
        )
        return instr

    def _is_response_sender_needed(self, instr):
        return (
            instr["number_of_pre_return_response"] > 0
            or instr["number_of_post_return_response"] > 0
            or instr["send_complete_final_flag_pre_return"]
            or instr["send_complete_final_flag_post_return"]
        )

    def _create_response(self, batch_size, response_id):
        output_tensor = pb_utils.Tensor(
            "DUMMY_OUT", np.array([[response_id] for _ in range(batch_size)], np.uint8)
        )
        response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
        return response

    def _send_responses(self, processed_requests, response_id_offset):
        for request in processed_requests:
            number_of_response = request["number_of_response"]
            batch_size = request["batch_size"]
            response_sender = request["response_sender"]
            send_complete_final_flag = request["send_complete_final_flag"]
            for response_id in range(number_of_response):
                response_sender.send(
                    self._create_response(
                        batch_size, response_id=(response_id_offset + response_id)
                    )
                )
            if send_complete_final_flag:
                response_sender.send(
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )

    def _send_responses_after_return(self, processed_requests, response_id_offset):
        def response_thread(send_responses, processed_requests, response_id_offset):
            time.sleep(1)  # response after requests are released
            send_responses(processed_requests, response_id_offset)

        thread = threading.Thread(
            target=response_thread,
            args=(self._send_responses, processed_requests, response_id_offset),
        )
        thread.daemon = True
        thread.start()

    def execute(self, requests):
        pre_return_processed_requests = []
        return_responses = []
        post_return_processed_requests = []

        for request in requests:
            instr = self._get_instructions_from_request(request)

            response_sender = None
            if self._is_response_sender_needed(instr):
                response_sender = request.get_response_sender()

            pre_return_processed_requests.append(
                {
                    "number_of_response": instr["number_of_pre_return_response"],
                    "batch_size": instr["batch_size"],
                    "response_sender": response_sender,
                    "send_complete_final_flag": instr[
                        "send_complete_final_flag_pre_return"
                    ],
                }
            )
            post_return_processed_requests.append(
                {
                    "number_of_response": instr["number_of_post_return_response"],
                    "batch_size": instr["batch_size"],
                    "response_sender": response_sender,
                    "send_complete_final_flag": instr[
                        "send_complete_final_flag_post_return"
                    ],
                }
            )

            response = None
            if instr["return_a_response"]:
                response = self._create_response(instr["batch_size"], response_id=0)
            return_responses.append(response)

        self._send_responses(pre_return_processed_requests, response_id_offset=1000)
        self._send_responses_after_return(
            post_return_processed_requests, response_id_offset=2000
        )

        if return_responses == [None for _ in requests]:
            return None
        return return_responses
