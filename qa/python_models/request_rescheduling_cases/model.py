# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import threading

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config
        )
        if not using_decoupled:
            raise pb_utils.TritonModelException(
                """the model `{}` can generate any number of responses per request,
                enable decoupled transaction policy in model configuration to
                serve this model""".format(
                    args["model_name"]
                )
            )

        # Get IN configuration
        in_config = pb_utils.get_input_config_by_name(model_config, "IN")

        # Validate the shape and data type of IN
        in_shape = in_config["dims"]
        if (len(in_shape) != 1) or (in_shape[0] != 1):
            raise pb_utils.TritonModelException(
                """the model `{}` requires the shape of 'IN' to be
                [1], got {}""".format(
                    args["model_name"], in_shape
                )
            )
        if in_config["data_type"] != "TYPE_INT32":
            raise pb_utils.TritonModelException(
                """the model `{}` requires the data_type of 'IN' to be
                'TYPE_INT32', got {}""".format(
                    args["model_name"], in_config["data_type"]
                )
            )

        # Get OUT configuration
        out_config = pb_utils.get_output_config_by_name(model_config, "OUT")

        # Validate the shape and data type of OUT
        out_shape = out_config["dims"]
        if (len(out_shape) != 1) or (out_shape[0] != 1):
            raise pb_utils.TritonModelException(
                """the model `{}` requires the shape of 'OUT' to be
                [1], got {}""".format(
                    args["model_name"], out_shape
                )
            )
        if out_config["data_type"] != "TYPE_INT32":
            raise pb_utils.TritonModelException(
                """the model `{}` requires the data_type of 'OUT' to be
                'TYPE_INT32', got {}""".format(
                    args["model_name"], out_config["data_type"]
                )
            )

        self.idx = 0
        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()

    def execute(self, requests):
        for request in requests:
            case = pb_utils.get_input_tensor_by_name(request, "IN").as_numpy()

            if case[0] == 0:
                self.send_final_flag_before_rescheduling_request(request)
            elif case[0] == 1:
                self.process_request_thread(request)
            else:
                raise pb_utils.TritonModelException("Unknown test case.")

        return None

    def send_final_flag_before_rescheduling_request(self, request):
        response_sender = request.get_response_sender()
        if self.idx == 0:
            out_output = pb_utils.Tensor("OUT", np.array([0], np.int32))
            response = pb_utils.InferenceResponse(output_tensors=[out_output])
            response_sender.send(response)
            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            request.set_release_flags(pb_utils.TRITONSERVER_REQUEST_RELEASE_RESCHEDULE)
            self.idx = 1

    def process_request_thread(self, request):
        thread = threading.Thread(
            target=self.response_thread,
            args=(
                request.get_response_sender(),
                pb_utils.get_input_tensor_by_name(request, "IN").as_numpy(),
            ),
        )

        thread.daemon = True

        with self.inflight_thread_count_lck:
            self.inflight_thread_count += 1

        if self.idx == 0:
            request.set_release_flags(pb_utils.TRITONSERVER_REQUEST_RELEASE_RESCHEDULE)
            thread.start()
            self.idx = 1

    def response_thread(self, response_sender, in_input):
        output_value = in_input[0]
        while output_value >= 0:
            out_output = pb_utils.Tensor("OUT", np.array([output_value], np.int32))
            response = pb_utils.InferenceResponse(output_tensors=[out_output])
            response_sender.send(response)
            output_value -= 1

        response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        with self.inflight_thread_count_lck:
            self.inflight_thread_count -= 1
