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

#
# This test case was added based on a prior crash. DO NOT MODIFY!
#

import json
import traceback

import numpy as np
import triton_python_backend_utils as pb_utils


def get_valid_param_value(param, default_value=""):
    value = param.get("string_value", "")
    return default_value if value.startswith("${") or value == "" else value


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        self.output_config = pb_utils.get_output_config_by_name(
            model_config, "text_output"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(
            self.output_config["data_type"]
        )
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(model_config)
        self.logger = pb_utils.Logger

    def create_triton_tensors(self, index):
        x = "bla" + str(index)
        output = [x.encode("utf8")]
        np_output = np.array(output).astype(self.output_dtype)
        seq_idx = np.array([[0]]).astype(np.int32)

        t1 = pb_utils.Tensor("text_output", np_output)
        t2 = pb_utils.Tensor("sequence_index", seq_idx)
        tensors = [t1, t2]
        return tensors

    def create_triton_response(self, index):
        tensors = self.create_triton_tensors(index)
        return pb_utils.InferenceResponse(output_tensors=tensors)

    def execute(self, requests):
        responses = []
        for request in requests:
            if self.decoupled:
                response_sender = request.get_response_sender()
            try:
                for index in range(0, 1):
                    triton_response = self.create_triton_response(index)
                    if self.decoupled:
                        response_sender.send(triton_response)
                    else:
                        responses.append(triton_response)

                if self.decoupled:
                    response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )

            except Exception:
                self.logger.log_error(traceback.format_exc())
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(traceback.format_exc()),
                )

                if self.decoupled:
                    response_sender.send(error_response)
                    response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )
                else:
                    responses.append(error_response)

        if self.decoupled:
            return None
        else:
            assert len(responses) == len(requests)
            return responses

    def finalize(self):
        self.logger.log_info("Cleaning up...")
