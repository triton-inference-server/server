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

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    This model takes 1 input tensor, an INT32 [ 1 ] input named "IN", and
    produces an output tensor "OUT" with the same shape as the input tensor.
    The input value indicates the total number of responses to be generated and
    the output value indicates the number of remaining responses. For example,
    if the request input has value 2, the model will:
        - Send a response with value 1.
        - Release request with RESCHEDULE flag.
        - When execute on the same request, send the last response with value 0.
        - Release request with ALL flag.
    """

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

        self.remaining_response = 0
        self.reset_flag = True

    def execute(self, requests):
        for request in requests:
            in_input = pb_utils.get_input_tensor_by_name(request, "IN").as_numpy()

            if self.reset_flag:
                self.remaining_response = in_input[0]
                self.reset_flag = False

            response_sender = request.get_response_sender()

            self.remaining_response -= 1

            out_output = pb_utils.Tensor(
                "OUT", np.array([self.remaining_response], np.int32)
            )
            response = pb_utils.InferenceResponse(output_tensors=[out_output])

            if self.remaining_response <= 0:
                response_sender.send(
                    response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
            else:
                request.set_release_flags(
                    pb_utils.TRITONSERVER_REQUEST_RELEASE_RESCHEDULE
                )
                response_sender.send(response)

        return None
