# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    This model is designed to test sending back response parameters when using BLS
    with decoupled model transaction policy.

    The only difference vs. response_parameters_bls model is the BLS composing model
    (i.e. response_parameters_decoupled) turns on decoupled model transaction policy.
    For more details, please check response_parameters_bls model.
    """

    def execute(self, requests):
        responses = []

        for request in requests:
            bls_input_tensor = pb_utils.get_input_tensor_by_name(
                request, "RESPONSE_PARAMETERS"
            )
            bls_request = pb_utils.InferenceRequest(
                model_name="response_parameters_decoupled",
                inputs=[bls_input_tensor],
                requested_output_names=["OUTPUT"],
            )

            res_params_numpy = bls_input_tensor.as_numpy()
            res_params_str = str(res_params_numpy[0][0], encoding="utf-8")
            res_params = json.loads(res_params_str)
            try:
                bls_responses = bls_request.exec(decoupled=True)

                for bls_response, r_params in zip(bls_responses, res_params):
                    if bls_response.has_error():
                        raise Exception(bls_response.error().message())

                    r_params_set = {}
                    if bls_response.parameters() != "":
                        r_params_set = json.loads(bls_response.parameters())
                        if r_params_set != r_params:
                            raise Exception(
                                "Response parameters set differ from provided"
                            )

                # no need to send back anything in the response since we already do the
                # parameters matching checking above.
                response = pb_utils.InferenceResponse()
            except Exception as e:
                error = pb_utils.TritonError(
                    message=str(e), code=pb_utils.TritonError.INVALID_ARG
                )
                response = pb_utils.InferenceResponse(error=error)

            responses.append(response)

        return responses
