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
    def execute(self, requests):
        responses = []

        for request in requests:
            num_params = int(
                pb_utils.get_input_tensor_by_name(
                    request, "NUMBER_PARAMETERS"
                ).as_numpy()[0]
            )
            params = json.loads(request.parameters())

            if num_params == 0:
                # Base case where the received parameters are returned as JSON
                response = json.dumps(params)
                response_tensors = [
                    pb_utils.Tensor(
                        "PARAMETERS_AGGREGATED", np.array([response], dtype=np.object_)
                    )
                ]
            else:
                # Add the parameters of num_params step to the received parameters
                params["bool_" + str(num_params)] = bool(num_params)
                params["int_" + str(num_params)] = num_params
                params["str_" + str(num_params)] = str(num_params)
                # Complete any remaining steps [1, num_params - 1] by calling self
                # recursively via BLS
                bls_request_tensor = pb_utils.Tensor(
                    "NUMBER_PARAMETERS", np.array([num_params - 1], dtype=np.ubyte)
                )
                bls_request = pb_utils.InferenceRequest(
                    model_name="bls_parameters",
                    inputs=[bls_request_tensor],
                    requested_output_names=["PARAMETERS_AGGREGATED"],
                    parameters=params,
                )
                bls_response = bls_request.exec()
                response_tensors = bls_response.output_tensors()

            inference_response = pb_utils.InferenceResponse(
                output_tensors=response_tensors
            )
            responses.append(inference_response)

        return responses
