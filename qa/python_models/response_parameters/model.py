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
    def execute(self, requests):
        responses = []

        for request in requests:
            res_params_tensor = pb_utils.get_input_tensor_by_name(
                request, "RESPONSE_PARAMETERS"
            ).as_numpy()
            res_params_str = str(res_params_tensor[0][0], encoding="utf-8")
            output_tensor = pb_utils.Tensor(
                "OUTPUT", np.array([[res_params_str]], dtype=np.object_)
            )
            try:
                res_params = json.loads(res_params_str)
                # convert all digit keys to int, for testing non-str key types
                if isinstance(res_params, dict):
                    res_params_new = {}
                    for key, value in res_params.items():
                        if isinstance(key, str) and key.isdigit():
                            key = int(key)
                        res_params_new[key] = value
                    res_params = res_params_new

                response = pb_utils.InferenceResponse(
                    output_tensors=[output_tensor], parameters=res_params
                )

                res_params_set = {}
                if response.parameters() != "":
                    res_params_set = json.loads(response.parameters())
                if res_params_set != res_params:
                    raise Exception("Response parameters set differ from provided")
            except Exception as e:
                error = pb_utils.TritonError(
                    message=str(e), code=pb_utils.TritonError.INVALID_ARG
                )
                response = pb_utils.InferenceResponse(error=error)

            responses.append(response)

        return responses
