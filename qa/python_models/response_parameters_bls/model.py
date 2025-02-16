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
    This model (A) is designed to test sending back response parameters when using BLS.
    It takes one input tensor, which is the RESPONSE_PARAMETERS and uses BLS to
    call response_parameters model (B). Model B would set RESPONSE_PARAMETERS (with a bit
    of data massage) as its response parameters. In the end, model A would also set its
    response parameters from model B's response parameters.

    With above model set up, we can easily test whether the real response parameters are
    the same as the input response parameters.
    """

    def execute(self, requests):
        responses = []

        for request in requests:
            passed = True

            # test bls response parameters from a regular model
            res_params_tensor = pb_utils.get_input_tensor_by_name(
                request, "RESPONSE_PARAMETERS"
            ).as_numpy()
            res_params_str = str(res_params_tensor[0][0], encoding="utf-8")
            res_params = json.loads(res_params_str)
            bls_input_tensor = pb_utils.Tensor("RESPONSE_PARAMETERS", res_params_tensor)
            bls_req = pb_utils.InferenceRequest(
                model_name="response_parameters",
                inputs=[bls_input_tensor],
                requested_output_names=["OUTPUT"],
            )
            bls_res = bls_req.exec()  # decoupled=False
            bls_res_params_str = bls_res.parameters()
            bls_res_params = (
                json.loads(bls_res_params_str) if bls_res_params_str != "" else {}
            )
            passed = passed and bls_res_params == res_params

            # test bls response parameters from a decoupled model
            res_params_decoupled_tensor = pb_utils.get_input_tensor_by_name(
                request, "RESPONSE_PARAMETERS_DECOUPLED"
            ).as_numpy()
            res_params_decoupled_str = str(
                res_params_decoupled_tensor[0][0], encoding="utf-8"
            )
            res_params_decoupled = json.loads(res_params_decoupled_str)
            bls_decoupled_input_tensor = pb_utils.Tensor(
                "RESPONSE_PARAMETERS", res_params_decoupled_tensor
            )  # response_parameters_decoupled model input name is RESPONSE_PARAMETERS
            bls_decoupled_req = pb_utils.InferenceRequest(
                model_name="response_parameters_decoupled",
                inputs=[bls_decoupled_input_tensor],
                requested_output_names=["OUTPUT"],
            )
            bls_decoupled_res = bls_decoupled_req.exec(decoupled=True)
            for bls_decoupled_r in bls_decoupled_res:
                if len(bls_decoupled_r.output_tensors()) == 0:
                    break  # meaning reached final response
                bls_decoupled_r_params_str = bls_decoupled_r.parameters()
                bls_decoupled_r_params = (
                    json.loads(bls_decoupled_r_params_str)
                    if bls_decoupled_r_params_str != ""
                    else {}
                )
                passed = passed and bls_decoupled_r_params in res_params_decoupled
                res_params_decoupled.remove(bls_decoupled_r_params)
            passed = passed and len(res_params_decoupled) == 0

            output_tensor = pb_utils.Tensor(
                "OUTPUT", np.array([[str(passed)]], dtype=np.object_)
            )
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)

        return responses
