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

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """This model will send a decoupled bls request to 'response_sender_until_cancelled' model
    The model will start adding the response from the model.
    When the MAX_SUM is reached. The model will call the response iterarior cancel() method to
    cancel the response stream.
    The number of response should not reach MAX_NUMBER_OF_RESPONSE.
    """

    def execute(self, requests):
        max_sum = (
            pb_utils.get_input_tensor_by_name(requests[0], "MAX_SUM").as_numpy().flat[0]
        )
        input = pb_utils.get_input_tensor_by_name(requests[0], "INPUT")
        delay = pb_utils.Tensor("DELAY", np.array([50], dtype=np.int32))
        max_num_response = pb_utils.Tensor(
            "MAX_NUMBER_OF_RESPONSE", np.array([100], dtype=np.int32)
        )

        infer_request = pb_utils.InferenceRequest(
            model_name="response_sender_until_cancelled",
            inputs=[input, max_num_response, delay],
            requested_output_names=["OUTPUT"],
        )

        response_stream = infer_request.exec(decoupled=True)

        is_cancelled = False
        error = None
        response_sum = 0
        for infer_response in response_stream:
            if infer_response.has_error():
                if infer_response.error().code() == pb_utils.TritonError.CANCELLED:
                    is_cancelled = True
                else:
                    error = infer_response.error()
                break

            out = pb_utils.get_output_tensor_by_name(
                infer_response, "OUTPUT"
            ).as_numpy()[0]
            if response_sum + out > max_sum:
                response_stream.cancel()
            else:
                response_sum += out

        if error is None and not is_cancelled:
            error = pb_utils.TritonError("request is not cancelled!")

        responses = [
            pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("SUM", np.array([response_sum], dtype=np.int32))
                ],
                error=error,
            )
        ]

        return responses
