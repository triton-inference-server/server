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
    """
    This model sends a decoupled bls inference request to 'response_sender_until_cancelled'
    model, and sums up its responses.
    Once the MAX_SUM is reached, the model will call the response iterator's
    cancel() method to cancel the response stream.
    If the IGNORE_CANCEL is set to True, the 'response_sender_until_cancelled' model will not hornor
    the request cancellation and keep sending the output to the model.
    The number of total responses should not reach MAX_RESPONSE_COUNT.
    """

    def execute(self, requests):
        max_sum = (
            pb_utils.get_input_tensor_by_name(requests[0], "MAX_SUM").as_numpy().flat[0]
        )
        input = pb_utils.get_input_tensor_by_name(requests[0], "INPUT")
        ignore_cancel = pb_utils.get_input_tensor_by_name(requests[0], "IGNORE_CANCEL")
        delay = pb_utils.Tensor("DELAY", np.array([50], dtype=np.int32))
        max_response_count = pb_utils.Tensor(
            "MAX_RESPONSE_COUNT", np.array([20], dtype=np.int32)
        )

        infer_request = pb_utils.InferenceRequest(
            model_name="response_sender_until_cancelled",
            inputs=[input, max_response_count, delay, ignore_cancel],
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

            response_sum += out
            if response_sum >= max_sum:
                response_stream.cancel()

        responses = [
            pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("SUM", np.array([response_sum], dtype=np.int32)),
                    pb_utils.Tensor(
                        "IS_CANCELLED", np.array([is_cancelled], dtype=np.bool_)
                    ),
                ],
                error=error,
            )
        ]

        return responses
