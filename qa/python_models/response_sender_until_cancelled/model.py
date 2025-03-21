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

import time

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """This model will keep repeating the INPUT as the OUTPUT,
    until the request is being cancelled or
    the MAX_NUMBER_OF_RESPONSE has been reached.
    """

    def execute(self, requests):
        request = requests[0]

        input = pb_utils.get_input_tensor_by_name(request, "INPUT").as_numpy()
        max_num_of_response = pb_utils.get_input_tensor_by_name(
            request, "MAX_NUMBER_OF_RESPONSE"
        ).as_numpy()[0]
        delay = pb_utils.get_input_tensor_by_name(request, "DELAY").as_numpy()[0]
        response_sender = request.get_response_sender()

        sent = 0
        while True:
            if request.is_cancelled():
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        message="request has been cancelled",
                        code=pb_utils.TritonError.CANCELLED,
                    )
                )
                response_sender.send(
                    response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
                break

            output = pb_utils.Tensor("OUTPUT", np.array([input[0]], np.int32))
            response = pb_utils.InferenceResponse(output_tensors=[output])

            if sent + 1 == max_num_of_response:
                response_sender.send(
                    response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
                break
            else:
                response_sender.send(response)
                sent += 1
                time.sleep(delay / 1000)

        return None
