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

import time

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self._logger = pb_utils.Logger

    def execute(self, requests):
        responses = []
        for request in requests:
            error = pb_utils.TritonError(message="not cancelled")
            delay_tensor = pb_utils.get_input_tensor_by_name(
                request, "EXECUTE_DELAY"
            ).as_numpy()
            delay = delay_tensor[0][0]  # seconds
            time_elapsed = 0.0  # seconds
            while time_elapsed < delay:
                time.sleep(1)
                time_elapsed += 1.0
                if request.is_cancelled():
                    self._logger.log_info(
                        "[execute_cancel] Request cancelled at "
                        + str(time_elapsed)
                        + " s"
                    )
                    error = pb_utils.TritonError(
                        message="cancelled", code=pb_utils.TritonError.CANCELLED
                    )
                    break
                self._logger.log_info(
                    "[execute_cancel] Request not cancelled at "
                    + str(time_elapsed)
                    + " s"
                )
            responses.append(pb_utils.InferenceResponse(error=error))
        return responses
