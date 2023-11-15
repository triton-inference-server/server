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

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        error_code_map = {
            "UNKNOWN": pb_utils.TritonError.UNKNOWN,
            "INTERNAL": pb_utils.TritonError.INTERNAL,
            "NOT_FOUND": pb_utils.TritonError.NOT_FOUND,
            "INVALID_ARG": pb_utils.TritonError.INVALID_ARG,
            "UNAVAILABLE": pb_utils.TritonError.UNAVAILABLE,
            "UNSUPPORTED": pb_utils.TritonError.UNSUPPORTED,
            "ALREADY_EXISTS": pb_utils.TritonError.ALREADY_EXISTS,
            "CANCELLED": pb_utils.TritonError.CANCELLED,
        }

        responses = []

        for request in requests:
            err_code_tensor = pb_utils.get_input_tensor_by_name(
                request, "ERROR_CODE"
            ).as_numpy()
            err_code_str = str(err_code_tensor[0][0], encoding="utf-8")
            if err_code_str in error_code_map:
                error = pb_utils.TritonError(
                    message=("error code: " + err_code_str),
                    code=error_code_map[err_code_str],
                )
            else:
                error = pb_utils.TritonError("unrecognized error code: " + err_code_str)
            responses.append(pb_utils.InferenceResponse(error=error))

        return responses
