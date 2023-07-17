# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    def initialize(self, args):
        logger = pb_utils.Logger
        logger.log("Initialize-Specific Msg!", logger.INFO)
        logger.log_info("Initialize-Info Msg!")
        logger.log_warn("Initialize-Warning Msg!")
        logger.log_error("Initialize-Error Msg!")
        logger.log_verbose("Initialize-Verbose Msg!")

    def execute(self, requests):
        """
        Identity model in Python backend.
        """
        # Log as early as possible
        logger = pb_utils.Logger
        logger.log("Execute-Specific Msg!", logger.INFO)
        logger.log_info("Execute-Info Msg!")
        logger.log_warn("Execute-Warning Msg!")
        logger.log_error("Execute-Error Msg!")
        logger.log_verbose("Execute-Verbose Msg!")

        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            out_tensor = pb_utils.Tensor("OUTPUT0", input_tensor.as_numpy())
            responses.append(pb_utils.InferenceResponse([out_tensor]))

        # Log as late as possible
        logger.log("Execute-Specific Msg!", logger.INFO)
        logger.log_info("Execute-Info Msg!")
        logger.log_warn("Execute-Warning Msg!")
        logger.log_error("Execute-Error Msg!")
        logger.log_verbose("Execute-Verbose Msg!")

        return responses

    def finalize(self):
        logger = pb_utils.Logger
        logger.log("Finalize-Specific Msg!", logger.INFO)
        logger.log_info("Finalize-Info Msg!")
        logger.log_warn("Finalize-Warning Msg!")
        logger.log_error("Finalize-Error Msg!")
        logger.log_verbose("Finalize-Verbose Msg!")
