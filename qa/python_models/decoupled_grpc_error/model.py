# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import threading
import time

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """
    Decoupled model that always sends an error.
    MODE input:
    "MODE_ERROR_ONLY" (error response without FINAL flag),
    "MODE_ERROR_FINAL" (error response with FINAL flag),
    "MODE_ERROR_WITH_DELAYED_FINAL" (error response then delayed FINAL response).
    Used to test gRPC server error handling behavior (e.g. triton_grpc_error stream closure and subsequent responses handling).
    """

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        if not pb_utils.using_decoupled_model_transaction_policy(self.model_config):
            raise pb_utils.TritonModelException(
                "This model requires decoupled transaction policy"
            )
        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()

    def execute(self, requests):
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "IN").as_numpy()
            mode_tensor = pb_utils.get_input_tensor_by_name(request, "MODE").as_numpy()
            mode = mode_tensor.flat[0].decode("utf-8")
            thread = threading.Thread(
                target=self._response_thread,
                args=(request.get_response_sender(), in_tensor, mode),
            )
            thread.daemon = True
            with self.inflight_thread_count_lck:
                self.inflight_thread_count += 1
            thread.start()
        return None

    def _response_thread(self, response_sender, in_tensor, mode):
        # Send a normal response first.
        out_tensor = pb_utils.Tensor("OUT", in_tensor)
        response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
        response_sender.send(response)

        # Send an error response next.
        error = pb_utils.TritonError("An error occurred during execution")
        response = pb_utils.InferenceResponse(error=error)

        if mode == "MODE_ERROR_ONLY" or mode == "MODE_ERROR_WITH_DELAYED_FINAL":
            response_sender.send(response)
        elif mode == "MODE_ERROR_FINAL":
            response_sender.send(
                response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Send a delayed FINAL flag.
        if mode == "MODE_ERROR_WITH_DELAYED_FINAL":
            time.sleep(0.5)
            response_sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        with self.inflight_thread_count_lck:
            self.inflight_thread_count -= 1

    def finalize(self):
        inflight_threads = True
        while inflight_threads:
            with self.inflight_thread_count_lck:
                inflight_threads = self.inflight_thread_count != 0
            if inflight_threads:
                time.sleep(0.1)
