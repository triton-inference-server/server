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
import threading
import time

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self._logger = pb_utils.Logger
        self._model_config = json.loads(args["model_config"])
        self._using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            self._model_config
        )

    def execute(self, requests):
        processed_requests = []
        for request in requests:
            delay_tensor = pb_utils.get_input_tensor_by_name(
                request, "EXECUTE_DELAY"
            ).as_numpy()
            delay = delay_tensor[0][0]  # seconds
            if self._using_decoupled:
                processed_requests.append(
                    {"response_sender": request.get_response_sender(), "delay": delay}
                )
            else:
                processed_requests.append({"request": request, "delay": delay})
        if self._using_decoupled:
            return self._execute_decoupled(processed_requests)
        return self._execute_processed_requests(processed_requests)

    def _execute_processed_requests(self, processed_requests):
        responses = []
        for processed_request in processed_requests:
            error = pb_utils.TritonError(message="not cancelled")
            object_to_check_cancelled = None
            if "response_sender" in processed_request:
                object_to_check_cancelled = processed_request["response_sender"]
            elif "request" in processed_request:
                object_to_check_cancelled = processed_request["request"]
            delay = processed_request["delay"]  # seconds
            time_elapsed = 0.0  # seconds
            while time_elapsed < delay:
                time.sleep(1)
                time_elapsed += 1.0
                if object_to_check_cancelled.is_cancelled():
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

    def _execute_decoupled(self, processed_requests):
        def response_thread(execute_processed_requests, processed_requests):
            time.sleep(2)  # execute after requests are released
            responses = execute_processed_requests(processed_requests)
            for i in range(len(responses)):  # len(responses) == len(processed_requests)
                response_sender = processed_requests[i]["response_sender"]
                response_sender.send(responses[i])
                response_sender.send(
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )

        thread = threading.Thread(
            target=response_thread,
            args=(self._execute_processed_requests, processed_requests),
        )
        thread.daemon = True
        thread.start()
        return None
