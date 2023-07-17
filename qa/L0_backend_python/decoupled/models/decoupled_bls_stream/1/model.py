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

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """This model sends a BLS request to a decoupled model 'square_int32' and
    returns the output from 'square_int32' as responses.
    """

    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config
        )
        if not using_decoupled:
            raise pb_utils.TritonModelException(
                """the model `{}` can generate any number of responses per request,
                enable decoupled transaction policy in model configuration to
                serve this model""".format(
                    args["model_name"]
                )
            )

        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()

    def execute(self, requests):
        """This function is called on inference request."""

        for request in requests:
            thread = threading.Thread(
                target=self.response_thread,
                args=(
                    request.get_response_sender(),
                    pb_utils.get_input_tensor_by_name(request, "IN").as_numpy(),
                ),
            )
            thread.daemon = True
            with self.inflight_thread_count_lck:
                self.inflight_thread_count += 1
            thread.start()

        return None

    def response_thread(self, response_sender, in_value):
        infer_request = pb_utils.InferenceRequest(
            model_name="square_int32",
            requested_output_names=["OUT"],
            inputs=[pb_utils.Tensor("IN", in_value)],
        )
        infer_responses = infer_request.exec(decoupled=True)

        response_count = 0
        for infer_response in infer_responses:
            if len(infer_response.output_tensors()) > 0:
                output0 = pb_utils.get_output_tensor_by_name(infer_response, "OUT")
                if infer_response.has_error():
                    response = pb_utils.InferenceResponse(
                        error=infer_response.error().message()
                    )
                    response_sender.send(
                        response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )
                elif np.any(in_value != output0.as_numpy()):
                    error_message = (
                        "BLS Request input and BLS response output do not match."
                        f" {in_value} != {output0.as_numpy()}"
                    )
                    response = pb_utils.InferenceResponse(error=error_message)
                    response_sender.send(
                        response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )
                else:
                    output_tensors = [pb_utils.Tensor("OUT", output0.as_numpy())]
                    response = pb_utils.InferenceResponse(output_tensors=output_tensors)
                    response_sender.send(response)

            response_count += 1

        if in_value != response_count - 1:
            error_message = "Expected {} responses, got {}".format(
                in_value, len(infer_responses) - 1
            )
            response = pb_utils.InferenceResponse(error=error_message)
            response_sender.send(
                response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            )
        else:
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
