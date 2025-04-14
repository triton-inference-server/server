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
import time

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.decoupled = self.model_config.get("model_transaction_policy", {}).get(
            "decoupled"
        )

    def execute(self, requests):
        if self.decoupled:
            return self.exec_decoupled(requests)
        else:
            return self.exec(requests)

    def exec(self, requests):
        responses = []
        for request in requests:
            params = json.loads(request.parameters())
            rep_count = params["REPETITION"] if "REPETITION" in params else 1

            input_np = pb_utils.get_input_tensor_by_name(request, "PROMPT").as_numpy()
            stream_np = pb_utils.get_input_tensor_by_name(request, "STREAM").as_numpy()
            stream = stream_np.flatten()[0]
            if stream:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(
                            "STREAM only supported in decoupled mode"
                        )
                    )
                )
            else:
                out_tensor = pb_utils.Tensor(
                    "TEXT", np.repeat(input_np, rep_count, axis=1)
                )
                responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses

    def exec_decoupled(self, requests):
        for request in requests:
            params = json.loads(request.parameters())
            rep_count = params["REPETITION"] if "REPETITION" in params else 1
            fail_last = params["FAIL_LAST"] if "FAIL_LAST" in params else False
            delay = params["DELAY"] if "DELAY" in params else None
            output_0_dim = params["OUTPUT_0_DIM"] if "OUTPUT_0_DIM" in params else False

            sender = request.get_response_sender()
            input_np = pb_utils.get_input_tensor_by_name(request, "PROMPT").as_numpy()
            stream_np = pb_utils.get_input_tensor_by_name(request, "STREAM").as_numpy()
            out_value = np.array([]) if output_0_dim else input_np
            out_tensor = pb_utils.Tensor("TEXT", out_value)
            response = pb_utils.InferenceResponse([out_tensor])
            # If stream enabled, just send multiple copies of response
            # FIXME: Could split up response string into tokens, but this is simpler for now.
            stream = stream_np.flatten()[0]
            if stream:
                for _ in range(rep_count):
                    if delay is not None:
                        time.sleep(delay)
                    if not sender.is_cancelled():
                        sender.send(response)
                    else:
                        break
                sender.send(
                    None
                    if not fail_last
                    else pb_utils.InferenceResponse(
                        error=pb_utils.TritonError("An Error Occurred")
                    ),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )
            # If stream disabled, just send one response
            else:
                sender.send(
                    response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
        return None
