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

import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.sequences = {}
        self.decoupled = self.model_config.get("model_transaction_policy", {}).get(
            "decoupled"
        )

    def get_next_sequence_output_tensor(self, request):
        sid = request.correlation_id()
        flags = request.flags()
        if flags == pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_START:
            if sid in self.sequences:
                raise pb_utils.TritonModelException(
                    "Can't start a new sequence with existing ID"
                )
            self.sequences[sid] = [1]
        else:
            if sid not in self.sequences:
                raise pb_utils.TritonModelException(
                    "Need START flag for a sequence ID that doesn't already exist."
                )

            last = self.sequences[sid][-1]
            self.sequences[sid].append(last + 1)

        output = self.sequences[sid][-1]
        output = np.array([output])
        out_tensor = pb_utils.Tensor("OUTPUT0", output.astype(np.int32))
        return out_tensor

    def execute(self, requests):
        if self.decoupled:
            return self.execute_decoupled(requests)
        else:
            return self.execute_non_decoupled(requests)

    def execute_non_decoupled(self, requests):
        responses = []
        for request in requests:
            output_tensor = self.get_next_sequence_output_tensor(request)
            response = pb_utils.InferenceResponse([output_tensor])
            responses.append(response)
        return responses

    def execute_decoupled(self, requests):
        for request in requests:
            sender = request.get_response_sender()
            output_tensor = self.get_next_sequence_output_tensor(request)

            # Send 3 responses per request
            for _ in range(3):
                response = pb_utils.InferenceResponse([output_tensor])
                sender.send(response)

            sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        return None

    def finalize(self):
        print(f"Cleaning up. Final sequences stored: {self.sequences}")
