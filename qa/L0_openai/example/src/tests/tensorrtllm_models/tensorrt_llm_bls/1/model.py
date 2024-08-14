# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import traceback

import triton_python_backend_utils as pb_utils
from lib.triton_decoder import TritonDecoder


class TritonPythonModel:
    def initialize(self, args):
        # Parse model configs
        model_config = json.loads(args["model_config"])

        params = model_config["parameters"]

        accumulate_tokens_str = ""
        if "accumulate_tokens" in params:
            accumulate_tokens_str = params["accumulate_tokens"]["string_value"]

        self.accumulate_tokens = accumulate_tokens_str.lower() in [
            "true",
            "yes",
            "1",
            "t",
        ]

        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(model_config)

        self.logger = pb_utils.Logger

        self.llm_model_name = "tensorrt_llm"
        if "tensorrt_llm_model_name" in params:
            self.llm_model_name = params["tensorrt_llm_model_name"]["string_value"]
        self.draft_llm_model_name = None
        if "tensorrt_llm_draft_model_name" in params:
            self.draft_llm_model_name = params["tensorrt_llm_draft_model_name"][
                "string_value"
            ]

        self.decoder = TritonDecoder(
            streaming=self.decoupled,
            accumulate=self.accumulate_tokens,
            preproc_model_name="preprocessing",
            postproc_model_name="postprocessing",
            llm_model_name=self.llm_model_name,
            draft_llm_model_name=self.draft_llm_model_name,
        )

    def execute(self, requests):
        responses = []

        for request in requests:
            if self.decoupled:
                response_sender = request.get_response_sender()
            try:
                req = self.decoder.convert_triton_request(request)
                req.validate()
                # print(f"[DEBUG] ========= [bls model.py] {req.temperature=} ===========")
                speculative_decode = (
                    req.num_draft_tokens is not None and req.num_draft_tokens[0][0] > 0
                )
                if speculative_decode and (
                    self.draft_llm_model_name is None or self.draft_llm_model_name == ""
                ):
                    raise Exception(
                        "cannot perform speculative decoding without draft model"
                    )
                res_gen = self.decoder.decode(
                    req, speculative_decoding=speculative_decode
                )

                for res in res_gen:
                    triton_response = self.decoder.create_triton_response(res)
                    if self.decoupled:
                        response_sender.send(triton_response)
                    else:
                        responses.append(triton_response)

                if self.decoupled:
                    response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )

            except Exception:
                self.logger.log_error(traceback.format_exc())
                # If encountering an error, send a response with err msg
                error_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(traceback.format_exc()),
                )

                if self.decoupled:
                    response_sender.send(error_response)
                    response_sender.send(
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    )
                else:
                    responses.append(error_response)

            self.decoder.reset_decoder()
            if self.decoupled:
                return None
            else:
                assert len(responses) == len(requests)
                return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
