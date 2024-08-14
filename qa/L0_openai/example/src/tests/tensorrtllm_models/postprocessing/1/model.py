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

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # Parse model configs
        model_config = json.loads(args["model_config"])
        tokenizer_dir = model_config["parameters"]["tokenizer_dir"]["string_value"]

        skip_special_tokens = model_config["parameters"].get("skip_special_tokens")
        if skip_special_tokens is not None:
            skip_special_tokens_str = skip_special_tokens["string_value"].lower()
            if skip_special_tokens_str in [
                "true",
                "false",
                "1",
                "0",
                "t",
                "f",
                "y",
                "n",
                "yes",
                "no",
            ]:
                self.skip_special_tokens = skip_special_tokens_str in [
                    "true",
                    "1",
                    "t",
                    "y",
                    "yes",
                ]
            else:
                print(
                    f"[TensorRT-LLM][WARNING] Don't setup 'skip_special_tokens' correctly (set value is {skip_special_tokens['string_value']}). Set it as True by default."
                )
                self.skip_special_tokens = True
        else:
            print(
                f"[TensorRT-LLM][WARNING] Don't setup 'skip_special_tokens'. Set it as True by default."
            )
            self.skip_special_tokens = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, legacy=False, padding_side="left", trust_remote_code=True
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Parse model output configs
        output_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT")

        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for idx, request in enumerate(requests):
            # Get input tensors
            tokens_batch = pb_utils.get_input_tensor_by_name(
                request, "TOKENS_BATCH"
            ).as_numpy()

            # Get sequence length
            sequence_lengths = pb_utils.get_input_tensor_by_name(
                request, "SEQUENCE_LENGTH"
            ).as_numpy()

            # Get cum log probs
            cum_log_probs = pb_utils.get_input_tensor_by_name(request, "CUM_LOG_PROBS")

            # Get sequence length
            output_log_probs = pb_utils.get_input_tensor_by_name(
                request, "OUTPUT_LOG_PROBS"
            )

            # Get context logits
            context_logits = pb_utils.get_input_tensor_by_name(
                request, "CONTEXT_LOGITS"
            )

            # Get generation logits
            generation_logits = pb_utils.get_input_tensor_by_name(
                request, "GENERATION_LOGITS"
            )

            # Reshape Input
            # tokens_batch = tokens_batch.reshape([-1, tokens_batch.shape[0]])
            # tokens_batch = tokens_batch.T

            # Postprocessing output data.
            outputs = self._postprocessing(tokens_batch, sequence_lengths)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            output_tensor = pb_utils.Tensor(
                "OUTPUT", np.array(outputs).astype(self.output_dtype)
            )

            outputs = []
            outputs.append(output_tensor)

            if cum_log_probs:
                out_cum_log_probs = pb_utils.Tensor(
                    "OUT_CUM_LOG_PROBS", cum_log_probs.as_numpy()
                )
                outputs.append(out_cum_log_probs)
            else:
                out_cum_log_probs = pb_utils.Tensor(
                    "OUT_CUM_LOG_PROBS", np.array([[0.0]], dtype=np.float32)
                )
                outputs.append(out_cum_log_probs)

            if output_log_probs:
                out_output_log_probs = pb_utils.Tensor(
                    "OUT_OUTPUT_LOG_PROBS", output_log_probs.as_numpy()
                )
                outputs.append(out_output_log_probs)
            else:
                out_output_log_probs = pb_utils.Tensor(
                    "OUT_OUTPUT_LOG_PROBS", np.array([[[0.0]]], dtype=np.float32)
                )
                outputs.append(out_output_log_probs)

            if context_logits:
                out_context_logits = pb_utils.Tensor(
                    "OUT_CONTEXT_LOGITS", context_logits.as_numpy()
                )
                outputs.append(out_context_logits)
            else:
                out_context_logits = pb_utils.Tensor(
                    "OUT_CONTEXT_LOGITS", np.array([[[0.0]]], dtype=np.float32)
                )
                outputs.append(out_context_logits)

            if generation_logits:
                out_generation_logits = pb_utils.Tensor(
                    "OUT_GENERATION_LOGITS", generation_logits.as_numpy()
                )
                outputs.append(out_generation_logits)
            else:
                out_generation_logits = pb_utils.Tensor(
                    "OUT_GENERATION_LOGITS", np.array([[[[0.0]]]], dtype=np.float32)
                )
                outputs.append(out_generation_logits)

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")

    def _postprocessing(self, tokens_batch, sequence_lengths):
        outputs = []
        for batch_idx, beam_tokens in enumerate(tokens_batch):
            for beam_idx, tokens in enumerate(beam_tokens):
                seq_len = sequence_lengths[batch_idx][beam_idx]
                output = self.tokenizer.decode(
                    tokens[:seq_len], skip_special_tokens=self.skip_special_tokens
                )
                outputs.append(output.encode("utf8"))
        return outputs
