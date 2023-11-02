# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
import os

import triton_python_backend_utils as pb_utils

_ADD_SUB_ARGS_FILENAME = "model.json"


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """This function is called only once when loading the model assuming
        the server was not started with `--disable-auto-complete-config`.

        Parameters
        ----------
        auto_complete_model_config : pb_utils.ModelConfig
          An object containing the existing model configuration.

        Returns
        -------
        pb_utils.ModelConfig
          An object containing the auto-completed model configuration
        """
        inputs = [
            {"name": "INPUT0", "data_type": "TYPE_FP32", "dims": [4]},
            {"name": "INPUT1", "data_type": "TYPE_FP32", "dims": [4]},
        ]
        outputs = [{"name": "OUTPUT", "data_type": "TYPE_FP32", "dims": [4]}]

        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []

        for input in config["input"]:
            input_names.append(input["name"])

        for output in config["output"]:
            output_names.append(output["name"])

        for input in inputs:
            if input["name"] not in input_names:
                auto_complete_model_config.add_input(input)

        for output in outputs:
            if output["name"] not in output_names:
                auto_complete_model_config.add_output(output)

        return auto_complete_model_config

    def initialize(self, args):
        """This function allows the model to initialize any state associated with this model.

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

        self.model_config = model_config = json.loads(args["model_config"])

        # Get OUTPUT configuration
        output_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT")

        engine_args_filepath = os.path.join(
            pb_utils.get_model_dir(), _ADD_SUB_ARGS_FILENAME
        )
        assert os.path.isfile(
            engine_args_filepath
        ), f"'{_ADD_SUB_ARGS_FILENAME}' containing add sub model args must be provided in '{pb_utils.get_model_dir()}'"

        with open(engine_args_filepath) as file:
            self.add_sub_config = json.load(file)

        assert (
            "operation" in self.add_sub_config
        ), f"Missing required key 'operation' in {_ADD_SUB_ARGS_FILENAME}"

        extra_keys = set(self.add_sub_config.keys()) - {"operation"}
        assert (
            not extra_keys
        ), f"Unsupported keys are provided in {_ADD_SUB_ARGS_FILENAME}: {', '.join(extra_keys)}"

        assert self.add_sub_config["operation"] in [
            "add",
            "sub",
        ], f"'operation' value must be 'add' or 'sub' in {_ADD_SUB_ARGS_FILENAME}"

        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):
        """This function is called when an inference request is made
        for this model.

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

        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1")

            if self.add_sub_config["operation"] == "add":
                out = in_0.as_numpy() + in_1.as_numpy()
            else:
                out = in_0.as_numpy() - in_1.as_numpy()

            # Create output tensors.
            out_tensor = pb_utils.Tensor("OUTPUT", out.astype(self.output_dtype))

            # Create InferenceResponse.
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(inference_response)

        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded."""
        print("Cleaning up...")
