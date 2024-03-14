import json
import time

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        inputs = [{"name": "INPUT0", "data_type": "TYPE_FP32", "dims": [1]}]
        outputs = [{"name": "OUTPUT0", "data_type": "TYPE_STRING", "dims": [-1]}]

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

        auto_complete_model_config.set_max_batch_size(1)

        return auto_complete_model_config

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(self.model_config, "OUTPUT0")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            context = request.trace().get_context(mode="opentelemetry")
            output_tensor = pb_utils.Tensor(
                "OUTPUT0", np.array(context).astype(np.bytes_)
            )
            inference_response = pb_utils.InferenceResponse([output_tensor])
            responses.append(inference_response)

        return responses
