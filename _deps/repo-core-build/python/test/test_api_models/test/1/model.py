import json

import numpy as np
import triton_python_backend_utils as pb_utils

try:
    import cupy
except:
    cupy = None


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        inputs = []
        outputs = []
        dims = [-1, -1]
        optional = True
        config = auto_complete_model_config.as_dict()

        for data_type in pb_utils.TRITON_STRING_TO_NUMPY.keys():
            type_name = data_type.split("_")[1].lower()
            input_name = f"{type_name}_input"
            output_name = f"{type_name}_output"
            inputs.append(
                {
                    "name": input_name,
                    "data_type": data_type,
                    "dims": dims,
                    "optional": optional,
                }
            )
            outputs.append({"name": output_name, "data_type": data_type, "dims": dims})

        outputs.append(
            {"name": "output_parameters", "data_type": "TYPE_STRING", "dims": [1]}
        )
        for input_ in inputs:
            auto_complete_model_config.add_input(input_)
        for output in outputs:
            auto_complete_model_config.add_output(output)

        auto_complete_model_config.set_max_batch_size(0)

        if config["parameters"]["decoupled"]["string_value"] == "True":
            auto_complete_model_config.set_model_transaction_policy({"decoupled": True})

        return auto_complete_model_config

    def initialize(self, args):
        self._model_config = json.loads(args["model_config"])
        self._decoupled = self._model_config.get("model_transaction_policy", {}).get(
            "decoupled"
        )
        self._request_gpu_memory = False
        if "parameters" in self._model_config:
            parameters = self._model_config["parameters"]
            if (
                "request_gpu_memory" in parameters
                and parameters["request_gpu_memory"]["string_value"] == "True"
            ):
                self._request_gpu_memory = True

    def execute_decoupled(self, requests):
        for request in requests:
            sender = request.get_response_sender()
            output_tensors = []
            for input_tensor in request.inputs():
                input_value = input_tensor.as_numpy()
                output_tensor = pb_utils.Tensor(
                    input_tensor.name().replace("input", "output"), input_value
                )
                output_tensors.append(output_tensor)
            sender.send(pb_utils.InferenceResponse(output_tensors=output_tensors))
            sender.send(flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
        return None

    def execute(self, requests):
        if self._decoupled:
            return self.execute_decoupled(requests)
        responses = []
        for request in requests:
            output_tensors = []
            parameters = []
            for input_tensor in request.inputs():
                input_value = input_tensor.as_numpy()

                if self._request_gpu_memory:
                    input_value = cupy.array(input_value)

                    output_tensor = pb_utils.Tensor.from_dlpack(
                        input_tensor.name().replace("input", "output"), input_value
                    )
                else:
                    output_tensor = pb_utils.Tensor(
                        input_tensor.name().replace("input", "output"), input_value
                    )
                output_tensors.append(output_tensor)

            output_parameters = np.array([request.parameters()]).astype(np.object_)
            output_tensors.append(
                pb_utils.Tensor("output_parameters", output_parameters)
            )

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=output_tensors,
                )
            )
        return responses
