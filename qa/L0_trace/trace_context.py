import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        inputs = [{"name": "expect_none", "data_type": "TYPE_BOOL", "dims": [1]}]
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

        return auto_complete_model_config

    def execute(self, requests):
        responses = []
        for request in requests:
            expect_none = pb_utils.get_input_tensor_by_name(
                request, "expect_none"
            ).as_numpy()[0]
            context = request.trace().get_context()
            if expect_none and context is not None:
                raise pb_utils.TritonModelException("Context should be None")
            if not expect_none and context is None:
                raise pb_utils.TritonModelException("Context should NOT be None")

            output_tensor = pb_utils.Tensor(
                "OUTPUT0", np.array(context).astype(np.bytes_)
            )
            inference_response = pb_utils.InferenceResponse([output_tensor])
            responses.append(inference_response)

        return responses
