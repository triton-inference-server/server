import json
import time
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        inputs = [{'name': 'INPUT0', 'data_type': 'TYPE_FP32', 'dims': [1]}]
        outputs = [{'name': 'OUTPUT0', 'data_type': 'TYPE_STRING', 'dims': [1]}]

        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config['input']:
            input_names.append(input['name'])
        for output in config['output']:
            output_names.append(output['name'])

        for input in inputs:
            if input['name'] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            if output['name'] not in output_names:
                auto_complete_model_config.add_output(output)

        auto_complete_model_config.set_max_batch_size(4)
        auto_complete_model_config.set_dynamic_batching()

        return auto_complete_model_config

    def execute(self, requests):
        # A simple model that puts the parameters in the in the request in the
        # output.
        responses = []
        for request in requests:
            output0 = np.asarray(request.parameters(), dtype=np.object)
            output_tensor = pb_utils.Tensor("OUTPUT0", output0)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor])
            responses.append(inference_response)

        return responses
