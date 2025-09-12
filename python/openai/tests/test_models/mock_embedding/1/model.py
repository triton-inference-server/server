import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
            input_count = len(input_tensor.as_numpy())
            embedding_dim = 384
            out_tensor = pb_utils.Tensor(
                "embedding",
                np.random.randn(input_count, embedding_dim).astype(np.float32),
            )
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
