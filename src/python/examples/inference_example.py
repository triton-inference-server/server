import pathlib

import numpy as np
import tritonclient.http as httpclient
import tritonserver
from tritonfrontend import KServeHttp

model_path = f"{pathlib.Path(__file__).parent.resolve()}/../test/test_model_repository"
server_options = tritonserver.Options(
    server_id="ExampleServer",
    model_repository=model_path,
    log_error=True,
    log_warn=True,
    log_info=True,
)
server = tritonserver.Server(server_options).start()

http_options = KServeHttp.Options()
http_service = KServeHttp.Server(server, http_options)
http_service.start()

model_name = "identity"
url = "0.0.0.0:8000"

# Create a Triton client
client = httpclient.InferenceServerClient(url=url)

# Prepare input data
input_data = np.array([["Roger Roger"]], dtype=object)

# Create input and output objects
inputs = [httpclient.InferInput("INPUT0", input_data.shape, "BYTES")]
outputs = [httpclient.InferRequestedOutput("OUTPUT0")]

# Set the data for the input tensor
inputs[0].set_data_from_numpy(input_data)

results = client.infer(model_name, inputs=inputs, outputs=outputs)

# Get the output data
output_data = results.as_numpy("OUTPUT0")
print("Input data:", input_data)
print("Output data:", output_data)

http_service.stop()
