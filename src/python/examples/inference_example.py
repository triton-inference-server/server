import pathlib

import numpy as np
import tritonclient.http as httpclient
import tritonserver
from tritonfrontend import KServeHttp

# Constructing path to Model Repository
model_path = f"{pathlib.Path(__file__).parent.resolve()}/example_model_repository"

# Selecting Server Options
server_options = tritonserver.Options(
    server_id="ExampleServer",
    model_repository=model_path,
    log_error=True,
    log_warn=True,
    log_info=True,
)

# Creating server instance
server = tritonserver.Server(server_options).start(wait_until_ready=True)

# Selecting Options for KServeHttp Frontend
http_options = KServeHttp.Options(port=8005)

with KServeHttp.Server(server, http_options) as http_service:
    # The identity model returns an exact duplicate of the input data as output
    model_name = "identity"
    url = "localhost:8005"

    # Create a Triton client
    client = httpclient.InferenceServerClient(url=url)

    # Prepare input data
    input_data = np.array([["Roger Roger"]], dtype=object)

    # Create input and output objects
    inputs = [httpclient.InferInput("INPUT0", input_data.shape, "BYTES")]

    # Set the data for the input tensor
    inputs[0].set_data_from_numpy(input_data)

    results = client.infer(model_name, inputs=inputs)

    # Get the output data
    output_data = results.as_numpy("OUTPUT0")

    print("--------------------- INFERENCE RESULTS ---------------------")
    print("Input data:", input_data)
    print("Output data:", output_data)
    print("-------------------------------------------------------------")


server.stop()
