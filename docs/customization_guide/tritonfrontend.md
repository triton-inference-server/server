### Triton Server (tritonfrontend) Bindings

The `tritonfrontend` python package is a set of bindings to Triton's existing frontends implemented in C++. Currently, `tritonfrontend` supports starting up `KServeHttp` and `KServeGrpc` frontends. These bindings used in-combination with Triton's Python In-Process API ([`tritonserver`](https://github.com/triton-inference-server/core/tree/main/python/tritonserver)) and [`tritonclient`](https://github.com/triton-inference-server/client/tree/main/src/python/library) extend the ability to use Triton's full feature set with a couple of lines of Python.

Let us walk through a simple example:
1. First we need to load the desired models and start the server with `tritonserver`.
```python
import tritonserver

# Constructing path to Model Repository
model_path = f"server/src/python/examples/example_model_repository"

server_options = tritonserver.Options(
    server_id="ExampleServer",
    model_repository=model_path,
    log_error=True,
    log_warn=True,
    log_info=True,
)
server = tritonserver.Server(server_options).start(wait_until_ready=True)
```
Note: `model_path` may need to be edited depending on your setup.


2. Now, to start up the respective services with `tritonfrontend`
```python
from tritonfrontend import KServeHttp, KServeGrpc
http_options = KServeHttp.Options(thread_count=5)
http_service = KServeHttp.Server(server, http_options)
http_service.start()

# Default options (if none provided)
grpc_service = KServeGrpc.Server(server)
grpc_service.start()
```

3. Finally, with running services, we can use `tritonclient` or simple `curl` commands to send requests and receive responses from the frontends.

```python
import tritonclient.http as httpclient
import numpy as np # Use version numpy < 2
model_name = "identity" # output == input
url = "localhost:8000"

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

# Print results
print("[INFERENCE RESULTS]")
print("Output data:", output_data)

# Stop respective services and server.
http_service.stop()
grpc_service.stop()
server.stop()
```

---

Additionally, `tritonfrontend` provides context manager support as well. So steps 2-3, could also be achieved through:
```python
from tritonfrontend import KServeHttp
import tritonclient.http as httpclient
import numpy as np  # Use version numpy < 2

with KServeHttp.Server(server) as http_service:
    # The identity model returns an exact duplicate of the input data as output
    model_name = "identity"
    url = "localhost:8000"
    # Create a Triton client
    with httpclient.InferenceServerClient(url=url) as client:
        # Prepare input data
        input_data = np.array(["Roger Roger"], dtype=object)
        # Create input and output objects
        inputs = [httpclient.InferInput("INPUT0", input_data.shape, "BYTES")]
        # Set the data for the input tensor
        inputs[0].set_data_from_numpy(input_data)
        # Perform inference
        results = client.infer(model_name, inputs=inputs)
        # Get the output data
        output_data = results.as_numpy("OUTPUT0")
        # Print results
        print("[INFERENCE RESULTS]")
        print("Output data:", output_data)

server.stop()
```
With this workflow, you can avoid having to stop each service after client requests have terminated.


## Known Issues
- The following features are not currently supported when launching the Triton frontend services through the python bindings:
    - [Tracing](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/trace.md)
    - [Shared Memory](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_shared_memory.md)
    - [Metrics](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/metrics.md)
    - [Restricted Protocols](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/inference_protocols.md#limit-endpoint-access-beta)
    - VertexAI
    - Sagemaker
- After a running server has been stopped, if the client sends an inference request, a Segmentation Fault will occur.