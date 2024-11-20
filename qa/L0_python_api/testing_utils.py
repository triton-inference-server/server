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

import os
import re
from functools import partial
from typing import Tuple, Union

import numpy as np
import requests
import tritonserver
from tritonfrontend import KServeGrpc, KServeHttp, Metrics


def setup_server(model_repository="test_model_repository") -> tritonserver.Server:
    """
    Using tritonserver, starts a server with the models: identity and delayed_identity
    """
    module_directory = os.path.split(os.path.abspath(__file__))[0]
    model_path = os.path.abspath(os.path.join(module_directory, model_repository))

    # Starting Server Instance
    server_options = tritonserver.Options(
        server_id="TestServer",
        model_repository=model_path,
        log_error=True,
        log_warn=True,
        log_info=True,
    )

    return tritonserver.Server(server_options).start(wait_until_ready=True)


def teardown_server(server: tritonserver.Server) -> None:
    server.stop()


def setup_service(
    server: tritonserver.Server,
    frontend: Union[KServeHttp, KServeGrpc, Metrics],
    options=None,
) -> Union[KServeHttp, KServeGrpc, Metrics]:
    """
    Used to create and start any of the frontends supported by tritonfrontend.
    """
    service = frontend(server=server, options=options)
    service.start()
    return service


def teardown_service(service: Union[KServeHttp, KServeGrpc]) -> None:
    service.stop()


def setup_client(
    frontend_client: Union["tritonclient.http", "tritonclient.grpc"], url: str
):
    """
    Sets up a client to communicate with the Server through the respective protocol.
    """
    return frontend_client.InferenceServerClient(url=url)


def teardown_client(
    client: Union[
        "tritonclient.http.InferenceServerClient",
        "tritonclient.grpc.InferenceServerClient",
    ]
) -> None:
    client.close()


def send_and_test_inference_identity(
    frontend_client: Union[
        "tritonclient.http.InferenceServerClient",
        "tritonclient.grpc.InferenceServerClient",
    ],
    url: str,
) -> bool:
    """
    Sends an inference request to the model at test_model_repository/identity
    and verifies input == output
    """
    model_name = "identity"
    client = setup_client(frontend_client, url)
    input_data = np.array(["testing"], dtype=object)

    # Create input and output objects
    inputs = [frontend_client.InferInput("INPUT0", input_data.shape, "BYTES")]
    outputs = [frontend_client.InferRequestedOutput("OUTPUT0")]
    # Set the data for the input tensor
    inputs[0].set_data_from_numpy(input_data)

    # Perform inference request
    results = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

    output_data = results.as_numpy("OUTPUT0")  # Gather output data

    teardown_client(client)
    return input_data[0] == output_data[0].decode()


def send_and_test_stream_inference(
    frontend_client: Union[
        "tritonclient.http.InferenceServerClient",
        "tritonclient.grpc.InferenceServerClient",
    ],
    url: str,
) -> bool:
    """
    Sends multiple streaming requests to "delayed_identity" model with negligible delays
    and verifies the inputs matches outputs and the ordering is preserved.
    """
    num_requests = 100
    requests = []
    for i in range(num_requests):
        input0_np = np.array([[float(i) / 1000]], dtype=np.float32)
        inputs = [frontend_client.InferInput("INPUT0", input0_np.shape, "FP32")]
        inputs[0].set_data_from_numpy(input0_np)
        requests.append(inputs)

    responses = []

    def callback(responses, result, error):
        responses.append({"result": result, "error": error})

    client = frontend_client.InferenceServerClient(url=url)
    client.start_stream(partial(callback, responses))
    for inputs in requests:
        client.async_stream_infer("delayed_identity", inputs)
    client.stop_stream()
    teardown_client(client)

    assert len(responses) == num_requests
    for i in range(len(responses)):
        assert responses[i]["error"] is None
        output0_np = responses[i]["result"].as_numpy(name="OUTPUT0")
        assert np.allclose(output0_np, [[float(i) / 1000]])

    return True  # test passed


def send_and_test_generate_inference() -> bool:
    """
    Sends an inference request to and identity model through the
    HTTP generate endpoint and verifies input == output
    """
    model_name = "identity"
    url = f"http://localhost:8000/v2/models/{model_name}/generate"
    input_text = "testing"
    data = {
        "INPUT0": input_text,
    }

    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        output_text = result.get("OUTPUT0", "")

        if output_text == input_text:
            return True

    return False


def get_metrics(metrics_url: str, model_name: str = "identity") -> Tuple[int, int]:
    """
    Sends a request to the metrics endpoint and returns the following information:
    1. Status Code = Indicates whether interaction with Metrics endpoint was successful
    2. Inference Count = Indicates whether metrics data being returned is accurate
    """
    response = requests.get(metrics_url)
    inference_count = None

    if response.status_code == 200:
        inference_count = _extract_inference_count(response.text, model_name)
    return response.status_code, inference_count


def _extract_inference_count(metrics_data: str, model_name: str):
    """
    Helper function for _get_metrics that parses metrics_data (prometheus-friendly
    format) with regex to extract the inference count of model_name.
    """
    pattern = (
        rf'nv_inference_count\{{.*?model="{re.escape(model_name)}".*?\}}\s+([0-9.]+)'
    )
    match = re.search(pattern, metrics_data)
    if match:
        return int(float(match.group(1)))

    return None
