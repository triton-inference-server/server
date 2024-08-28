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
from typing import Union

import numpy as np
import tritonserver
from tritonfrontend import KServeGrpc, KServeHttp


def setup_server(model_repository="test_model_repository") -> tritonserver.Server:
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
    frontend: Union[KServeHttp, KServeGrpc],
    options=None,
) -> Union[KServeHttp, KServeGrpc]:
    service = frontend.Server(server=server, options=options)
    service.start()
    return service


def teardown_service(service: Union[KServeHttp, KServeGrpc]) -> None:
    service.stop()


def setup_client(frontend_client, url: str):
    return frontend_client.InferenceServerClient(url=url)


def teardown_client(client) -> None:
    client.close()


# Sends an inference to test_model_repository/identity model and verifies input == output.
def send_and_test_inference_identity(frontend_client, url: str) -> bool:
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
