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

import numpy as np
import pytest
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import tritonserver
from tritonfrontend import AlreadyExistsError, KServeGrpc, KServeHttp


# Context Manager to start and stop/close respective client
class Client:  # Default http settings
    def __init__(self, frontend_service=httpclient, url="0.0.0.0:8000"):
        self.frontend_service = frontend_service
        self.url = url

    def __enter__(self):
        self.client = self.frontend_service.InferenceServerClient(url=self.url)
        return self.client

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.close()
        self.frontend_service = None
        self.client = None


class TestKServeHttp:
    @pytest.fixture(scope="class")
    def server_service(self):
        model_path = "./test_model_repository"

        # Starting Server Instance
        server_options = tritonserver.Options(
            server_id="TestServer",
            model_repository=model_path,
            log_error=True,
            log_warn=True,
            log_info=True,
        )

        server = tritonserver.Server(server_options).start()

        http_options = KServeHttp.Options()
        http_service = KServeHttp.Server(server, http_options)
        http_service.start()

        yield server, http_service

        http_service.stop()
        server.stop()

    def test_server_ready(self, server_service):
        server, http_service = server_service
        with Client(httpclient, url="0.0.0.0:8000") as http_client:
            assert http_client.is_server_ready()

    def test_server_live(self, server_service):
        server, http_service = server_service
        with Client(httpclient, url="0.0.0.0:8000") as http_client:
            assert http_client.is_server_live()

    def test_already_exists_error(self, server_service):
        server, http_service = server_service
        # Should throw error because http service already started.
        with pytest.raises(AlreadyExistsError):
            http_service.start()

    def test_wrong_parameters(self, server_service):
        server, _ = server_service
        with pytest.raises(Exception):
            incorrect_http_options = KServeHttp.Options(port=-15)

    def test_invalid_server(self):
        with pytest.raises(tritonserver.InvalidArgumentError):
            invalid_server = tritonserver.Server(tritonserver.Options())
            custom_http_service = KServeHttp.Server(
                invalid_server, KServeHttp.Options()
            )

    def test_inference(self):
        model_name = "identity"
        with Client(httpclient, url="0.0.0.0:8000") as http_client:
            input_data = np.array([["testing"]], dtype=object)
            # Create input and output objects
            inputs = [httpclient.InferInput("INPUT0", input_data.shape, "BYTES")]
            outputs = [httpclient.InferRequestedOutput("OUTPUT0")]

            # Set the data for the input tensor
            inputs[0].set_data_from_numpy(input_data)

            results = http_client.infer(model_name, inputs=inputs, outputs=outputs)

            output_data = results.as_numpy("OUTPUT0")

            assert input_data[0][0] == output_data[0][0].decode()


class TestKServeGrpc:
    @pytest.fixture(scope="class")
    def server_service(self):
        # Making empty model repository
        model_path = "./test_model_repository"

        # Starting Server Instance
        server_options = tritonserver.Options(
            server_id="TestServer",
            model_repository=model_path,
            log_error=True,
            log_warn=True,
            log_info=True,
        )

        server = tritonserver.Server(server_options).start()

        grpc_options = KServeGrpc.Options()
        print(grpc_options)
        grpc_service = KServeGrpc.Server(server, grpc_options)
        grpc_service.start()

        yield server, grpc_service

        grpc_service.stop()
        server.stop()

    def test_server_ready(self, server_service):
        server, grpc_service = server_service
        with Client(grpcclient, url="0.0.0.0:8001") as grpc_client:
            assert grpc_client.is_server_ready()

    def test_server_live(self, server_service):
        server, grpc_service = server_service
        with Client(grpcclient, url="0.0.0.0:8001") as grpc_client:
            assert grpc_client.is_server_live()

    def test_already_exists_error(self, server_service):
        server, grpc_service = server_service
        # Should throw error because http service already started.
        with pytest.raises(AlreadyExistsError):
            grpc_service.start()

    def test_wrong_parameters(self, server_service):
        server, _ = server_service
        with pytest.raises(Exception):
            incorrect_grpc_options = KServeGrpc.Options(port=-15)

    def test_invalid_server(self):
        with pytest.raises(tritonserver.InvalidArgumentError):
            invalid_server = tritonserver.Server(tritonserver.Options())
            custom_grpc_service = KServeHttp.Server(
                invalid_server, KServeGrpc.Options()
            )

    def test_inference(self):
        model_name = "identity"
        with Client(grpcclient, url="0.0.0.0:8001") as grpc_client:
            input_data = np.array([["testing"]], dtype=object)
            # Create input and output objects
            inputs = [grpcclient.InferInput("INPUT0", input_data.shape, "BYTES")]
            outputs = [grpcclient.InferRequestedOutput("OUTPUT0")]

            # Set the data for the input tensor
            inputs[0].set_data_from_numpy(input_data)

            results = grpc_client.infer(model_name, inputs=inputs, outputs=outputs)

            output_data = results.as_numpy("OUTPUT0")

            assert input_data[0][0] == output_data[0][0].decode()
