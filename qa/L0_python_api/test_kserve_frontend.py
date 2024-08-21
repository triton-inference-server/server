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
import pytest
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import tritonserver
from tritonfrontend import (
    AlreadyExistsError,
    InvalidArgumentError,
    KServeGrpc,
    KServeHttp,
)


class TestingUtils:
    @staticmethod
    def setup_server(model_repository="test_model_repository") -> tritonserver:
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

        server = tritonserver.Server(server_options).start(wait_until_ready=True)
        return server

    @staticmethod
    def teardown_server(server: tritonserver) -> None:
        server.stop()

    @staticmethod
    def setup_service(
        server: tritonserver, frontend: Union[KServeHttp, KServeGrpc], options=None
    ) -> Union[KServeHttp, KServeGrpc]:
        service = frontend.Server(server=server, options=options)
        service.start()
        return service

    @staticmethod
    def teardown_service(service: Union[KServeHttp, KServeGrpc]) -> None:
        service.stop()

    @staticmethod
    def setup_client(frontend_client, url: str):
        return frontend_client.InferenceServerClient(url=url)

    @staticmethod
    def teardown_client(client):
        client.close()


class TestKServeHttp:
    def test_server_ready(self):
        server = TestingUtils.setup_server()
        http_service = TestingUtils.setup_service(server, KServeHttp)
        http_client = TestingUtils.setup_client(httpclient, url="localhost:8000")

        assert http_client.is_server_ready()

        TestingUtils.teardown_client(http_client)
        TestingUtils.teardown_service(http_service)
        TestingUtils.teardown_server(server)

    def test_already_exists_error(self):
        server = TestingUtils.setup_server()
        http_service = TestingUtils.setup_service(server, KServeHttp)

        with pytest.raises(AlreadyExistsError):
            http_service.start()

        TestingUtils.teardown_server(server)
        TestingUtils.teardown_service(http_service)

    def test_wrong_parameters(self):
        with pytest.raises(Exception):
            KServeHttp.Options(port=-15)
        with pytest.raises(Exception):
            KServeHttp.Options(thread_count=-5)

    def test_invalid_options(self):
        server = TestingUtils.setup_server()

        with pytest.raises(InvalidArgumentError):
            custom_grpc_service = KServeHttp.Server(server, {"port": 8001})

        TestingUtils.teardown_server(server)

    def test_service_custom_port(self):
        server = TestingUtils.setup_server()
        http_options = KServeHttp.Options(port=8005)
        http_service = TestingUtils.setup_service(server, KServeHttp, http_options)
        http_client = TestingUtils.setup_client(httpclient, url="localhost:8005")

        http_client.is_server_ready()

        TestingUtils.teardown_client(http_client)
        TestingUtils.teardown_service(http_service)
        TestingUtils.teardown_server(server)

    def test_inference(self):
        server = TestingUtils.setup_server()
        http_service = TestingUtils.setup_service(server, KServeHttp)
        http_client = TestingUtils.setup_client(httpclient, url="localhost:8000")

        model_name = "identity"
        input_data = np.array([["testing"]], dtype=object)
        # Create input and output objects
        inputs = [httpclient.InferInput("INPUT0", input_data.shape, "BYTES")]
        outputs = [httpclient.InferRequestedOutput("OUTPUT0")]

        # Set the data for the input tensor
        inputs[0].set_data_from_numpy(input_data)

        results = http_client.infer(model_name, inputs=inputs, outputs=outputs)

        output_data = results.as_numpy("OUTPUT0")

        assert input_data[0][0] == output_data[0][0].decode()

        TestingUtils.teardown_client(http_client)
        TestingUtils.teardown_service(http_service)
        TestingUtils.teardown_server(server)

    def test_server_service_order(self):
        server = TestingUtils.setup_server()
        http_service = TestingUtils.setup_service(server, KServeHttp)

        server.stop()
        http_service.stop()

    # KNOWN ISSUE: CAUSES SEGFAULT
    # Created  [DLIS-7231] to address at future date
    # def test_inference_after_server_stop(self):
    #     server = TestingUtils.setup_server()
    #     http_service = TestingUtils.setup_service(server, KServeHttp)
    #     http_client = TestingUtils.setup_client(httpclient, url="localhost:8000")

    #     TestingUtils.teardown_server(server) # Server has been stopped

    #     model_name = "identity"
    #     input_data = np.array([["testing"]], dtype=object)
    #     # Create input and output objects
    #     inputs = [httpclient.InferInput("INPUT0", input_data.shape, "BYTES")]
    #     outputs = [httpclient.InferRequestedOutput("OUTPUT0")]

    #     # Set the data for the input tensor
    #     inputs[0].set_data_from_numpy(input_data)

    #     results = http_client.infer(model_name, inputs=inputs, outputs=outputs)

    #     TestingUtils.teardown_client(http_client)
    #     TestingUtils.teardown_service(http_service)


class TestKServeGrpc:
    def test_server_ready(self):
        server = TestingUtils.setup_server()
        grpc_service = TestingUtils.setup_service(server, KServeGrpc)
        grpc_client = TestingUtils.setup_client(grpcclient, url="localhost:8001")

        assert grpc_client.is_server_ready()

        TestingUtils.teardown_client(grpc_client)
        TestingUtils.teardown_service(grpc_service)
        TestingUtils.teardown_server(server)

    def test_already_exists_error(self):
        server = TestingUtils.setup_server()
        grpc_service = TestingUtils.setup_service(server, KServeGrpc)

        with pytest.raises(AlreadyExistsError):
            grpc_service.start()

        TestingUtils.teardown_server(server)
        TestingUtils.teardown_service(grpc_service)

    def test_wrong_parameters(self):
        with pytest.raises(Exception):
            KServeGrpc.Options(port=-15)

    def test_invalid_options(self):
        server = TestingUtils.setup_server()

        with pytest.raises(InvalidArgumentError):
            custom_grpc_service = KServeGrpc.Server(server, {"port": 8001})

        TestingUtils.teardown_server(server)

    def test_service_custom_port(self):
        server = TestingUtils.setup_server()
        grpc_options = KServeGrpc.Options(port=8005)
        grpc_service = TestingUtils.setup_service(server, KServeGrpc, grpc_options)
        grpc_client = TestingUtils.setup_client(grpcclient, url="localhost:8005")

        assert grpc_client.is_server_ready()

        TestingUtils.teardown_client(grpc_client)
        TestingUtils.teardown_service(grpc_service)
        TestingUtils.teardown_server(server)

    def test_inference(self):
        server = TestingUtils.setup_server()
        grpc_service = TestingUtils.setup_service(server, KServeGrpc)
        grpc_client = TestingUtils.setup_client(grpcclient, url="localhost:8001")

        model_name = "identity"
        input_data = np.array([["testing"]], dtype=object)
        # Create input and output objects
        inputs = [grpcclient.InferInput("INPUT0", input_data.shape, "BYTES")]
        outputs = [grpcclient.InferRequestedOutput("OUTPUT0")]

        # Set the data for the input tensor
        inputs[0].set_data_from_numpy(input_data)

        results = grpc_client.infer(model_name, inputs=inputs, outputs=outputs)

        output_data = results.as_numpy("OUTPUT0")

        assert input_data[0][0] == output_data[0][0].decode()

        TestingUtils.teardown_client(grpc_client)
        TestingUtils.teardown_service(grpc_service)
        TestingUtils.teardown_server(server)

    def test_server_service_order(self):
        server = TestingUtils.setup_server()
        grpc_service = TestingUtils.setup_service(server, KServeGrpc)

        server.stop()
        grpc_service.stop()  # Should have graceful exitG
