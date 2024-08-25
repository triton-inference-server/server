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
from testing_utils import TestingUtils
from tritonfrontend import (
    AlreadyExistsError,
    InvalidArgumentError,
    KServeGrpc,
    KServeHttp,
)


class TestGrpcOptions:
    def test_correct_grpc_parameters(self):
        grpc_options = KServeGrpc.Options(
            infer_compression_level=KServeGrpc.Grpc_compression_level.HIGH,
            reuse_port=True,
            infer_allocation_pool_size=12,
            http2_max_pings_without_data=10,
        )

    def test_wrong_grpc_parameters(self):
        with pytest.raises(Exception):
            KServeGrpc.Options(port="testing")
        with pytest.raises(Exception):
            KServeGrpc.Options(infer_allocation_pool_size="big pool")
        with pytest.raises(Exception):
            KServeGrpc.Options(server_key=10)


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
        # setup_service() performs grpc_service.start()
        grpc_service = TestingUtils.setup_service(server, KServeGrpc)

        with pytest.raises(AlreadyExistsError):
            grpc_service.start()

        TestingUtils.teardown_server(server)
        TestingUtils.teardown_service(grpc_service)

    def test_invalid_options(self):
        server = TestingUtils.setup_server()

        with pytest.raises(InvalidArgumentError):
            custom_grpc_service = KServeGrpc.Server(server, {"port": 8001})

        TestingUtils.teardown_server(server)

    def test_server_service_order(self):
        server = TestingUtils.setup_server()
        grpc_service = TestingUtils.setup_service(server, KServeGrpc)

        TestingUtils.teardown_server(server)
        TestingUtils.teardown_service(grpc_service)

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

        assert TestingUtils.send_and_test_inference_identity(
            grpcclient, "localhost:8001"
        )

        TestingUtils.teardown_service(grpc_service)
        TestingUtils.teardown_server(server)
