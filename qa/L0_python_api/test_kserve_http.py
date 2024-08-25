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
import time
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


class TestHttpOptions:
    def test_correct_http_parameters(self):
        http_options = KServeHttp.Options(
            address="0.0.0.1", port="8080", reuse_port=True, thread_count=16
        )

    def test_wrong_http_parameters(self):
        # Out of range
        with pytest.raises(Exception):
            KServeHttp.Options(port=-15)
        with pytest.raises(Exception):
            KServeHttp.Options(thread_count=-5)

        # Wrong data type
        with pytest.raises(Exception):
            KServeHttp.Options(header_forward_pattern=10)


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
        # setup_service() performs http_service.start()
        http_service = TestingUtils.setup_service(server, KServeHttp)

        with pytest.raises(AlreadyExistsError):
            http_service.start()

        TestingUtils.teardown_server(server)
        TestingUtils.teardown_service(http_service)

    def test_invalid_options(self):
        server = TestingUtils.setup_server()

        with pytest.raises(InvalidArgumentError):
            custom_grpc_service = KServeHttp.Server(server, {"port": 8001})

        TestingUtils.teardown_server(server)

    def test_server_service_order(self):
        server = TestingUtils.setup_server()
        http_service = TestingUtils.setup_service(server, KServeHttp)

        server.stop()
        http_service.stop()

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

        assert TestingUtils.send_and_test_inference_identity(
            httpclient, "localhost:8000"
        )

        TestingUtils.teardown_service(http_service)
        TestingUtils.teardown_server(server)

    def test_req_during_shutdown(self):
        server = TestingUtils.setup_server()
        http_service = TestingUtils.setup_service(server, KServeHttp)
        http_client = httpclient.InferenceServerClient(url="localhost:8000")
        model_name = "delayed_identity"
        delay = 10  # seconds
        input_data0 = np.array([[delay]], dtype=np.float32)

        input0 = httpclient.InferInput("INPUT0", input_data0.shape, "FP32")
        input0.set_data_from_numpy(input_data0)

        inputs = [input0]
        outputs = [httpclient.InferRequestedOutput("OUTPUT0")]

        async_request = http_client.async_infer(
            model_name=model_name, inputs=inputs, outputs=outputs
        )
        print("Beginning to sleep")
        time.sleep(3)
        print("Slept for 3 seconds")
        TestingUtils.teardown_service(http_service)
        # print(http_client.is_server_ready())
        response = async_request.get_result()
        output_data0 = response.as_numpy("OUTPUT0")

        assert input_data0[0][0] == output_data0[0][0]

        TestingUtils.teardown_client(server)
        TestingUtils.teardown_server(server)

    # KNOWN ISSUE: CAUSES SEGFAULT
    # Created  [DLIS-7231] to address at future date
    # Once the server has been stopped, the underlying TRITONSERVER_Server instance
    # is deleted. However, the frontend does not know the server instance
    # is no longer valid.
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


# def test_long_inference()
