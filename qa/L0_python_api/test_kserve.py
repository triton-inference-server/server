# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import time
from functools import partial

import numpy as np
import pytest
import testing_utils as utils
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import tritonserver
from tritonclient.utils import InferenceServerException
from tritonfrontend import KServeGrpc, KServeHttp, Metrics


class TestHttpOptions:
    def test_correct_http_parameters(self):
        KServeHttp.Options(
            address="0.0.0.1", port=8080, reuse_port=True, thread_count=16
        )

    def test_wrong_http_parameters(self):
        # Out of range
        with pytest.raises(Exception):
            KServeHttp.Options(port=-15)
        with pytest.raises(Exception):
            KServeHttp.Options(thread_count=0)

        # Wrong data type
        with pytest.raises(Exception):
            KServeHttp.Options(header_forward_pattern=10)


class TestGrpcOptions:
    def test_correct_grpc_parameters(self):
        KServeGrpc.Options(
            infer_compression_level=KServeGrpc.Grpc_compression_level.HIGH,
            reuse_port=True,
            infer_allocation_pool_size=12,
            http2_max_pings_without_data=10,
        )

    def test_wrong_grpc_parameters(self):
        # Out of Range
        with pytest.raises(Exception):
            KServeGrpc.Options(port=-5)
        with pytest.raises(Exception):
            KServeGrpc.Options(keepalive_timeout_ms=-20_000)
        with pytest.raises(Exception):
            KServeGrpc.Options(keepalive_time_ms=-1)
        with pytest.raises(Exception):
            KServeGrpc.Options(keepalive_timeout_ms=-1)
        with pytest.raises(Exception):
            KServeGrpc.Options(http2_max_pings_without_data=-1)
        with pytest.raises(Exception):
            KServeGrpc.Options(http2_min_recv_ping_interval_without_data_ms=-1)
        with pytest.raises(Exception):
            KServeGrpc.Options(http2_max_ping_strikes=-1)
        with pytest.raises(Exception):
            KServeGrpc.Options(max_connection_age_ms=-1)
        with pytest.raises(Exception):
            KServeGrpc.Options(max_connection_age_grace_ms=-1)

        # Wrong data type
        with pytest.raises(Exception):
            KServeGrpc.Options(infer_allocation_pool_size="big pool")
        with pytest.raises(Exception):
            KServeGrpc.Options(server_key=10)


class TestMetricsOptions:
    def test_correct_http_parameters(self):
        Metrics.Options(address="0.0.0.1", port=8080, thread_count=16)

    def test_wrong_http_parameters(self):
        # Out of range
        with pytest.raises(Exception):
            Metrics.Options(port=-15)
        with pytest.raises(Exception):
            Metrics.Options(thread_count=0)

        # Wrong data type
        with pytest.raises(Exception):
            Metrics.Options(thread_count="ten")


HTTP_ARGS = (KServeHttp, httpclient, "localhost:8000")  # Default HTTP args
GRPC_ARGS = (KServeGrpc, grpcclient, "localhost:8001")  # Default GRPC args
METRICS_ARGS = (Metrics, "localhost:8002")  # Default Metrics args


class TestKServe:
    @pytest.mark.xfail(run=False, reason="Python model may not load after gRPC import")
    @pytest.mark.parametrize("frontend, client_type, url", [HTTP_ARGS, GRPC_ARGS])
    def test_server_ready(self, frontend, client_type, url):
        server = utils.setup_server()
        service = utils.setup_service(server, frontend)
        client = utils.setup_client(client_type, url=url)

        assert client.is_server_ready()

        utils.teardown_client(client)
        utils.teardown_service(service)
        utils.teardown_server(server)

    @pytest.mark.xfail(run=False, reason="Python model may not load after gRPC import")
    @pytest.mark.parametrize("frontend", [HTTP_ARGS[0], GRPC_ARGS[0]])
    def test_service_double_start(self, frontend):
        server = utils.setup_server()
        # setup_service() performs service.start()
        service = utils.setup_service(server, frontend)

        with pytest.raises(
            tritonserver.AlreadyExistsError, match="server is already running."
        ):
            service.start()

        utils.teardown_server(server)
        utils.teardown_service(service)

    @pytest.mark.xfail(run=False, reason="Python model may not load after gRPC import")
    @pytest.mark.parametrize("frontend", [HTTP_ARGS[0], GRPC_ARGS[0]])
    def test_invalid_options(self, frontend):
        server = utils.setup_server()
        # Current flow is KServeHttp.Options or KServeGrpc.Options have to be
        # provided to ensure type and range validation occurs.
        with pytest.raises(
            tritonserver.InvalidArgumentError,
            match="Incorrect type for options. options argument must be of type",
        ):
            frontend(server, {"port": 8001})

        utils.teardown_server(server)

    @pytest.mark.xfail(run=False, reason="Python model may not load after gRPC import")
    @pytest.mark.parametrize("frontend", [HTTP_ARGS[0], GRPC_ARGS[0]])
    def test_server_service_order(self, frontend):
        server = utils.setup_server()
        service = utils.setup_service(server, frontend)

        utils.teardown_server(server)
        utils.teardown_service(service)

    @pytest.mark.xfail(run=False, reason="Python model may not load after gRPC import")
    @pytest.mark.parametrize("frontend, client_type", [HTTP_ARGS[:2], GRPC_ARGS[:2]])
    def test_service_custom_port(self, frontend, client_type):
        server = utils.setup_server()
        options = frontend.Options(port=8005)
        service = utils.setup_service(server, frontend, options)
        client = utils.setup_client(client_type, url="localhost:8005")

        # Confirms that service starts at port 8005
        client.is_server_ready()

        utils.teardown_client(client)
        utils.teardown_service(service)
        utils.teardown_server(server)

    @pytest.mark.xfail(run=False, reason="Python model may not load after gRPC import")
    @pytest.mark.parametrize("frontend, client_type, url", [HTTP_ARGS, GRPC_ARGS])
    def test_inference(self, frontend, client_type, url):
        server = utils.setup_server()
        service = utils.setup_service(server, frontend)

        # TODO: use common/test_infer
        assert utils.send_and_test_inference_identity(client_type, url=url)

        utils.teardown_service(service)
        utils.teardown_server(server)

    @pytest.mark.xfail(run=False, reason="Python model may not load after gRPC import")
    @pytest.mark.parametrize("frontend, client_type, url", [GRPC_ARGS])
    def test_streaming_inference(self, frontend, client_type, url):
        server = utils.setup_server()
        service = utils.setup_service(server, frontend)

        assert utils.send_and_test_stream_inference(client_type, url)

        utils.teardown_service(service)
        utils.teardown_server(server)

    @pytest.mark.xfail(run=False, reason="Python model may not load after gRPC import")
    @pytest.mark.parametrize("frontend, client_type, url", [HTTP_ARGS])
    def test_http_generate_inference(self, frontend, client_type, url):
        server = utils.setup_server()
        service = utils.setup_service(server, frontend)

        assert utils.send_and_test_generate_inference()

        utils.teardown_service(service)
        utils.teardown_server(server)

    @pytest.mark.xfail(run=False, reason="Python model may not load after gRPC import")
    @pytest.mark.parametrize("frontend, client_type, url", [HTTP_ARGS])
    def test_http_req_during_shutdown(self, frontend, client_type, url):
        server = utils.setup_server()
        http_service = utils.setup_service(server, frontend)
        http_client = httpclient.InferenceServerClient(url="localhost:8000")
        model_name = "delayed_identity"
        delay = 2  # seconds
        input_data0 = np.array([[delay]], dtype=np.float32)

        input0 = httpclient.InferInput("INPUT0", input_data0.shape, "FP32")
        input0.set_data_from_numpy(input_data0)

        inputs = [input0]
        outputs = [httpclient.InferRequestedOutput("OUTPUT0")]

        async_request = http_client.async_infer(
            model_name=model_name, inputs=inputs, outputs=outputs
        )
        # http_service.stop() does not use graceful shutdown
        utils.teardown_service(http_service)

        # So, inference request will fail as http endpoints have been stopped.
        with pytest.raises(
            InferenceServerException, match="failed to obtain inference response"
        ):
            async_request.get_result(block=True, timeout=delay)

        # http_client.close() calls join() to terminate pool of greenlets
        # However, due to an unsuccessful get_result(), async_request is still
        # an active thread. Hence, join stalls until greenlet timeouts.
        # Does not throw an exception, but displays error in logs.
        utils.teardown_client(http_client)

        # delayed_identity will still be an active model
        # Hence, server.stop() causes InternalError: Timeout.
        with pytest.raises(
            tritonserver.InternalError,
            match="Exit timeout expired. Exiting immediately.",
        ):
            utils.teardown_server(server)

    @pytest.mark.xfail(run=False, reason="Python model may not load after gRPC import")
    @pytest.mark.parametrize("frontend, client_type, url", [GRPC_ARGS])
    def test_grpc_req_during_shutdown(self, frontend, client_type, url):
        server = utils.setup_server()
        grpc_service = utils.setup_service(server, frontend)
        grpc_client = grpcclient.InferenceServerClient(url=url)
        user_data = []

        def callback(user_data, result, error):
            if error:
                user_data.append(error)
            else:
                user_data.append(result)

        model_name = "delayed_identity"
        delay = 2  # seconds

        input_data0 = np.array([[delay]], dtype=np.float32)
        input0 = client_type.InferInput("INPUT0", input_data0.shape, "FP32")
        input0.set_data_from_numpy(input_data0)

        inputs = [input0]
        outputs = [client_type.InferRequestedOutput("OUTPUT0")]

        grpc_client.async_infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            callback=partial(callback, user_data),
        )

        utils.teardown_service(grpc_service)

        time_out = delay + 1
        while (len(user_data) == 0) and time_out > 0:
            time_out = time_out - 1
            time.sleep(1)

        # Depending on when gRPC frontend shut down StatusCode can vary
        acceptable_failure_msgs = [
            "[StatusCode.CANCELLED] CANCELLED",
            "[StatusCode.UNAVAILABLE] failed to connect to all addresses",
        ]

        assert (
            len(user_data) == 1
            and isinstance(user_data[0], InferenceServerException)
            and any(
                failure_msg in str(user_data[0])
                for failure_msg in acceptable_failure_msgs
            )
        )

        utils.teardown_client(grpc_client)
        utils.teardown_server(server)

    @pytest.mark.xfail(run=False, reason="Python model may not load after gRPC import")
    @pytest.mark.parametrize("frontend, url", [METRICS_ARGS])
    def test_metrics_default_port(self, frontend, url):
        server = utils.setup_server()
        service = utils.setup_service(server, frontend)

        metrics_url = f"http://{url}/metrics"
        status_code, _ = utils.get_metrics(metrics_url)

        assert status_code == 200

        utils.teardown_service(service)
        utils.teardown_server(server)

    @pytest.mark.xfail(run=False, reason="Python model may not load after gRPC import")
    @pytest.mark.parametrize("frontend", [Metrics])
    def test_metrics_custom_port(self, frontend, port=8005):
        server = utils.setup_server()
        service = utils.setup_service(server, frontend, Metrics.Options(port=port))

        metrics_url = f"http://localhost:{port}/metrics"
        status_code, _ = utils.get_metrics(metrics_url)

        assert status_code == 200

        utils.teardown_service(service)
        utils.teardown_server(server)

    @pytest.mark.xfail(run=False, reason="Python model may not load after gRPC import")
    @pytest.mark.parametrize("frontend, url", [METRICS_ARGS])
    def test_metrics_update(self, frontend, url):
        # Setup Server, KServeGrpc, Metrics
        server = utils.setup_server()
        grpc_service = utils.setup_service(
            server, KServeGrpc
        )  # Needed to send inference request
        metrics_service = utils.setup_service(server, frontend)

        # Get Metrics and verify inference count == 0 before inference
        before_status_code, before_inference_count = utils.get_metrics(
            f"http://{url}/metrics"
        )
        assert before_status_code == 200 and before_inference_count == 0

        # Send 1 Inference Request with send_and_test_inference()
        assert utils.send_and_test_inference_identity(GRPC_ARGS[1], GRPC_ARGS[2])

        # Get Metrics and verify inference count == 1 after inference
        after_status_code, after_inference_count = utils.get_metrics(
            f"http://{url}/metrics"
        )
        assert after_status_code == 200 and after_inference_count == 1

        # Teardown Metrics, GrpcService, Server
        utils.teardown_service(grpc_service)
        utils.teardown_service(metrics_service)
        utils.teardown_server(server)

    # KNOWN ISSUE: CAUSES SEGFAULT
    # Created  [DLIS-7231] to address at future date
    # Once the server has been stopped, the underlying TRITONSERVER_Server instance
    # is deleted. However, the frontend does not know the server instance
    # is no longer valid.
    # @pytest.mark.xfail(run=False, reason="Python model may not load after gRPC import")
    # def test_inference_after_server_stop(self):
    #     server = utils.setup_server()
    #     http_service = utils.setup_service(server, KServeHttp)
    #     http_client = setup_client(httpclient, url="localhost:8000")

    #     teardown_server(server) # Server has been stopped

    #     model_name = "identity"
    #     input_data = np.array([["testing"]], dtype=object)
    #     # Create input and output objects
    #     inputs = [httpclient.InferInput("INPUT0", input_data.shape, "BYTES")]
    #     outputs = [httpclient.InferRequestedOutput("OUTPUT0")]

    #     # Set the data for the input tensor
    #     inputs[0].set_data_from_numpy(input_data)

    #     results = http_client.infer(model_name, inputs=inputs, outputs=outputs)

    #     utils.teardown_client(http_client)
    #     utils.teardown_service(http_service)
