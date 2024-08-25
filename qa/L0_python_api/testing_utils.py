import os
from typing import Union

import numpy as np
import tritonserver
from tritonfrontend import KServeGrpc, KServeHttp


class TestingUtils:
    @staticmethod
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

    @staticmethod
    def teardown_server(server: tritonserver.Server) -> None:
        server.stop()

    @staticmethod
    def setup_service(
        server: tritonserver.Server,
        frontend: Union[KServeHttp, KServeGrpc],
        options=None,
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
    def teardown_client(client) -> None:
        client.close()

    # Sends an inference to an identity model and verifies input == output.
    def send_and_test_inference_identity(frontend_client, url) -> bool:
        model_name = "identity"
        client = TestingUtils.setup_client(frontend_client, url)
        input_data = np.array(["testing"], dtype=object)

        # Create input and output objects
        inputs = [frontend_client.InferInput("INPUT0", input_data.shape, "BYTES")]
        outputs = [frontend_client.InferRequestedOutput("OUTPUT0")]
        # Set the data for the input tensor
        inputs[0].set_data_from_numpy(input_data)

        # Perform inference request
        results = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

        output_data = results.as_numpy("OUTPUT0")  # Gather output data

        TestingUtils.teardown_client(client)
        return input_data[0] == output_data[0].decode()
