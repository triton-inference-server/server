import pytest

import tritonserver
from tritonfrontend import KServeHttp, KServeGrpc
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient

from contextlib import contextmanager
from typing import Union

from subprocess import call

# Making empty model repository
model_path = "/root/models"
call(f"mkdir -p {model_path}", shell=True)

# Starting Server Instance
server_options = tritonserver.Options(server_id="TestServer", model_repository=model_path, log_error=True, log_warn=True, log_info=True)
server = tritonserver.Server(server_options).start() 

# Starting http frontend
http_options = KServeHttp.Options()
http_service = KServeHttp.Server(server, http_options)
http_service.start()

# Starting grpc frontend
grpc_options = KServeGrpc.Options()
grpc_service = KServeGrpc.Server(server, grpc_options)
grpc_service.start()


# Context Manager to start and stop/close respective client
class MockClient: # Default http settings
    def __init__ (self, frontend_service = httpclient, url = "localhost:8000"):
        self.frontend_service = frontend_service
        self.url = url

    def __enter__(self):
        self.client = self.frontend_service.InferenceServerClient(url=self.url)
        return self.client

    def __exit__(self, exc_type, exc_value, traceback):
        self.client.close()
        self.frontend_service = None
        self.cleint = None



class TestKServeHttp:
    def test_server_ready(self):
        with MockClient(httpclient) as http_client:
            assert http_client.is_server_ready()
    
    def test_server_live(self):
        with MockClient(httpclient) as http_client:
            assert http_client.is_server_live()
    
    def test_load_model(model_name):
        pass

    def test_unload_model(model_name):
        pass

    def test_get_model_metadata(model_name):
        pass

    def test_get_model_config(model_name):
        pass

    

class TestKServeGrpc:
    def test_server_ready(self):
        with MockClient(grpcclient, url = "localhost:8001") as grpc_client:
            assert grpc_client.is_server_ready()
    
    def test_server_live(self):
        with MockClient(grpcclient, url = "localhost:8001") as grpc_client:
            assert grpc_client.is_server_live()
    
