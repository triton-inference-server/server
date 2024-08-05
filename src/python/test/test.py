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

from subprocess import call

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import tritonserver
from tritonfrontend import KServeGrpc, KServeHttp

# Making empty model repository
model_path = "/root/models"
call(f"mkdir -p {model_path}", shell=True)

# Starting Server Instance
server_options = tritonserver.Options(
    server_id="TestServer",
    model_repository=model_path,
    log_error=True,
    log_warn=True,
    log_info=True,
)
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
class ClientHelper:  # Default http settings
    def __init__(self, frontend_service=httpclient, url="localhost:8000"):
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
    def test_server_ready(self):
        with ClientHelper(httpclient) as http_client:
            assert http_client.is_server_ready()

    def test_server_live(self):
        with ClientHelper(httpclient) as http_client:
            assert http_client.is_server_live()

    def test_load_model(self, model_name):
        pass

    def test_unload_model(self, model_name):
        pass

    def test_get_model_metadata(self, model_name):
        pass

    def test_get_model_config(self, model_name):
        pass


class TestKServeGrpc:
    def test_server_ready(self):
        with ClientHelper(grpcclient, url="localhost:8001") as grpc_client:
            assert grpc_client.is_server_ready()

    def test_server_live(self):
        with ClientHelper(grpcclient, url="localhost:8001") as grpc_client:
            assert grpc_client.is_server_live()
