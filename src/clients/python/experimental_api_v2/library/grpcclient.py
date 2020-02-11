# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import grpc
from google.protobuf.json_format import MessageToJson
import rapidjson as json

from tensorrtserverV2.api import grpc_service_v2_pb2
from tensorrtserverV2.api import grpc_service_v2_pb2_grpc
from tensorrtserverV2.common import *


def raise_error_grpc(rpc_error):
    raise InferenceServerException(
        msg=rpc_error.details(),
        status=str(rpc_error.code()),
        debug_details=rpc_error.debug_error_string()) from None


class InferenceServerClient:
    """An InferenceServerClient object is used to perform any kind of
    communication with the InferenceServer using gRPC protocol.

    Parameters
    ----------
    url : str
        The inference server URL, e.g. 'localhost:8001'.     

    verbose : bool
        If True generate verbose output. Default value is False.

    Raises
        ------
        Exception
            If unable to create a client.

    """

    def __init__(self, url, verbose=False):
        # FixMe: Are any of the channel options worth exposing?
        # https://grpc.io/grpc/core/group__grpc__arg__keys.html
        self._channel = grpc.insecure_channel(url, options=None)
        self._client_stub = grpc_service_v2_pb2_grpc.GRPCInferenceServiceStub(
            self._channel)
        self._verbose = verbose

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        """Close the client. Any future calls to server
        will result in an Error.

        """
        self._channel.close()

    def is_server_live(self):
        """Contact the inference server and get liveness.

        Returns
        -------
        bool
            True if server is live, False if server is not live.

        Raises
        ------
        InferenceServerException
            If unable to get liveness.

        """
        try:
            self._request = grpc_service_v2_pb2.ServerLiveRequest()
            self._response = self._client_stub.ServerLive(self._request)
            return self._response.live
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def is_server_ready(self):
        """Contact the inference server and get readiness.

        Returns
        -------
        bool
            True if server is ready, False if server is not ready.

        Raises
        ------
        InferenceServerException
            If unable to get readiness.

        """
        try:
            self._request = grpc_service_v2_pb2.ServerReadyRequest()
            self._response = self._client_stub.ServerReady(self._request)
            return self._response.ready
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def is_model_ready(self, model_name, model_version=-1):
        """Contact the inference server and get the readiness of specified model.

         Parameters
        ----------
        model_name: str
            The name of the model to check for readiness.

        model_version: int
            The version of the model to check for readiness. If -1 is given the 
            server will choose a version based on the model and internal policy.


        Returns
        -------
        bool
            True if the model is ready, false if not ready.

        Raises
        ------
        InferenceServerException
            If unable to get model readiness.

        """
        try:
            self._request = grpc_service_v2_pb2.ModelReadyRequest(
                name=model_name, version=model_version)
            self._response = self._client_stub.ModelReady(self._request)
            return self._response.ready
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_server_metadata(self):
        """Contact the inference server and get its metadata.

        Returns
        -------
        dict
            The JSON object holding the metadata

        Raises
        ------
        InferenceServerException
            If unable to get server metadata.

        """
        try:
            self._request = grpc_service_v2_pb2.ServerMetadataRequest()
            self._response = self._client_stub.ServerMetadata(self._request)
            return json.loads(MessageToJson(self._response))
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_model_metadata(self, model_name, model_version=-1):
        """Contact the inference server and get the metadata for specified model.

        Parameters
        ----------
        model_name: str
            The name of the model
        
        model_version: int
            The version of the model to get metadata. If -1 is given the 
            server will choose a version based on the model and internal policy.

        Returns
        -------
        dict
            The JSON object holding the model metadata

        Raises
        ------
        InferenceServerException
            If unable to get model metadata.

        """
        try:
            self._request = grpc_service_v2_pb2.ModelMetadataRequest(
                name=model_name, version=model_version)
            self._response = self._client_stub.ModelMetadata(self._request)
            return json.loads(MessageToJson(self._response))
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_model_config(self, model_name, model_version=-1):
        """Contact the inference server and get the configuration for specified model.

        Parameters
        ----------
        model_name: str
            The name of the model
        
        model_version: int
            The version of the model to get configuration. If -1 is given the 
            server will choose a version based on the model and internal policy.

        Returns
        -------
        dict
            The JSON object holding the model configuration

        Raises
        ------
        InferenceServerException
            If unable to get model configuration.

        """
        try:
            self._request = grpc_service_v2_pb2.ModelConfigRequest(
                name=model_name, version=model_version)
            self._response = self._client_stub.ModelConfig(self._request)
            return json.loads(MessageToJson(self._response))
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def load_model(self, model_name):
        """Request the inference server to load or reload specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to be loaded.

        Raises
        ------
        InferenceServerException
            If unable to load the model.

        """
        raise_error("Not implemented yet")

    def unload_model(self, model_name):
        """Request the inference server to unload specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to be unloaded.

        Raises
        ------
        InferenceServerException
            If unable to unload the model.

        """
        raise_error("Not implemented yet")
