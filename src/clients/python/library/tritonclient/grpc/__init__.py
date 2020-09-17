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

try:
    from google.protobuf.json_format import MessageToJson
    import grpc
    import base64
    import numpy as np
    import rapidjson as json
    import threading
    import queue
    import struct
except ModuleNotFoundError as error:
    raise RuntimeError(
        'The installation does not include grpc support. Specify \'grpc\' or \'all\' while installing the tritonclient package to include the support'
    ) from error

from tritonclient.grpc import model_config_pb2
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc

from tritonclient.utils import *

# Should be kept consistent with the value specified in
# src/core/constants.h, which specifies MAX_GRPC_MESSAGE_SIZE
# as INT32_MAX.
MAX_GRPC_MESSAGE_SIZE = 2**(struct.Struct('i').size * 8 - 1) - 1


def get_error_grpc(rpc_error):
    return InferenceServerException(
        msg=rpc_error.details(),
        status=str(rpc_error.code()),
        debug_details=rpc_error.debug_error_string())


def raise_error_grpc(rpc_error):
    raise get_error_grpc(rpc_error) from None


def _get_inference_request(model_name, inputs, model_version, request_id,
                           outputs, sequence_id, sequence_start, sequence_end,
                           priority, timeout):
    request = service_pb2.ModelInferRequest()
    request.model_name = model_name
    request.model_version = model_version
    if request_id != "":
        request.id = request_id
    for infer_input in inputs:
        request.inputs.extend([infer_input._get_tensor()])
        if infer_input._get_content() is not None:
            request.raw_input_contents.extend([infer_input._get_content()])
    if outputs is not None:
        for infer_output in outputs:
            request.outputs.extend([infer_output._get_tensor()])
    if sequence_id != 0:
        request.parameters['sequence_id'].int64_param = sequence_id
        request.parameters['sequence_start'].bool_param = sequence_start
        request.parameters['sequence_end'].bool_param = sequence_end
    if priority != 0:
        request.parameters['priority'].int64_param = priority
    if timeout is not None:
        request.parameters['timeout'].int64_param = timeout
    return request


class InferenceServerClient:
    """An InferenceServerClient object is used to perform any kind of
    communication with the InferenceServer using gRPC protocol. Most
    of the methods are thread-safe except start_stream, stop_stream
    and async_stream_infer. Accessing a client stream with different
    threads will cause undefined behavior.

    Parameters
    ----------
    url : str
        The inference server URL, e.g. 'localhost:8001'.

    verbose : bool
        If True generate verbose output. Default value is False.

    ssl : bool
        If True use SSL encrypted secure channel. Default is False.

    root_certificates : str
        File holding the PEM-encoded root certificates as a byte
        string, or None to retrieve them from a default location
        chosen by gRPC runtime. The option is ignored if `ssl`
        is False. Default is None.

    private_key : str
        File holding the PEM-encoded private key as a byte string,
        or None if no private key should be used. The option is
        ignored if `ssl` is False. Default is None.

    certificate_chain : str
        File holding PEM-encoded certificate chain as a byte string
        to use or None if no certificate chain should be used. The
        option is ignored if `ssl` is False. Default is None.

    Raises
    ------
    Exception
        If unable to create a client.

    """

    def __init__(self,
                 url,
                 verbose=False,
                 ssl=False,
                 root_certificates=None,
                 private_key=None,
                 certificate_chain=None):
        # FixMe: Are any of the channel options worth exposing?
        # https://grpc.io/grpc/core/group__grpc__arg__keys.html
        channel_opt = [('grpc.max_send_message_length', MAX_GRPC_MESSAGE_SIZE),
                       ('grpc.max_receive_message_length',
                        MAX_GRPC_MESSAGE_SIZE)]
        if ssl:
            rc_bytes = pk_bytes = cc_bytes = None
            if root_certificates is not None:
                with open(root_certificates, 'rb') as rc_fs:
                    rc_bytes = rc_fs.read()
            if private_key is not None:
                with open(private_key, 'rb') as pk_fs:
                    pk_bytes = pk_fs.read()
            if certificate_chain is not None:
                with open(certificate_chain, 'rb') as cc_fs:
                    cc_bytes = cc_fs.read()
            creds = grpc.ssl_channel_credentials(root_certificates=rc_bytes,
                                                 private_key=pk_bytes,
                                                 certificate_chain=cc_bytes)
            self._channel = grpc.secure_channel(url, creds, options=channel_opt)
        else:
            self._channel = grpc.insecure_channel(url, options=channel_opt)
        self._client_stub = service_pb2_grpc.GRPCInferenceServiceStub(
            self._channel)
        self._verbose = verbose
        self._stream = None

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
        self.stop_stream()
        self._channel.close()

    def is_server_live(self, headers=None):
        """Contact the inference server and get liveness.

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Returns
        -------
        bool
            True if server is live, False if server is not live.

        Raises
        ------
        InferenceServerException
            If unable to get liveness.

        """
        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            request = service_pb2.ServerLiveRequest()
            if self._verbose:
                print("is_server_live, metadata {}\n{}".format(
                    metadata, request))
            response = self._client_stub.ServerLive(request=request,
                                                    metadata=metadata)
            if self._verbose:
                print(response)
            return response.live
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def is_server_ready(self, headers=None):
        """Contact the inference server and get readiness.

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Returns
        -------
        bool
            True if server is ready, False if server is not ready.

        Raises
        ------
        InferenceServerException
            If unable to get readiness.

        """
        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            request = service_pb2.ServerReadyRequest()
            if self._verbose:
                print("is_server_ready, metadata {}\n{}".format(
                    metadata, request))
            response = self._client_stub.ServerReady(request=request,
                                                     metadata=metadata)
            if self._verbose:
                print(response)
            return response.ready
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def is_model_ready(self, model_name, model_version="", headers=None):
        """Contact the inference server and get the readiness of specified model.

        Parameters
        ----------
        model_name: str
            The name of the model to check for readiness.
        model_version: str
            The version of the model to check for readiness. The default value
            is an empty string which means then the server will choose a version
            based on the model and internal policy.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Returns
        -------
        bool
            True if the model is ready, False if not ready.

        Raises
        ------
        InferenceServerException
            If unable to get model readiness.

        """
        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            if type(model_version) != str:
                raise_error("model version must be a string")
            request = service_pb2.ModelReadyRequest(name=model_name,
                                                    version=model_version)
            if self._verbose:
                print("is_model_ready, metadata {}\n{}".format(
                    metadata, request))
            response = self._client_stub.ModelReady(request=request,
                                                    metadata=metadata)
            if self._verbose:
                print(response)
            return response.ready
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_server_metadata(self, headers=None, as_json=False):
        """Contact the inference server and get its metadata.

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns server metadata as a json dict,
            otherwise as a protobuf message. Default value is
            False. The returned json is generated from the protobuf
            message using MessageToJson and as a result int64 values
            are represented as string. It is the caller's
            responsibility to convert these strings back to int64
            values as necessary.

        Returns
        -------
        dict or protobuf message
            The JSON dict or ServerMetadataResponse message
            holding the metadata.

        Raises
        ------
        InferenceServerException
            If unable to get server metadata.

        """
        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            request = service_pb2.ServerMetadataRequest()
            if self._verbose:
                print("get_server_metadata, metadata {}\n{}".format(
                    metadata, request))
            response = self._client_stub.ServerMetadata(request=request,
                                                        metadata=metadata)
            if self._verbose:
                print(response)
            if as_json:
                return json.loads(
                    MessageToJson(response, preserving_proto_field_name=True))
            else:
                return response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_model_metadata(self,
                           model_name,
                           model_version="",
                           headers=None,
                           as_json=False):
        """Contact the inference server and get the metadata for specified model.

        Parameters
        ----------
        model_name: str
            The name of the model
        model_version: str
            The version of the model to get metadata. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns model metadata as a json dict,
            otherwise as a protobuf message. Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.

        Returns
        -------
        dict or protobuf message
            The JSON dict or ModelMetadataResponse message holding
            the metadata.

        Raises
        ------
        InferenceServerException
            If unable to get model metadata.

        """
        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            if type(model_version) != str:
                raise_error("model version must be a string")
            request = service_pb2.ModelMetadataRequest(name=model_name,
                                                       version=model_version)
            if self._verbose:
                print("get_model_metadata, metadata {}\n{}".format(
                    metadata, request))
            response = self._client_stub.ModelMetadata(request=request,
                                                       metadata=metadata)
            if self._verbose:
                print(response)
            if as_json:
                return json.loads(
                    MessageToJson(response, preserving_proto_field_name=True))
            else:
                return response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_model_config(self,
                         model_name,
                         model_version="",
                         headers=None,
                         as_json=False):
        """Contact the inference server and get the configuration for specified model.

        Parameters
        ----------
        model_name: str
            The name of the model
        model_version: str
            The version of the model to get configuration. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns configuration as a json dict, otherwise
            as a protobuf message. Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.

        Returns
        -------
        dict or protobuf message
            The JSON dict or ModelConfigResponse message holding
            the metadata.

        Raises
        ------
        InferenceServerException
            If unable to get model configuration.

        """
        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            if type(model_version) != str:
                raise_error("model version must be a string")
            request = service_pb2.ModelConfigRequest(name=model_name,
                                                     version=model_version)
            if self._verbose:
                print("get_model_config, metadata {}\n{}".format(
                    metadata, request))
            response = self._client_stub.ModelConfig(request=request,
                                                     metadata=metadata)
            if self._verbose:
                print(response)
            if as_json:
                return json.loads(
                    MessageToJson(response, preserving_proto_field_name=True))
            else:
                return response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_model_repository_index(self, headers=None, as_json=False):
        """Get the index of model repository contents

        Parameters
        ----------
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns model repository index
            as a json dict, otherwise as a protobuf message.
            Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.

        Returns
        -------
        dict or protobuf message
            The JSON dict or RepositoryIndexResponse message holding
            the model repository index.

        """
        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            request = service_pb2.RepositoryIndexRequest()
            if self._verbose:
                print("get_model_repository_index, metadata {}\n{}".format(
                    metadata, request))
            response = self._client_stub.RepositoryIndex(request=request,
                                                         metadata=metadata)
            if self._verbose:
                print(response)
            if as_json:
                return json.loads(
                    MessageToJson(response, preserving_proto_field_name=True))
            else:
                return response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def load_model(self, model_name, headers=None):
        """Request the inference server to load or reload specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to be loaded.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Raises
        ------
        InferenceServerException
            If unable to load the model.

        """
        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            request = service_pb2.RepositoryModelLoadRequest(
                model_name=model_name)
            if self._verbose:
                print("load_model, metadata {}\n{}".format(metadata, request))
            self._client_stub.RepositoryModelLoad(request=request,
                                                  metadata=metadata)
            if self._verbose:
                print("Loaded model '{}'".format(model_name))
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def unload_model(self, model_name, headers=None):
        """Request the inference server to unload specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to be unloaded.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Raises
        ------
        InferenceServerException
            If unable to unload the model.

        """
        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            request = service_pb2.RepositoryModelUnloadRequest(
                model_name=model_name)
            if self._verbose:
                print("unload_model, metadata {}\n{}".format(metadata, request))
            self._client_stub.RepositoryModelUnload(request=request,
                                                    metadata=metadata)
            if self._verbose:
                print("Unloaded model '{}'".format(model_name))
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_inference_statistics(self,
                                 model_name="",
                                 model_version="",
                                 headers=None,
                                 as_json=False):
        """Get the inference statistics for the specified model name and
        version.

        Parameters
        ----------
        model_name : str
            The name of the model to get statistics. The default value is
            an empty string, which means statistics of all models will
            be returned.
        model_version: str
            The version of the model to get inference statistics. The
            default value is an empty string which means then the server
            will return the statistics of all available model versions.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns inference statistics
            as a json dict, otherwise as a protobuf message.
            Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.

        Raises
        ------
        InferenceServerException
            If unable to get the model inference statistics.

        """
        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            if type(model_version) != str:
                raise_error("model version must be a string")
            request = service_pb2.ModelStatisticsRequest(name=model_name,
                                                         version=model_version)
            if self._verbose:
                print("get_inference_statistics, metadata {}\n{}".format(
                    metadata, request))
            response = self._client_stub.ModelStatistics(request=request,
                                                         metadata=metadata)
            if self._verbose:
                print(response)
            if as_json:
                return json.loads(
                    MessageToJson(response, preserving_proto_field_name=True))
            else:
                return response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_system_shared_memory_status(self,
                                        region_name="",
                                        headers=None,
                                        as_json=False):
        """Request system shared memory status from the server.

        Parameters
        ----------
        region_name : str
            The name of the region to query status. The default
            value is an empty string, which means that the status
            of all active system shared memory will be returned.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns system shared memory status as a json
            dict, otherwise as a protobuf message. Default value is
            False.  The returned json is generated from the protobuf
            message using MessageToJson and as a result int64 values
            are represented as string. It is the caller's
            responsibility to convert these strings back to int64
            values as necessary.

        Returns
        -------
        dict or protobuf message
            The JSON dict or SystemSharedMemoryStatusResponse message holding
            the system shared memory status.

        Raises
        ------
        InferenceServerException
            If unable to get the status of specified shared memory.

        """
        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            request = service_pb2.SystemSharedMemoryStatusRequest(
                name=region_name)
            if self._verbose:
                print("get_system_shared_memory_status, metadata {}\n{}".format(
                    metadata, request))
            response = self._client_stub.SystemSharedMemoryStatus(
                request=request, metadata=metadata)
            if self._verbose:
                print(response)
            if as_json:
                return json.loads(
                    MessageToJson(response, preserving_proto_field_name=True))
            else:
                return response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def register_system_shared_memory(self,
                                      name,
                                      key,
                                      byte_size,
                                      offset=0,
                                      headers=None):
        """Request the server to register a system shared memory with the
        following specification.

        Parameters
        ----------
        name : str
            The name of the region to register.
        key : str
            The key of the underlying memory object that contains the
            system shared memory region.
        byte_size : int
            The size of the system shared memory region, in bytes.
        offset : int
            Offset, in bytes, within the underlying memory object to
            the start of the system shared memory region. The default
            value is zero.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Raises
        ------
        InferenceServerException
            If unable to register the specified system shared memory.

        """
        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            request = service_pb2.SystemSharedMemoryRegisterRequest(
                name=name, key=key, offset=offset, byte_size=byte_size)
            if self._verbose:
                print("register_system_shared_memory, metadata {}\n{}".format(
                    metadata, request))
            self._client_stub.SystemSharedMemoryRegister(request=request,
                                                         metadata=metadata)
            if self._verbose:
                print("Registered system shared memory with name '{}'".format(
                    name))
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def unregister_system_shared_memory(self, name="", headers=None):
        """Request the server to unregister a system shared memory with the
        specified name.

        Parameters
        ----------
        name : str
            The name of the region to unregister. The default value is empty
            string which means all the system shared memory regions will be
            unregistered.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Raises
        ------
        InferenceServerException
            If unable to unregister the specified system shared memory region.

        """
        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            request = service_pb2.SystemSharedMemoryUnregisterRequest(name=name)
            if self._verbose:
                print("unregister_system_shared_memory, metadata {}\n{}".format(
                    metadata, request))
            self._client_stub.SystemSharedMemoryUnregister(request=request,
                                                           metadata=metadata)
            if self._verbose:
                if name is not "":
                    print("Unregistered system shared memory with name '{}'".
                          format(name))
                else:
                    print("Unregistered all system shared memory regions")
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_cuda_shared_memory_status(self,
                                      region_name="",
                                      headers=None,
                                      as_json=False):
        """Request cuda shared memory status from the server.

        Parameters
        ----------
        region_name : str
            The name of the region to query status. The default
            value is an empty string, which means that the status
            of all active cuda shared memory will be returned.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.
        as_json : bool
            If True then returns cuda shared memory status as a json
            dict, otherwise as a protobuf message. Default value is
            False.  The returned json is generated from the protobuf
            message using MessageToJson and as a result int64 values
            are represented as string. It is the caller's
            responsibility to convert these strings back to int64
            values as necessary.

        Returns
        -------
        dict or protobuf message
            The JSON dict or CudaSharedMemoryStatusResponse message holding
            the cuda shared memory status.

        Raises
        ------
        InferenceServerException
            If unable to get the status of specified shared memory.

        """

        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            request = service_pb2.CudaSharedMemoryStatusRequest(
                name=region_name)
            if self._verbose:
                print("get_cuda_shared_memory_status, metadata {}\n{}".format(
                    metadata, request))
            response = self._client_stub.CudaSharedMemoryStatus(
                request=request, metadata=metadata)
            if self._verbose:
                print(response)
            if as_json:
                return json.loads(
                    MessageToJson(response, preserving_proto_field_name=True))
            else:
                return response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def register_cuda_shared_memory(self,
                                    name,
                                    raw_handle,
                                    device_id,
                                    byte_size,
                                    headers=None):
        """Request the server to register a system shared memory with the
        following specification.

        Parameters
        ----------
        name : str
            The name of the region to register.
        raw_handle : bytes
            The raw serialized cudaIPC handle in base64 encoding.
        device_id : int
            The GPU device ID on which the cudaIPC handle was created.
        byte_size : int
            The size of the cuda shared memory region, in bytes.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Raises
        ------
        InferenceServerException
            If unable to register the specified cuda shared memory.

        """
        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            request = service_pb2.CudaSharedMemoryRegisterRequest(
                name=name,
                raw_handle=base64.b64decode(raw_handle),
                device_id=device_id,
                byte_size=byte_size)
            if self._verbose:
                print("register_cuda_shared_memory, metadata {}\n{}".format(
                    metadata, request))
            self._client_stub.CudaSharedMemoryRegister(request=request,
                                                       metadata=metadata)
            if self._verbose:
                print(
                    "Registered cuda shared memory with name '{}'".format(name))
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def unregister_cuda_shared_memory(self, name="", headers=None):
        """Request the server to unregister a cuda shared memory with the
        specified name.

        Parameters
        ----------
        name : str
            The name of the region to unregister. The default value is empty
            string which means all the cuda shared memory regions will be
            unregistered.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Raises
        ------
        InferenceServerException
            If unable to unregister the specified cuda shared memory region.

        """
        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()
        try:
            request = service_pb2.CudaSharedMemoryUnregisterRequest(name=name)
            if self._verbose:
                print("unregister_cuda_shared_memory, metadata {}\n{}".format(
                    metadata, request))
            self._client_stub.CudaSharedMemoryUnregister(request=request,
                                                         metadata=metadata)
            if self._verbose:
                if name is not "":
                    print(
                        "Unregistered cuda shared memory with name '{}'".format(
                            name))
                else:
                    print("Unregistered all cuda shared memory regions")
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def infer(self,
              model_name,
              inputs,
              model_version="",
              outputs=None,
              request_id="",
              sequence_id=0,
              sequence_start=False,
              sequence_end=False,
              priority=0,
              timeout=None,
              client_timeout=None,
              headers=None):
        """Run synchronous inference using the supplied 'inputs' requesting
        the outputs specified by 'outputs'.

        Parameters
        ----------
        model_name: str
            The name of the model to run inference.
        inputs : list
            A list of InferInput objects, each describing data for a input
            tensor required by the model.
        model_version : str
            The version of the model to run inference. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        outputs : list
            A list of InferRequestedOutput objects, each describing how the output
            data must be returned. If not specified all outputs produced
            by the model will be returned using default settings.
        request_id : str
            Optional identifier for the request. If specified will be returned
            in the response. Default value is an empty string which means no
            request_id will be used.
        sequence_id : int
            The unique identifier for the sequence being represented by the
            object. Default value is 0 which means that the request does not
            belong to a sequence.
        sequence_start : bool
            Indicates whether the request being added marks the start of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0.
        sequence_end : bool
            Indicates whether the request being added marks the end of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0.
        priority : int
            Indicates the priority of the request. Priority value zero
            indicates that the default priority level should be used
            (i.e. same behavior as not specifying the priority parameter).
            Lower value priorities indicate higher priority levels. Thus
            the highest priority level is indicated by setting the parameter
            to 1, the next highest is 2, etc. If not provided, the server
            will handle the request using default setting for the model.
        timeout : int
            The timeout value for the request, in microseconds. If the request
            cannot be completed within the time the server can take a
            model-specific action such as terminating the request. If not
            provided, the server will handle the request using default setting
            for the model.
        client_timeout : float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and raise
            InferenceServerExeption with message "Deadline Exceeded" when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.
        headers : dict
            Optional dictionary specifying additional HTTP headers to include
            in the request.

        Returns
        -------
        InferResult
            The object holding the result of the inference.

        Raises
        ------
        InferenceServerException
            If server fails to perform inference.
        """

        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()

        if type(model_version) != str:
            raise_error("model version must be a string")

        request = _get_inference_request(model_name=model_name,
                                         inputs=inputs,
                                         model_version=model_version,
                                         request_id=request_id,
                                         outputs=outputs,
                                         sequence_id=sequence_id,
                                         sequence_start=sequence_start,
                                         sequence_end=sequence_end,
                                         priority=priority,
                                         timeout=timeout)
        if self._verbose:
            print("infer, metadata {}\n{}".format(metadata, request))

        try:
            response = self._client_stub.ModelInfer(request=request,
                                                    metadata=metadata,
                                                    timeout=client_timeout)
            if self._verbose:
                print(response)
            result = InferResult(response)
            return result
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def async_infer(self,
                    model_name,
                    inputs,
                    callback,
                    model_version="",
                    outputs=None,
                    request_id="",
                    sequence_id=0,
                    sequence_start=False,
                    sequence_end=False,
                    priority=0,
                    timeout=None,
                    client_timeout=None,
                    headers=None):
        """Run asynchronous inference using the supplied 'inputs' requesting
        the outputs specified by 'outputs'.

        Parameters
        ----------
        model_name: str
            The name of the model to run inference.
        inputs : list
            A list of InferInput objects, each describing data for a input
            tensor required by the model.
        callback : function
            Python function that is invoked once the request is completed.
            The function must reserve the last two arguments (result, error)
            to hold InferResult and InferenceServerException objects
            respectively which will be provided to the function when executing
            the callback. The ownership of these objects will be given to the
            user. The 'error' would be None for a successful inference.
        model_version: str
            The version of the model to run inference. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        outputs : list
            A list of InferRequestedOutput objects, each describing how the output
            data must be returned. If not specified all outputs produced
            by the model will be returned using default settings.
        request_id : str
            Optional identifier for the request. If specified will be returned
            in the response. Default value is an empty string which means no
            request_id will be used.
        sequence_id : int
            The unique identifier for the sequence being represented by the
            object. Default value is 0 which means that the request does not
            belong to a sequence.
        sequence_start: bool
            Indicates whether the request being added marks the start of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0.
        sequence_end: bool
            Indicates whether the request being added marks the end of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0.
        priority : int
            Indicates the priority of the request. Priority value zero
            indicates that the default priority level should be used
            (i.e. same behavior as not specifying the priority parameter).
            Lower value priorities indicate higher priority levels. Thus
            the highest priority level is indicated by setting the parameter
            to 1, the next highest is 2, etc. If not provided, the server
            will handle the request using default setting for the model.
        timeout : int
            The timeout value for the request, in microseconds. If the request
            cannot be completed within the time the server can take a
            model-specific action such as terminating the request. If not
            provided, the server will handle the request using default setting
            for the model.
        client_timeout : float
            The maximum end-to-end time, in seconds, the request is allowed
            to take. The client will abort request and provide
            error with message "Deadline Exceeded" in the callback when the
            specified time elapses. The default value is None which means
            client will wait for the response from the server.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Raises
        ------
        InferenceServerException
            If server fails to issue inference.
        """

        def wrapped_callback(call_future):
            error = result = None
            try:
                response = call_future.result()
                if self._verbose:
                    print(response)
                result = InferResult(response)
            except grpc.RpcError as rpc_error:
                error = get_error_grpc(rpc_error)
            callback(result=result, error=error)

        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()

        if type(model_version) != str:
            raise_error("model version must be a string")

        request = _get_inference_request(model_name=model_name,
                                         inputs=inputs,
                                         model_version=model_version,
                                         request_id=request_id,
                                         outputs=outputs,
                                         sequence_id=sequence_id,
                                         sequence_start=sequence_start,
                                         sequence_end=sequence_end,
                                         priority=priority,
                                         timeout=timeout)
        if self._verbose:
            print("async_infer, metadata {}\n{}".format(metadata, request))

        try:
            self._call_future = self._client_stub.ModelInfer.future(
                request=request, metadata=metadata, timeout=client_timeout)
            self._call_future.add_done_callback(wrapped_callback)
            if self._verbose:
                verbose_message = "Sent request"
                if request_id is not "":
                    verbose_message = verbose_message + " '{}'".format(
                        request_id)
                print(verbose_message)
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def start_stream(self, callback, stream_timeout=None, headers=None):
        """Starts a grpc bi-directional stream to send streaming inferences.
        Note: When using stream, user must ensure the InferenceServerClient.close()
        gets called at exit.

        Parameters
        ----------
        callback : function
            Python function that is invoked upon receiving response from
            the underlying stream. The function must reserve the last two
            arguments (result, error) to hold InferResult and
            InferenceServerException objects respectively which will be
            provided to the function when executing the callback. The
            ownership of these objects will be given to the user. The
            'error' would be None for a successful inference.
        stream_timeout : float
            Optional stream timeout. The stream will be closed once the
            specified timeout expires.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request.

        Raises
        ------
        InferenceServerException
            If unable to start a stream or a stream was already running
            for this client.

        """
        if self._stream is not None:
            raise_error("cannot start another stream with one already running. "
                        "'InferenceServerClient' supports only a single active "
                        "stream at a given time.")

        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()

        self._stream = _InferStream(callback, self._verbose)

        try:
            response_iterator = self._client_stub.ModelStreamInfer(
                _RequestIterator(self._stream),
                metadata=metadata,
                timeout=stream_timeout)
            self._stream._init_handler(response_iterator)
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def stop_stream(self):
        """Stops a stream if one available.
        """
        if self._stream is not None:
            self._stream.close()
        self._stream = None

    def async_stream_infer(self,
                           model_name,
                           inputs,
                           model_version="",
                           outputs=None,
                           request_id="",
                           sequence_id=0,
                           sequence_start=False,
                           sequence_end=False,
                           priority=0,
                           timeout=None):
        """Runs an asynchronous inference over gRPC bi-directional streaming
        API. A stream must be established with a call to start_stream()
        before calling this function. All the results will be provided to the
        callback function associated with the stream.

        Parameters
        ----------
        model_name: str
            The name of the model to run inference.
        inputs : list
            A list of InferInput objects, each describing data for a input
            tensor required by the model.
        model_version: str
            The version of the model to run inference. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        outputs : list
            A list of InferRequestedOutput objects, each describing how the output
            data must be returned. If not specified all outputs produced
            by the model will be returned using default settings.
        request_id : str
            Optional identifier for the request. If specified will be returned
            in the response. Default value is an empty string which means no
            request_id will be used.
        sequence_id : int
            The unique identifier for the sequence being represented by the
            object. Default value is 0 which means that the request does not
            belong to a sequence.
        sequence_start: bool
            Indicates whether the request being added marks the start of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0.
        sequence_end: bool
            Indicates whether the request being added marks the end of the
            sequence. Default value is False. This argument is ignored if
            'sequence_id' is 0.
        priority : int
            Indicates the priority of the request. Priority value zero
            indicates that the default priority level should be used
            (i.e. same behavior as not specifying the priority parameter).
            Lower value priorities indicate higher priority levels. Thus
            the highest priority level is indicated by setting the parameter
            to 1, the next highest is 2, etc. If not provided, the server
            will handle the request using default setting for the model.
        timeout : int
            The timeout value for the request, in microseconds. If the request
            cannot be completed within the time the server can take a
            model-specific action such as terminating the request. If not
            provided, the server will handle the request using default setting
            for the model.

        Raises
        ------
        InferenceServerException
            If server fails to issue inference.
        """

        if self._stream is None:
            raise_error(
                "stream not available, use start_stream() to make one available."
            )

        if type(model_version) != str:
            raise_error("model version must be a string")

        request = _get_inference_request(model_name=model_name,
                                         inputs=inputs,
                                         model_version=model_version,
                                         request_id=request_id,
                                         outputs=outputs,
                                         sequence_id=sequence_id,
                                         sequence_start=sequence_start,
                                         sequence_end=sequence_end,
                                         priority=priority,
                                         timeout=timeout)
        if self._verbose:
            print("async_stream_infer\n{}".format(request))
        # Enqueues the request to the stream
        self._stream._enqueue_request(request)
        if self._verbose:
            print("enqueued request {} to stream...".format(request_id))


class InferInput:
    """An object of InferInput class is used to describe
    input tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of input whose data will be described by this object
    shape : list
        The shape of the associated input.
    datatype : str
        The datatype of the associated input.

    """

    def __init__(self, name, shape, datatype):
        self._input = service_pb2.ModelInferRequest().InferInputTensor()
        self._input.name = name
        self._input.ClearField('shape')
        self._input.shape.extend(shape)
        self._input.datatype = datatype
        self._raw_content = None

    def name(self):
        """Get the name of input associated with this object.

        Returns
        -------
        str
            The name of input
        """
        return self._input.name

    def datatype(self):
        """Get the datatype of input associated with this object.

        Returns
        -------
        str
            The datatype of input
        """
        return self._input.datatype

    def shape(self):
        """Get the shape of input associated with this object.

        Returns
        -------
        list
            The shape of input
        """
        return self._input.shape

    def set_shape(self, shape):
        """Set the shape of input.

        Parameters
        ----------
        shape : list
            The shape of the associated input.
        """
        self._input.ClearField('shape')
        self._input.shape.extend(shape)

    def set_data_from_numpy(self, input_tensor):
        """Set the tensor data from the specified numpy array for
        input associated with this object.

        Parameters
        ----------
        input_tensor : numpy array
            The tensor data in numpy array format

        Raises
        ------
        InferenceServerException
            If failed to set data for the tensor.
        """
        if not isinstance(input_tensor, (np.ndarray,)):
            raise_error("input_tensor must be a numpy array")
        dtype = np_to_triton_dtype(input_tensor.dtype)
        if self._input.datatype != dtype:
            raise_error(
                "got unexpected datatype {} from numpy array, expected {}".
                format(dtype, self._input.datatype))
        valid_shape = True
        if len(self._input.shape) != len(input_tensor.shape):
            valid_shape = False
        for i in range(len(self._input.shape)):
            if self._input.shape[i] != input_tensor.shape[i]:
                valid_shape = False
        if not valid_shape:
            raise_error(
                "got unexpected numpy array shape [{}], expected [{}]".format(
                    str(input_tensor.shape)[1:-1],
                    str(self._input.shape)[1:-1]))

        self._input.parameters.pop('shared_memory_region', None)
        self._input.parameters.pop('shared_memory_byte_size', None)
        self._input.parameters.pop('shared_memory_offset', None)

        if self._input.datatype == "BYTES":
            self._raw_content = serialize_byte_tensor(input_tensor).tobytes()
        else:
            self._raw_content = input_tensor.tobytes()

    def set_shared_memory(self, region_name, byte_size, offset=0):
        """Set the tensor data from the specified shared memory region.

        Parameters
        ----------
        region_name : str
            The name of the shared memory region holding tensor data.
        byte_size : int
            The size of the shared memory region holding tensor data.
        offset : int
            The offset, in bytes, into the region where the data for
            the tensor starts. The default value is 0.

        """
        self._input.ClearField("contents")
        self._raw_content = None

        self._input.parameters[
            'shared_memory_region'].string_param = region_name
        self._input.parameters[
            'shared_memory_byte_size'].int64_param = byte_size
        if offset != 0:
            self._input.parameters['shared_memory_offset'].int64_param = offset

    def _get_tensor(self):
        """Retrieve the underlying InferInputTensor message.
        Returns
        -------
        protobuf message
            The underlying InferInputTensor protobuf message.
        """
        return self._input

    def _get_content(self):
        """Retrieve the contents for this tensor in raw bytes.
        Returns
        -------
        bytes
            The associated contents for this tensor in raw bytes.
        """
        return self._raw_content


class InferRequestedOutput:
    """An object of InferRequestedOutput class is used to describe a
    requested output tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of output tensor to associate with this object
    class_count : int
        The number of classifications to be requested. The default
        value is 0 which means the classification results are not
        requested.
    """

    def __init__(self, name, class_count=0):
        self._output = service_pb2.ModelInferRequest(
        ).InferRequestedOutputTensor()
        self._output.name = name
        if class_count != 0:
            self._output.parameters['classification'].int64_param = class_count

    def name(self):
        """Get the name of output associated with this object.

        Returns
        -------
        str
            The name of output
        """
        return self._output.name

    def set_shared_memory(self, region_name, byte_size, offset=0):
        """Marks the output to return the inference result in
        specified shared memory region.

        Parameters
        ----------
        region_name : str
            The name of the shared memory region to hold tensor data.
        byte_size : int
            The size of the shared memory region to hold tensor data.
        offset : int
            The offset, in bytes, into the region where the data for
            the tensor starts. The default value is 0.

        Raises
        ------
        InferenceServerException
            If failed to set shared memory for the tensor.
        """
        if 'classification' in self._output.parameters:
            raise_error("shared memory can't be set on classification output")

        self._output.parameters[
            'shared_memory_region'].string_param = region_name
        self._output.parameters[
            'shared_memory_byte_size'].int64_param = byte_size
        if offset != 0:
            self._output.parameters['shared_memory_offset'].int64_param = offset

    def unset_shared_memory(self):
        """Clears the shared memory option set by the last call to
        InferRequestedOutput.set_shared_memory(). After call to this
        function requested output will no longer be returned in a
        shared memory region.
        """

        self._output.parameters.pop('shared_memory_region', None)
        self._output.parameters.pop('shared_memory_byte_size', None)
        self._output.parameters.pop('shared_memory_offset', None)

    def _get_tensor(self):
        """Retrieve the underlying InferRequestedOutputTensor message.
        Returns
        -------
        protobuf message
            The underlying InferRequestedOutputTensor protobuf message.
        """
        return self._output


class InferResult:
    """An object of InferResult class holds the response of
    an inference request and provide methods to retrieve
    inference results.

    Parameters
    ----------
    result : protobuf message
        The ModelInferResponse returned by the server
    """

    def __init__(self, result):
        self._result = result

    def as_numpy(self, name):
        """Get the tensor data for output associated with this object
        in numpy format

        Parameters
        ----------
        name : str
            The name of the output tensor whose result is to be retrieved.

        Returns
        -------
        numpy array
            The numpy array containing the response data for the tensor or
            None if the data for specified tensor name is not found.
        """
        index = 0
        for output in self._result.outputs:
            if output.name == name:
                shape = []
                for value in output.shape:
                    shape.append(value)

                datatype = output.datatype
                if index < len(self._result.raw_output_contents):
                    if datatype == 'BYTES':
                        # String results contain a 4-byte string length
                        # followed by the actual string characters. Hence,
                        # need to decode the raw bytes to convert into
                        # array elements.
                        np_array = deserialize_bytes_tensor(
                            self._result.raw_output_contents[index])
                    else:
                        np_array = np.frombuffer(
                            self._result.raw_output_contents[index],
                            dtype=triton_to_np_dtype(datatype))
                elif len(output.contents.byte_contents) != 0:
                    np_array = np.array(output.contents.byte_contents)
                else:
                    np_array = np.empty(0)
                np_array = np.resize(np_array, shape)
                return np_array
            else:
                index += 1
        return None

    def get_output(self, name, as_json=False):
        """Retrieves the InferOutputTensor corresponding to the
        named ouput.

        Parameters
        ----------
        name : str
            The name of the tensor for which Output is to be
            retrieved.
        as_json : bool
            If True then returns response as a json dict, otherwise
            as a protobuf message. Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.

        Returns
        -------
        protobuf message or dict
            If a InferOutputTensor with specified name is present in
            ModelInferResponse then returns it as a protobuf messsage
            or dict, otherwise returns None.
        """
        for output in self._result.outputs:
            if output.name == name:
                if as_json:
                    return json.loads(
                        MessageToJson(output, preserving_proto_field_name=True))
                else:
                    return output

        return None

    def get_response(self, as_json=False):
        """Retrieves the complete ModelInferResponse as a
        json dict object or protobuf message

        Parameters
        ----------
        as_json : bool
            If True then returns response as a json dict, otherwise
            as a protobuf message. Default value is False.
            The returned json is generated from the protobuf message
            using MessageToJson and as a result int64 values are
            represented as string. It is the caller's responsibility
            to convert these strings back to int64 values as
            necessary.

        Returns
        -------
        protobuf message or dict
            The underlying ModelInferResponse as a protobuf message or dict.
        """
        if as_json:
            return json.loads(
                MessageToJson(self._result, preserving_proto_field_name=True))
        else:
            return self._result


class _InferStream:
    """Supports sending inference requests and receiving corresponding
    requests on a gRPC bi-directional stream.

    Parameters
    ----------
    callback : function
        Python function that is invoked upon receiving response from
        the underlying stream. The function must reserve the last two
        arguments (result, error) to hold InferResult and
        InferenceServerException objects respectively which will be
        provided to the function when executing the callback. The
        ownership of these objects will be given to the user. The
        'error' would be None for a successful inference.
    """

    def __init__(self, callback, verbose):
        self._callback = callback
        self._verbose = verbose
        self._request_queue = queue.Queue()
        self._handler = None

    def __del__(self):
        self.close()

    def close(self):
        """Gracefully close underlying gRPC streams. Note that this call
        blocks till response of all currently enqueued requests are not
        received.
        """
        if self._handler is not None:
            self._request_queue.put(None)
            if self._handler.is_alive():
                self._handler.join()
                if self._verbose:
                    print("stream stopped...")
            self._handler = None

    def _init_handler(self, response_iterator):
        """Initializes the handler to process the response from
        stream and execute the callbacks.

        Parameters
        ----------
        response_iterator : iterator
            The iterator over the gRPC response stream.

        """
        if self._handler is not None:
            raise_error(
                'Attempted to initialize already initialized InferStream')
        # Create a new thread to handle the gRPC response stream
        self._handler = threading.Thread(target=self._process_response,
                                         args=(response_iterator,))
        self._handler.start()
        if self._verbose:
            print("stream started...")

    def _enqueue_request(self, request):
        """Enqueues the specified request object to be provided
        in gRPC request stream.

        Parameters
        ----------
        request : ModelInferRequest
            The protobuf message holding the ModelInferRequest

        """
        self._request_queue.put(request)

    def _get_request(self):
        """Returns the request details in the order they were added.
        The call to this function will block until the requests
        are available in the queue. InferStream._enqueue_request
        adds the request to the queue.

        Returns
        -------
        protobuf message
            The ModelInferRequest protobuf message.

        """
        request = self._request_queue.get()
        return request

    def _process_response(self, responses):
        """Worker thread function to iterate through the response stream and
        executes the provided callbacks.

        Parameters
        ----------
        responses : iterator
            The iterator to the response from the server for the
            requests in the stream.

        """
        try:
            for response in responses:
                if self._verbose:
                    print(response)
                result = error = None
                if response.error_message != "":
                    error = InferenceServerException(msg=response.error_message)
                else:
                    result = InferResult(response.infer_response)
                self._callback(result=result, error=error)
        except grpc.RpcError as rpc_error:
            error = get_error_grpc(rpc_error)
            self._callback(result=None, error=error)


class _RequestIterator:
    """An iterator class to provide data to gRPC request stream.

    Parameters
    ----------
    stream : InferStream
        The InferStream that holds the context to an active stream.

    """

    def __init__(self, stream):
        self._stream = stream

    def __iter__(self):
        return self

    def __next__(self):
        request = self._stream._get_request()
        if request is None:
            raise StopIteration

        return request
