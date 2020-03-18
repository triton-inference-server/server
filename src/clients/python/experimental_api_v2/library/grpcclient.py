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

import base64
import numpy as np
import grpc
import rapidjson as json
from google.protobuf.json_format import MessageToJson

from tritongrpcclient import grpc_service_v2_pb2
from tritongrpcclient import grpc_service_v2_pb2_grpc
from tritongrpcclient.utils import *


def get_error_grpc(rpc_error):
    return InferenceServerException(
        msg=rpc_error.details(),
        status=str(rpc_error.code()),
        debug_details=rpc_error.debug_error_string())


def raise_error_grpc(rpc_error):
    raise get_error_grpc(rpc_error) from None


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
            request = grpc_service_v2_pb2.ServerLiveRequest()
            response = self._client_stub.ServerLive(request=request,
                                                    metadata=metadata)
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
            request = grpc_service_v2_pb2.ServerReadyRequest()
            response = self._client_stub.ServerReady(request=request,
                                                     metadata=metadata)
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
            request = grpc_service_v2_pb2.ModelReadyRequest(
                name=model_name, version=model_version)
            response = self._client_stub.ModelReady(request=request,
                                                    metadata=metadata)
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
            otherwise as a protobuf message. Default value is False.

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
            request = grpc_service_v2_pb2.ServerMetadataRequest()
            response = self._client_stub.ServerMetadata(
                request=request, metadata=metadata)
            if as_json:
                return json.loads(MessageToJson(response))
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
            If True then returns model metadata as a json dict, otherwise
            as a protobuf message. Default value is False.

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
            request = grpc_service_v2_pb2.ModelMetadataRequest(
                name=model_name, version=model_version)
            response = self._client_stub.ModelMetadata(request=request,
                                                       metadata=metadata)
            if as_json:
                return json.loads(MessageToJson(response))
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
            request = grpc_service_v2_pb2.ModelConfigRequest(
                name=model_name, version=model_version)
            response = self._client_stub.ModelConfig(request=request,
                                                     metadata=metadata)
            if as_json:
                return json.loads(MessageToJson(response))
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
            request = grpc_service_v2_pb2.RepositoryIndexRequest()
            response = self._client_stub.RepositoryIndex(
                request=request, metadata=metadata)
            if as_json:
                return json.loads(MessageToJson(response))
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
            request = grpc_service_v2_pb2.RepositoryModelLoadRequest(
                model_name=model_name)
            self._client_stub.RepositoryModelLoad(request=request,
                                                  metadata=metadata)
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
            request = grpc_service_v2_pb2.RepositoryModelUnloadRequest(
                model_name=model_name)
            self._client_stub.RepositoryModelUnload(request=request,
                                                    metadata=metadata)
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
            If True then returns system shared memory status as a 
            json dict, otherwise as a protobuf message. Default
            value is False.

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
            request = grpc_service_v2_pb2.SystemSharedMemoryStatusRequest(
                name=region_name)
            response = self._client_stub.SystemSharedMemoryStatus(
                request=request, metadata=metadata)
            if as_json:
                return json.loads(MessageToJson(response))
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
            request = grpc_service_v2_pb2.SystemSharedMemoryRegisterRequest(
                name=name, key=key, offset=offset, byte_size=byte_size)
            self._client_stub.SystemSharedMemoryRegister(
                request=request, metadata=metadata)
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
            request = grpc_service_v2_pb2.SystemSharedMemoryUnregisterRequest(
                name=name)
            self._client_stub.SystemSharedMemoryUnregister(
                request=request, metadata=metadata)
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
            If True then returns cuda shared memory status as a 
            json dict, otherwise as a protobuf message. Default
            value is False.

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
            request = grpc_service_v2_pb2.CudaSharedMemoryStatusRequest(
                name=region_name)
            response = self._client_stub.CudaSharedMemoryStatus(
                request=request, metadata=metadata)
            if as_json:
                return json.loads(MessageToJson(response))
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
            request = grpc_service_v2_pb2.CudaSharedMemoryRegisterRequest(
                name=name,
                raw_handle=base64.b64decode(raw_handle),
                device_id=device_id,
                byte_size=byte_size)
            self._client_stub.CudaSharedMemoryRegister(request=request,
                                                       metadata=metadata)
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
            request = grpc_service_v2_pb2.CudaSharedMemoryUnregisterRequest(
                name=name)
            self._client_stub.CudaSharedMemoryUnregister(
                request=request, metadata=metadata)
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def infer(self,
              inputs,
              outputs,
              model_name,
              model_version="",
              request_id=None,
              sequence_id=0,
              sequence_start=False,
              sequence_end=False,
              headers=None):
        """Run synchronous inference using the supplied 'inputs' requesting
        the outputs specified by 'outputs'.

        Parameters
        ----------
        inputs : list
            A list of InferInput objects, each describing data for a input
            tensor required by the model.
        outputs : list
            A list of InferOutput objects, each describing how the output
            data must be returned. Only the output tensors present in the
            list will be requested from the server.
        model_name: str
            The name of the model to run inference.
        model_version: str
            The version of the model to run inference. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        request_id: str
            Optional identifier for the request. If specified will be returned
            in the response. Default value is 'None' which means no request_id
            will be used.
        sequence_id: int
            Indicates which sequence does the request belongs to. All inference
            requests that belong to the same sequence must use the same sequence
            ID. A sequence ID of 0 indicates the inference request is not part of
            a sequence. Default value is 0.
        sequence_start: bool
            Indicates whether the request is first in the sequence. Default value
            is False.
        sequence_end: bool
            Indicates whether the request is last in the sequence. Default value
            is False.
        headers: dict
            Optional dictionary specifying additional HTTP
            headers to include in the request

        Returns
        -------
        InferResult
            The object holding the result of the inference, including the
            statistics.

        Raises
        ------
        InferenceServerException
            If server fails to perform inference.
        """

        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()

        request = self._get_inference_request(inputs, outputs, model_name,
                                              model_version, request_id,
                                              parameters)

        try:
            response = self._client_stub.ModelInfer(request=request,
                                                    metadata=metadata)
            result = InferResult(response)
            return result
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def async_infer(self,
                    callback,
                    inputs,
                    outputs,
                    model_name,
                    model_version="",
                    request_id=None,
                    headers=None):
        """Run asynchronous inference using the supplied 'inputs' requesting
        the outputs specified by 'outputs'.

        Parameters
        ----------
        callback : function
            Python function that is invoked once the request is completed.
            The function must reserve the last two arguments (result, error)
            to hold InferResult and InferenceServerException objects
            respectively which will be provided to the function when executing
            the callback. The ownership of these objects will be given to the
            user. The 'error' would be None for a successful inference.
        inputs : list
            A list of InferInput objects, each describing data for a input
            tensor required by the model.
        outputs : list
            A list of InferOutput objects, each describing how the output
            data must be returned. Only the output tensors present in the
            list will be requested from the server.
        model_name: str
            The name of the model to run inference.
        model_version: str
            The version of the model to run inference. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        request_id: str
            Optional identifier for the request. If specified will be returned
            in the response. Default value is 'None' which means no request_id
            will be used.
        sequence_id: int
            Indicates which sequence does the request belongs to. All inference
            requests that belong to the same sequence must use the same sequence
            ID. A sequence ID of 0 indicates the inference request is not part of
            a sequence. Default value is 0.
        sequence_start: bool
            Indicates whether the request is first in the sequence. Default value
            is False.
        sequence_end: bool
            Indicates whether the request is last in the sequence. Default value
            is False.
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
                result = InferResult(call_future.result())
            except grpc.RpcError as rpc_error:
                error = get_error_grpc(rpc_error)
            callback(result=result, error=error)

        if headers is not None:
            metadata = headers.items()
        else:
            metadata = ()

        request = self._get_inference_request(inputs, outputs, model_name,
                                              model_version, request_id,
                                              parameters)

        try:
            self._call_future = self._client_stub.ModelInfer.future(
                request=request, metadata=metadata)
            self._call_future.add_done_callback(wrapped_callback)
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def _get_inference_request(self, inputs, outputs, model_name, model_version,
                               request_id, parameters):
        """Creates and initializes an inference request.

        Parameters
        ----------
        inputs : list
            A list of InferInput objects, each describing data for a input
            tensor required by the model.
        outputs : list
            A list of InferOutput objects, each describing how the output
            data must be returned. Only the output tensors present in the
            list will be requested from the server.
        model_name: str
            The name of the model to run inference.
        model_version: str
            The version of the model to run inference. The default value
            is an empty string which means then the server will choose
            a version based on the model and internal policy.
        request_id: str
            Optional identifier for the request. If specified will be returned
            in the response. Default value is 'None' which means no request_id
            will be used.
        parameters: dict
            Optional inference parameters described as key-value pairs.

        Returns
        -------
        ModelInferRequest
            The protobuf message holding the inference request.
        
        Raises
        ------
        InferenceServerException
            If server fails to issue inference.

        """

        request = grpc_service_v2_pb2.ModelInferRequest()
        request.model_name = model_name
        request.model_version = model_version
        if request_id != None:
            request.id = request_id
        for infer_input in inputs:
            request.inputs.extend([infer_input._get_tensor()])
        for infer_output in outputs:
            request.outputs.extend([infer_output._get_tensor()])
        if parameters:
            for param_key in parameters:
                _set_parameter(request,
                               key=param_key,
                               value=parameters[param_key])

        return request

    def _set_parameter(self, request, key, value):
        """Adds the specified key-value pair to the request

        Parameters
        ----------
        request : protobuf message
            The ModelInferRequest object to add the parameter to.
        key : str
            The name of the parameter to be included in the request. 
        value : str/int/bool
            The value of the parameter

        Raises
        ------
        InferenceServerException
            If server fails to add the parameter to request.

        """
        if not type(key) is str:
            raise_error(
                "only string data type for key is supported in parameters")

        param = request.parameters[key]
        if type(value) is int:
            param.int64_param = value
        elif type(value) is bool:
            param.bool_param = value
        elif type(value) is str:
            param.string_param = value
        else:
            raise_error("unsupported value type for the parameter")


class InferInput:
    """An object of InferInput class is used to describe
    input tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of input whose data will be described by this object
    shape : list
        The shape of the associated input. Default value is None.
    datatype : str
        The datatype of the associated input. Default is None.

    """

    def __init__(self, name, shape=None, datatype=None):
        self._input = grpc_service_v2_pb2.ModelInferRequest().InferInputTensor()
        self._input.name = name
        if shape:
            self._input.ClearField('shape')
            self._input.shape.extend(shape)
        if datatype:
            self._input.datatype = datatype

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

    def set_data_from_numpy(self, input_tensor):
        """Set the tensor data (datatype, shape, contents) from the
        specified numpy array for input associated with this object.

        Parameters
        ----------
        input_tensor : numpy array
            The tensor data in numpy array format
        """
        if not isinstance(input_tensor, (np.ndarray,)):
            raise_error("input_tensor must be a numpy array")
        self._input.datatype = np_to_triton_dtype(input_tensor.dtype)
        self._input.ClearField('shape')
        self._input.shape.extend(input_tensor.shape)
        if self._input.datatype == "BYTES":
            self._input.contents.raw_contents = serialize_byte_tensor(
                input_tensor).tobytes()
        else:
            self._input.contents.raw_contents = input_tensor.tobytes()

    def set_parameter(self, key, value):
        """Adds the specified key-value pair in the requested input parameters

        Parameters
        ----------
        key : str
            The name of the parameter to be included in the request. 
        value : str/int/bool
            The value of the parameter
        
        """
        if not type(key) is str:
            raise_error(
                "only string data type for key is supported in parameters")

        param = self._input.parameters[key]
        if type(value) is int:
            param.int64_param = value
        elif type(value) is bool:
            param.bool_param = value
        elif type(value) is str:
            param.string_param = value
        else:
            raise_error("unsupported value type for the parameter")

    def clear_parameters(self):
        """Clears all the parameters that have been added to the input request.
        
        """
        self._input.parameters.clear()

    def _get_tensor(self):
        """Retrieve the underlying InferInputTensor message.
        Returns
        -------
        protobuf message 
            The underlying InferInputTensor protobuf message.
        """
        return self._input


class InferOutput:
    """An object of InferOutput class is used to describe a
    requested output tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of output tensor to associate with this object
    """

    def __init__(self, name):
        self._output = grpc_service_v2_pb2.ModelInferRequest(
        ).InferRequestedOutputTensor()
        self._output.name = name

    def name(self):
        """Get the name of output associated with this object.

        Returns
        -------
        str
            The name of output
        """
        return self._output.name

    def set_parameter(self, key, value):
        """Adds the specified key-value pair in the requested output parameters

        Parameters
        ----------
        key : str
            The name of the parameter to be included in the request. 
        value : str/int/bool
            The value of the parameter
        
        """
        if not type(key) is str:
            raise_error(
                "only string data type for key is supported in parameters")

        param = self._output.parameters[key]
        if type(value) is int:
            param.int64_param = value
        elif type(value) is bool:
            param.bool_param = value
        elif type(value) is str:
            param.string_param = value
        else:
            raise_error("unsupported value type for the parameter")

    def clear_parameters(self):
        """Clears all the parameters that have been added to the output request.
        
        """
        self._output.parameters.clear()

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
        for output in self._result.outputs:
            if output.name == name:
                shape = []
                for value in output.shape:
                    shape.append(value)

                datatype = output.datatype
                if len(output.contents.raw_contents) != 0:
                    if datatype == 'BYTES':
                        # String results contain a 4-byte string length
                        # followed by the actual string characters. Hence,
                        # need to decode the raw bytes to convert into
                        # array elements.
                        np_array = deserialize_bytes_tensor(
                            output.contents.raw_contents)
                    else:
                        np_array = np.frombuffer(
                            output.contents.raw_contents,
                            dtype=triton_to_np_dtype(datatype))
                elif len(output.contents.byte_contents) != 0:
                    np_array = np.array(output.contents.byte_contents)
                np_array = np.resize(np_array, shape)
                return np_array
        return None

    def get_statistics(self, as_json=False):
        """Retrieves the InferStatistics for this response as
        a json dict object or protobuf message

        Parameters
        ----------
        as_json : bool
            If True then returns statistics as a json dict, otherwise
            as a protobuf message. Default value is False.
        
        Returns
        -------
        protobuf message or dict
            The InferStatistics protobuf message or dict for this response.
        """
        if as_json:
            return json.loads(MessageToJson(self._result.statistics))
        else:
            return self._result.statistics

    def get_response(self, as_json=False):
        """Retrieves the complete ModelInferResponse as a
        json dict object or protobuf message

        Parameters
        ----------
        as_json : bool
            If True then returns response as a json dict, otherwise
            as a protobuf message. Default value is False.
    
        Returns
        -------
        protobuf message or dict
            The underlying ModelInferResponse as a protobuf message or dict.
        """
        if as_json:
            return json.loads(MessageToJson(self._result))
        else:
            return self._result
