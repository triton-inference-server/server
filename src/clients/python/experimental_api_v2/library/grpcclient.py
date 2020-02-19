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

import numpy as np
import grpc
import rapidjson as json
from google.protobuf.json_format import MessageToJson

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

    def get_server_metadata(self, as_json=False):
        """Contact the inference server and get its metadata.

        Parameters
        ----------
        as_json : bool
            If true then returns server metadata as a json dict,
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
        try:
            self._request = grpc_service_v2_pb2.ServerMetadataRequest()
            self._response = self._client_stub.ServerMetadata(self._request)
            if as_json:
                return json.loads(MessageToJson(self._response))
            else:
                return self._response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_model_metadata(self, model_name, model_version=-1, as_json=False):
        """Contact the inference server and get the metadata for specified model.

        Parameters
        ----------
        model_name: str
            The name of the model
        model_version: int
            The version of the model to get metadata. If -1 is given the 
            server will choose a version based on the model and internal policy.
        as_json : bool
            If true then returns model metadata as a json dict, otherwise
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
        try:
            self._request = grpc_service_v2_pb2.ModelMetadataRequest(
                name=model_name, version=model_version)
            self._response = self._client_stub.ModelMetadata(self._request)
            if as_json:
                return json.loads(MessageToJson(self._response))
            else:
                return self._response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def get_model_config(self, model_name, model_version=-1, as_json=False):
        """Contact the inference server and get the configuration for specified model.

        Parameters
        ----------
        model_name: str
            The name of the model
        model_version: int
            The version of the model to get configuration. If -1 is given the 
            server will choose a version based on the model and internal policy.
        as_json : bool
            If true then returns configuration as a json dict, otherwise
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
        try:
            self._request = grpc_service_v2_pb2.ModelConfigRequest(
                name=model_name, version=model_version)
            self._response = self._client_stub.ModelConfig(self._request)
            if as_json:
                return json.loads(MessageToJson(self._response))
            else:
                return self._response
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    # FIXMEPV2: Add model control support
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

    # FIXMEPV2: Add model control support
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

    # FIXMEPV2: Add parameter support
    @property
    def parameters(self):
        raise_error("Not implemented yet")

    def infer(self,
              inputs,
              outputs,
              model_name,
              model_version=-1,
              request_id=None,
              sequence_id=0):
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
        model_version: int
            The version of the model to run inference. If -1 is given the 
            server will choose a version based on the model and internal policy.
        request_id: string
            Optional identifier for the request. If specified will be returned
            in the response. Default value is 'None' which means no request_id
            will be used.
        sequence_id : int
            The sequence ID of the inference request. Default is 0, which
            indicates that the request is not part of a sequence. The
            sequence ID is used to indicate that two or more inference
            requests are in the same sequence.

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

        self._get_inference_request(inputs, outputs, model_name, model_version,
                                    request_id, sequence_id)

        try:
            self._response = self._client_stub.ModelInfer(self._request)
            self._result = InferResult(self._response)
            return self._result
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def async_infer(self,
                    callback,
                    inputs,
                    outputs,
                    model_name,
                    model_version=-1,
                    request_id=None,
                    sequence_id=None):
        """Run asynchronous inference using the supplied 'inputs' requesting
        the outputs specified by 'outputs'.

        Parameters
        ----------
        callback : function
            Python function that is invoked once the request is completed.
            The function must reserve the last argument to hold InferResult
            object which will be provided to the function when executing
            the callback. The ownership of this InferResult object will be
            given to the user and the its lifetime is limited to the scope
            of this function.
        inputs : list
            A list of InferInput objects, each describing data for a input
            tensor required by the model.
        outputs : list
            A list of InferOutput objects, each describing how the output
            data must be returned. Only the output tensors present in the
            list will be requested from the server.
        model_name: str
            The name of the model to run inference.
        model_version: int
            The version of the model to run inference. If -1 is given the 
            server will choose a version based on the model and internal policy.
        request_id: string
            Optional identifier for the request. If specified will be returned
            in the response. Default value is 'None' which means no request_id
            will be used.
        sequence_id : int
            The sequence ID of the inference request. Default is 0, which
            indicates that the request is not part of a sequence. The
            sequence ID is used to indicate that two or more inference
            requests are in the same sequence.
    
        Raises
        ------
        InferenceServerException
            If server fails to issue inference.
        """

        def wrapped_callback(call_future):
            try:
                result = InferResult(call_future.result())
            except grpc.RpcError as rpc_error:
                raise_error_grpc(rpc_error)
            callback(result=result)

        self._get_inference_request(inputs, outputs, model_name, model_version,
                                    request_id, sequence_id)

        try:
            self._call_future = self._client_stub.ModelInfer.future(
                self._request)
            self._call_future.add_done_callback(wrapped_callback)
        except grpc.RpcError as rpc_error:
            raise_error_grpc(rpc_error)

    def _get_inference_request(self, inputs, outputs, model_name, model_version,
                               request_id, sequence_id):
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
        model_version: int
            The version of the model to run inference. If -1 is given the 
            server will choose a version based on the model and internal policy.
        request_id: string
            Optional identifier for the request. If specified will be returned
            in the response. Default value is 'None' which means no request_id
            will be used.
        sequence_id : int
            The sequence ID of the inference request. Default is 0, which
            indicates that the request is not part of a sequence. The
            sequence ID is used to indicate that two or more inference
            requests are in the same sequence.
        """

        self._request = grpc_service_v2_pb2.ModelInferRequest()
        self._request.model_name = model_name
        self._request.model_version = model_version
        if request_id != None:
            self._request.id = request_id
        if sequence_id != None:
            self._request.sequence_id = sequence_id
        for infer_input in inputs:
            self._request.inputs.extend([infer_input._get_tensor()])
        for infer_output in outputs:
            self._request.outputs.extend([infer_output._get_tensor()])


class InferInput:
    """An object of InferInput class is used to describe
    input tensor for an inference request.

    Parameters
    ----------
    name : str
        The name of input whose data will be described by this object

    """

    def __init__(self, name):
        self._input = grpc_service_v2_pb2.ModelInferRequest().InferInputTensor()
        self._input.name = name

    @property
    def name(self):
        """Get the name of input associated with this object.

        Returns
        -------
        str
            The name of input
        """
        return self._input.name

    @property
    def datatype(self):
        """Get the datatype of input associated with this object.

        Returns
        -------
        str
            The datatype of input
        """
        return self._input.datatype

    @property
    def shape(self):
        """Get the shape of input associated with this object.

        Returns
        -------
        str
            The shape of input
        """
        return self._input.shape

    def set_data_from_numpy(self, input_tensor):
        """Set the tensor data (datatype, shape, contents) for
        input associated with this object.

        Parameters
        ----------
        input_tensor : numpy array
            The tensor data in numpy array format
        """
        if not isinstance(input_tensor, (np.ndarray,)):
            raise_error("input_tensor must be a numpy array")
        self._input.datatype = np_to_trtis_dtype(input_tensor.dtype)
        self._input.ClearField('shape')
        self._input.shape.extend(input_tensor.shape)
        if self._input.datatype == "STRING":
            self._input.contents.raw_contents = serialize_string_tensor(
                input_tensor).tobytes()
        else:
            self._input.contents.raw_contents = input_tensor.tobytes()

    # FIXMEPV2: Add parameter support
    @property
    def parameters(self):
        raise_error("Not implemented yet")

    def _get_tensor(self):
        """Retrieve the underlying InferInputTensor message.
        Returns
        -------
        Protobuf Message 
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
        
    data_format : str
        The format to use when returning the ouput tensor data. Options
        are "explicit", "binary" and "shared_memory".
        Default is "binary". If "shared_memory" is specified then
        the "shared_memory_data" field must also be specified.
        Server support for “binary” and “shared_memory”
        is optional.
    """

    def __init__(self, name, data_format="binary"):
        self._output = grpc_service_v2_pb2.ModelInferRequest(
        ).InferRequestedOutputTensor()
        self._output.name = name
        self._output.data_format = data_format

    @property
    def name(self):
        """Get the name of output associated with this object.

        Returns
        -------
        str
            The name of output
        """
        return self._output.name

    @property
    def data_format(self):
        """Get the requested format of ouput associated with this object.

        Returns
        -------
        str
            The data format in which output will be requested
        """
        return self._output.data_format

    @data_format.setter
    def data_format(self, data_format):
        """Set the requested format of ouput associated with this object.

        Parameter
        ---------
        data_format : str
            The data format in which output will be requested.
        """
        self._output.data_format = data_format

    # FIXMEPV2: Add parameter support
    @property
    def parameters(self):
        raise_error("Not implemented yet")

    def _get_tensor(self):
        """Retrieve the underlying InferRequestedOutputTensor message.
        Returns
        -------
        Protobuf Message 
            The underlying InferRequestedOutputTensor protobuf message.
        """
        return self._output


class InferResult:
    """An object of InferResult class holds the response of
    an inference request and provide methods to retrieve
    inference results.

    Parameters
    ----------
    result : Protobuf Message
        The ModelInferResponse returned by the server
    """

    def __init__(self, result):
        self._result = result

    def as_numpy(self, name):
        """Get the output tensor data (datatype, shape, contents) for
        output associated with this object in numpy format

        Parameters
        ----------
        name : string
            The name of the output tensor whose result is to be retrieved.
    
        Returns
        -------
        numpy array
            The numpy array containing the response data for the tensor or
            None if the data for specified tensor name is not found.
        """
        for self._output in self._result.outputs:
            if self._output.name == name:
                self._shape = []
                for self._value in self._output.shape:
                    self._shape.append(self._value)

                # FIXMEPV2 datatype is not yet provided by the server
                # for ouput yet. hard-coding to INT32
                self._datatype = 'INT32'
                if self._datatype == 'STRING':
                    # String results contain a 4-byte string length
                    # followed by the actual string characters. Hence,
                    # need to decode the raw bytes to convert into
                    # array elements.
                    self._np_array = deserialize_string_tensor(
                        self._output.contents.raw_contents)
                else:
                    self._np_array = np.frombuffer(
                        self._output.contents.raw_contents,
                        dtype=trtis_to_np_dtype(self._datatype))
                self._np_array = np.resize(self._np_array, self._shape)
                return self._np_array
        return None

    def get_request(self, as_json=False):
        """Retrieves the ModelInferRequest for the request associated
        with this response as a json dict object or protobuf message

        Parameters
        ----------
        as_json : bool
            If true then returns request as a json dict, otherwise
            as a protobuf message. Default value is False.

        Returns
        -------
        Protobuf Message or dict
            The ModelInferRequest protobuf message or dict for the request
            associated  with this response.
        """
        if as_json:
            return json.loads(MessageToJson(self._result.request))
        else:
            return self._result.request

    def get_statistics(self, as_json=False):
        """Retrieves the InferStatistics for this response as
        a json dict object or protobuf message

        Parameters
        ----------
        as_json : bool
            If true then returns statistics as a json dict, otherwise
            as a protobuf message. Default value is False.
        
        Returns
        -------
        Protobuf Message or dict
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
            If true then returns response as a json dict, otherwise
            as a protobuf message. Default value is False.
    
        Returns
        -------
        Protobuf Message or dict
            The underlying ModelInferResponse as a protobuf message or dict.
        """
        if as_json:
            return json.loads(MessageToJson(self._result))
        else:
            return self._result
