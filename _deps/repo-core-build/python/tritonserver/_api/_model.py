# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Class for interacting with Triton Inference Server Models"""

from __future__ import annotations

import asyncio
import json
import queue
from typing import TYPE_CHECKING, Any, Optional

from tritonserver._api._allocators import ResponseAllocator
from tritonserver._api._request import InferenceRequest
from tritonserver._api._response import AsyncResponseIterator, ResponseIterator
from tritonserver._c.triton_bindings import InvalidArgumentError
from tritonserver._c.triton_bindings import (
    TRITONSERVER_ModelBatchFlag as ModelBatchFlag,
)
from tritonserver._c.triton_bindings import (
    TRITONSERVER_ModelTxnPropertyFlag as ModelTxnPropertyFlag,
)
from tritonserver._c.triton_bindings import TRITONSERVER_Server

if TYPE_CHECKING:
    from tritonserver._api._server import Server


class Model:
    """Class for interacting with Triton Inference Server models

    Model objects are returned from server factory methods and allow
    users to query metadata and execute inference
    requests.

    """

    def __init__(
        self,
        server: Server,
        name: str,
        version: int = -1,
        state: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Initialize model

        Model objects should be obtainted from Server factory methods
        and not instantiated directly. See `Server` documentation.

        Parameters
        ----------
        server : Server
            Server instance.
        name : str
            model name
        version : int
            model version
        state : Optional[str]
            state of model (if known)
        reason : Optional[str]
            reason for model state (if known)

        Examples
        --------
        >>> server.model("test")
        server.model("test")
        {'name': 'test', 'version': -1, 'state': None}

        """

        self._server = server._server
        if not isinstance(self._server, TRITONSERVER_Server):
            raise InvalidArgumentError("Server not started")
        self.name = name
        self.version = version
        self.state = state
        self.reason = reason

    def create_request(self, **kwargs: Unpack[InferenceRequest]) -> InferenceRequest:
        """Inference request factory method

        Return an inference request object that can be used with
        model.infer() ro model.async_infer()

        Parameters
        ----------

        kwargs : Unpack[InferenceRequest]
            Keyword arguments passed to `InferenceRequest` constructor. See
            `InferenceRequest` documentation for details.

        Returns
        -------
        InferenceRequest
            Inference request associated with this model

        Examples
        --------

        >>> server.model("test").create_request()
        server.model("test").create_request()
        InferenceRequest(model={'name': 'test', 'version': -1,
        'state': None},
        _server=<tritonserver._c.triton_bindings.TRITONSERVER_Server
        object at 0x7f5827156bf0>, request_id=None, flags=0,
        correlation_id=None, priority=0, timeout=0, inputs={},
        parameters={}, output_memory_type=None,
        output_memory_allocator=None, response_queue=None,
        _serialized_inputs={})

        """
        if "model" in kwargs:
            kwargs.pop("model")
        return InferenceRequest(model=self, **kwargs)

    def async_infer(
        self,
        inference_request: Optional[InferenceRequest] = None,
        raise_on_error: bool = True,
        **kwargs: Unpack[InferenceRequest],
    ) -> AsyncResponseIterator:
        """Send an inference request to the model for execution

        Sends an inference request to the model. Responses are
        returned using an asyncio compatible iterator. See
        c:func:`TRITONSERVER_ServerInferAsync`

        Parameters
        ----------
        inference_request : Optional[InferenceRequest]
            inference request object. If not provided inference
            request will be created using remaining keyword
            arguments.
        raise_on_error : bool, default True
            if True iterator will raise an error on any response
            errors returned from the model. If False errors will be
            returned as part of the response object.
        kwargs : Unpack[InferenceRequest]
            If a request object is not provided, a new object will be
            created with remaining keyword arguments. See
            `InferenceRequest` documentation for valid arguments.


        Returns
        -------
        AsyncResponseIterator
            asyncio compatible iterator

        Raises
        ------
        InvalidArgumentError
            if any invalid arguments are provided

        """

        if inference_request is None:
            inference_request = InferenceRequest(model=self, **kwargs)

        if (inference_request.response_queue is not None) and (
            not isinstance(inference_request.response_queue, asyncio.Queue)
        ):
            raise InvalidArgumentError(
                "asyncio.Queue must be used for async response iterator"
            )

        request = inference_request._create_tritonserver_inference_request()

        response_iterator = AsyncResponseIterator(
            self,
            request,
            inference_request.response_queue,
            raise_on_error,
        )

        response_allocator = ResponseAllocator(
            inference_request.output_memory_allocator,
            inference_request.output_memory_type,
        ).create_tritonserver_response_allocator()

        request.set_response_callback(
            response_allocator, None, response_iterator._response_callback, None
        )

        self._server.infer_async(request)
        return response_iterator

    def infer(
        self,
        inference_request: Optional[InferenceRequest] = None,
        raise_on_error: bool = True,
        **kwargs: Unpack[InferenceRequest],
    ) -> ResponseIterator:
        """Send an inference request to the model for execution

        Sends an inference request to the model. Responses are
        returned asynchronously using an iterator. See
        c:func:`TRITONSERVER_ServerInferAsync`


        Parameters
        ----------
        inference_request : Optional[InferenceRequest]
            inference request object. If not provided inference
            request will be created using remaining keyword
            arguments.
        raise_on_error : bool, default True
            if True iterator will raise an error on any response
            errors returned from the model. If False errors will be
            returned as part of the response object.
        kwargs : Unpack[InferenceRequest]
            If a request object is not provided, a new object will be
            created with remaining keyword arguments. See
            `InferenceRequest` documentation for valid arguments.

        Returns
        -------
        ResponseIterator
            Response iterator

        Raises
        ------
        InvalidArgumentError
            if any invalid arguments are provided

        Examples
        --------
        >>> responses = server.model("test_2").infer(inputs={"text_input":["hello"]})
        responses = list(server.model("test_2").infer(inputs={"text_input":["hello"]}))

        >>> response = responses[0]
        print(response)
        InferenceResponse(model={'name': 'test_2', 'version': 1,
        'state': None},
        _server=<tritonserver._c.triton_bindings.TRITONSERVER_Server
        object at 0x7f5827156bf0>, request_id='', parameters={},
        outputs={'text_output':
        Tensor(data_type=<TRITONSERVER_DataType.BYTES: 13>,
        shape=array([1]),
        memory_buffer=MemoryBuffer(data_ptr=140003384498080,
        memory_type=<TRITONSERVER_MemoryType.CPU: 0>,
        memory_type_id=0, size=9, owner=array([ 5, 0, 0, 0, 104, 101,
        108, 108, 111], dtype=int8)))}, error=None,
        classification_label=None, final=False)

        >>> response.outputs["text_output"].to_bytes_array()
        response.outputs["text_output"].to_bytes_array()
        array([b'hello'], dtype=object)

        """

        if inference_request is None:
            inference_request = InferenceRequest(model=self, **kwargs)

        if (inference_request.response_queue is not None) and (
            not isinstance(inference_request.response_queue, queue.SimpleQueue)
        ):
            raise InvalidArgumentError(
                "queue.SimpleQueue must be used for response iterator"
            )

        request = inference_request._create_tritonserver_inference_request()
        response_iterator = ResponseIterator(
            self,
            request,
            inference_request.response_queue,
            raise_on_error,
        )
        response_allocator = ResponseAllocator(
            inference_request.output_memory_allocator,
            inference_request.output_memory_type,
        ).create_tritonserver_response_allocator()

        request.set_response_callback(
            response_allocator, None, response_iterator._response_callback, None
        )

        self._server.infer_async(request)
        return response_iterator

    def metadata(self) -> dict[str, Any]:
        """Returns medatadata about a model and its inputs and outputs

        See c:func:`TRITONSERVER_ServerModelMetadata()`

        Returns
        -------
        dict[str, Any]
            Model metadata as a dictionary of key value pairs

        Examples
        --------
        server.model("test").metadata()
        {'name': 'test', 'versions': ['1'], 'platform': 'python',
        'inputs': [{'name': 'text_input', 'datatype': 'BYTES',
        'shape': [-1]}, {'name': 'fp16_input', 'datatype': 'FP16',
        'shape': [-1, 1]}], 'outputs': [{'name': 'text_output',
        'datatype': 'BYTES', 'shape': [-1]}, {'name': 'fp16_output',
        'datatype': 'FP16', 'shape': [-1, 1]}]}

        """

        return json.loads(
            self._server.model_metadata(self.name, self.version).serialize_to_json()
        )

    def ready(self) -> bool:
        """Returns whether a model is ready to accept requests

        See :c:func:`TRITONSERVER_ServerModelIsReady()`

        Returns
        -------
        bool
            True if model is ready. False otherwise.

        Examples
        --------
        >>> server.model("test").ready()
        server.model("test").ready()
        True

        """

        return self._server.model_is_ready(self.name, self.version)

    def batch_properties(self) -> ModelBatchFlag:
        """Returns the batch properties of the model

        See :c:func:`TRITONSERVER_ServerModelBatchProperties`

        Returns
        -------
        ModelBatchFlag
            ModelBatchFlag.UNKNOWN or ModelBatchFlag.FIRST_DIM

        Examples
        --------
        >>> server.model("resnet50_libtorch").batch_properties()
        server.model("resnet50_libtorch").batch_properties()
        <TRITONSERVER_ModelBatchFlag.FIRST_DIM: 2>

        """

        flags, _ = self._server.model_batch_properties(self.name, self.version)
        return ModelBatchFlag(flags)

    def transaction_properties(self) -> ModelTxnPropertyFlag:
        """Returns the transaction properties of the model

        See :c:func:`TRITONSERVER_ServerModelTransactionProperties`

        Returns
        -------
        ModelTxnPropertyFlag
            ModelTxnPropertyFlag.ONE_TO_ONE or
            ModelTxnPropertyFlag.DECOUPLED

        Examples
        --------
        >>> server.model("resnet50_libtorch").transaction_properties()
        server.model("resnet50_libtorch").transaction_properties()
        <TRITONSERVER_ModelTxnPropertyFlag.ONE_TO_ONE: 1>

        """

        txn_properties, _ = self._server.model_transaction_properties(
            self.name, self.version
        )
        return ModelTxnPropertyFlag(txn_properties)

    def statistics(self) -> dict[str, Any]:
        """Returns model statistics

        See :c:func:`TRITONSERVER_ServerModelStatistics`

        Returns
        -------
        dict[str, Any]
            Dictionary of key value pairs representing model
            statistics

        Examples
        --------
        >>> server.model("test").statistics()
        server.model("test").statistics()

        {'model_stats': [{'name':
        'test', 'version': '1', 'last_inference': 1704731597736,
        'inference_count': 2, 'execution_count': 2, 'inference_stats':
        {'success': {'count': 2, 'ns': 3079473}, 'fail': {'count': 0, 'ns':
        0}, 'queue': {'count': 2, 'ns': 145165}, 'compute_input': {'count': 2,
        'ns': 124645}, 'compute_infer': {'count': 2, 'ns': 2791809},
        'compute_output': {'count': 2, 'ns': 10240}, 'cache_hit': {'count': 0,
        'ns': 0}, 'cache_miss': {'count': 0, 'ns': 0}}, 'batch_stats':
        [{'batch_size': 1, 'compute_input': {'count': 2, 'ns': 124645},
        'compute_infer': {'count': 2, 'ns': 2791809}, 'compute_output':
        {'count': 2, 'ns': 10240}}], 'memory_usage': []}]}

        """

        return json.loads(
            self._server.model_statistics(self.name, self.version).serialize_to_json()
        )

    def config(self, config_version: int = 1) -> dict[str, Any]:
        """Returns model configuration

        See :c:func:`TRITONSERVER_ServerModelConfiguration`

        Parameters
        ----------
        config_version : int
            configuration version in case multiple are supported

        Returns
        -------
        dict[str, Any]
            Dictionary of key value pairs for model configuration

        Examples
        --------

        >>> server.model("test").config()
        server.model("test").config()

        {'name': 'test', 'platform':
        '', 'backend': 'python', 'version_policy': {'latest':
        {'num_versions': 1}}, 'max_batch_size': 0, 'input': [{'name':
        'text_input', 'data_type': 'TYPE_STRING', 'format':
        'FORMAT_NONE', 'dims': [-1], 'is_shape_tensor': False,
        'allow_ragged_batch': False, 'optional': True}, {'name':
        'fp16_input', 'data_type': 'TYPE_FP16', 'format':
        'FORMAT_NONE', 'dims': [-1, 1], 'is_shape_tensor': False,
        'allow_ragged_batch': False, 'optional': True}], 'output':
        [{'name': 'text_output', 'data_type': 'TYPE_STRING', 'dims':
        [-1], 'label_filename': '', 'is_shape_tensor': False},
        {'name': 'fp16_output', 'data_type': 'TYPE_FP16', 'dims': [-1,
        1], 'label_filename': '', 'is_shape_tensor': False}],
        'batch_input': [], 'batch_output': [], 'optimization':
        {'priority': 'PRIORITY_DEFAULT', 'input_pinned_memory':
        {'enable': True}, 'output_pinned_memory': {'enable': True},
        'gather_kernel_buffer_threshold': 0, 'eager_batching': False},
        'instance_group': [{'name': 'test_2', 'kind': 'KIND_GPU',
        'count': 1, 'gpus': [0], 'secondary_devices': [], 'profile':
        [], 'passive': False, 'host_policy': ''}],
        'default_model_filename': 'model.py', 'cc_model_filenames':
        {}, 'metric_tags': {}, 'parameters': {}, 'model_warmup': [],
        'model_transaction_policy': {'decoupled': True}}
        """

        return json.loads(
            self._server.model_config(
                self.name, self.version, config_version
            ).serialize_to_json()
        )

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "%s" % (
            {"name": self.name, "version": self.version, "state": self.state}
        )
