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

"""Class for receiving inference responses to Triton Inference Server Models"""

from __future__ import annotations

import asyncio
import inspect
import queue
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Optional

from tritonserver._api import _model

if TYPE_CHECKING:
    from tritonserver._api._model import Model

from tritonserver._api._logging import LogMessage
from tritonserver._api._tensor import Tensor
from tritonserver._c.triton_bindings import (
    InternalError,
    TritonError,
    TRITONSERVER_InferenceRequest,
)
from tritonserver._c.triton_bindings import TRITONSERVER_LogLevel as LogLevel
from tritonserver._c.triton_bindings import (
    TRITONSERVER_ResponseCompleteFlag,
    TRITONSERVER_Server,
)


class AsyncResponseIterator:

    """Asyncio compatible response iterator

    Response iterators are returned from model inference methods and
    allow users to process inference responses in the order they were
    received for a request.

    """

    def __init__(
        self,
        model: _model.Model,
        request: TRITONSERVER_InferenceRequest,
        user_queue: Optional[asyncio.Queue] = None,
        raise_on_error: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        """Initialize AsyncResponseIterator

        AsyncResponseIterator objects are obtained from Model inference
        methods and not instantiated directly. See `Model` documentation.

        Parameters
        ----------
        model : Model
            model associated with inference request
        request : TRITONSERVER_InferenceRequest
            Underlying C binding TRITONSERVER_InferenceRequest
            object. Private.
        user_queue : Optional[asyncio.Queue]
            Optional user queue for responses in addition to internal
            iterator queue.
        raise_on_error : bool
            if True response errors will be raised as exceptions.
        loop : Optional[asyncio.AbstractEventLoop]
            asyncio loop object

        """

        if loop is None:
            loop = asyncio.get_running_loop()
        self._loop = loop
        self._queue = asyncio.Queue()
        self._user_queue = user_queue
        self._complete = False
        self._request = request
        self._model = model
        self._raise_on_error = raise_on_error

    def __aiter__(self) -> AsyncResponseIterator:
        """Return async iterator. For use with async for loops.

        Returns
        -------
        AsyncResponseIterator

        Examples
        --------

        responses = server.model("test").async_infer(inputs={"fp16_input":numpy.array([[1]],dtype=numpy.float16)})
        async for response in responses:
            print(nummpy.from_dlpack(response.outputs["fp16_output"]))


        """

        return self

    async def __anext__(self):
        """Returns the next response received for a request

        Returns the next response received for a request as an
        awaitable object.

        Raises
        ------
        response.error
            If raise_on_error is set to True, response errors are
            raised as exceptions
        StopAsyncIteration
            Indicates all responses for a request have been received.
            Final responses may or may not include outputs and must be
            checked.

        """

        if self._complete:
            raise StopAsyncIteration
        response = await self._queue.get()
        self._complete = response.final
        if response.error is not None and self._raise_on_error:
            raise response.error
        return response

    def cancel(self) -> None:
        """Cancel an inflight request

        Cancels an in-flight request. Cancellation is handled on a
        best effort basis and may not prevent execution of a request
        if it is already started or completed.

        See c:func:`TRITONSERVER_ServerInferenceRequestCancel`

        Examples
        --------

        responses = server.model("test").infer(inputs={"text_input":["hello"]})

        responses.cancel()

        """

        if self._request is not None:
            self._request.cancel()

    def _response_callback(self, response, flags, unused):
        try:
            if self._request is None:
                raise InternalError("Response received after final response flag")

            response = InferenceResponse._from_tritonserver_inference_response(
                self._model, self._request, response, flags
            )
            asyncio.run_coroutine_threadsafe(self._queue.put(response), self._loop)
            if self._user_queue is not None:
                asyncio.run_coroutine_threadsafe(
                    self._user_queue.put(response), self._loop
                )
            if flags == TRITONSERVER_ResponseCompleteFlag.FINAL:
                del self._request
                self._request = None
        except Exception as e:
            message = f"Catastrophic failure in response callback: {e}"
            LogMessage(LogLevel.ERROR, message)
            # catastrophic failure
            raise e from None


class ResponseIterator:
    """Response iterator

    Response iterators are returned from model inference methods and
    allow users to process inference responses in the order they were
    received for a request.

    """

    def __init__(
        self,
        model: Model,
        request: TRITONSERVER_InferenceRequest,
        user_queue: Optional[queue.SimpleQueue] = None,
        raise_on_error: bool = False,
    ):
        """Initialize ResponseIterator

        ResponseIterator objects are obtained from Model inference
        methods and not instantiated directly. See `Model` documentation.

        Parameters
        ----------
        model : Model
            model associated with inference request
        request : TRITONSERVER_InferenceRequest
            Underlying C binding TRITONSERVER_InferenceRequest
            object. Private.
        user_queue : Optional[asyncio.Queue]
            Optional user queue for responses in addition to internal
            iterator queue.
        raise_on_error : bool
            if True response errors will be raised as exceptions.

        """

        self._queue = queue.SimpleQueue()
        self._user_queue = user_queue
        self._complete = False
        self._request = request
        self._model = model
        self._raise_on_error = raise_on_error

    def __iter__(self) -> ResponseIterator:
        """Return response iterator.

        Returns
        -------
        ResponseIterator

        Examples
        --------

        responses = server.model("test").infer(inputs={"fp16_input":numpy.array([[1]],dtype=numpy.float16)})
        for response in responses:
           print(nummpy.from_dlpack(response.outputs["fp16_output"]))


        """

        return self

    def __next__(self):
        """Returns the next response received for a request

        Raises
        ------
        response.error
            If raise_on_error is set to True, response errors are
            raised as exceptions
        StopIteration
            Indicates all responses for a request have been received.
            Final responses may or may not include outputs and must be
            checked.

        """

        if self._complete:
            raise StopIteration
        response = self._queue.get()
        self._complete = response.final
        if response.error is not None and self._raise_on_error:
            raise response.error
        return response

    def cancel(self):
        """Cancel an inflight request

        Cancels an in-flight request. Cancellation is handled on a
        best effort basis and may not prevent execution of a request
        if it is already started or completed.

        See c:func:`TRITONSERVER_ServerInferenceRequestCancel`

        Examples
        --------

        responses = server.model("test").infer(inputs={"text_input":["hello"]})

        responses.cancel()

        """
        if self._request is not None:
            self._request.cancel()

    def _response_callback(self, response, flags, unused):
        try:
            if self._request is None:
                raise InternalError("Response received after final response flag")

            response = InferenceResponse._from_tritonserver_inference_response(
                self._model, self._request, response, flags
            )
            self._queue.put(response)
            if self._user_queue is not None:
                self._user_queue.put(response)
            if flags == TRITONSERVER_ResponseCompleteFlag.FINAL:
                del self._request
                self._request = None
        except Exception as e:
            message = f"Catastrophic failure in response callback: {e}"
            LogMessage(LogLevel.ERROR, message)
            # catastrophic failure
            raise e from None


@dataclass
class InferenceResponse:
    """Dataclass representing an inference response.

    Inference response objects are returned from response iterators
    which are in turn returned from model inference methods. They
    contain output data, output parameters, any potential errors
    reported and a flag to indicate if the response is the final one
    for a request.

    See c:func:`TRITONSERVER_InferenceResponse` for more details

    Parameters
    ----------
    model : Model
        Model instance associated with the response.
    request_id : Optional[str], default None
        Unique identifier for the inference request (if provided)
    parameters : dict[str, str | int | bool], default {}
        Additional parameters associated with the response.
    outputs : dict [str, Tensor], default {}
        Output tensors for the inference.
    error : Optional[TritonError], default None
        Error (if any) that occurred in the processing of the request.
    classification_label : Optional[str], default None
        Classification label associated with the inference. Not currently supported.
    final : bool, default False
        Flag indicating if the response is final

    """

    model: _model.Model
    request_id: Optional[str] = None
    parameters: dict[str, str | int | bool] = field(default_factory=dict)
    outputs: dict[str, Tensor] = field(default_factory=dict)
    error: Optional[TritonError] = None
    classification_label: Optional[str] = None
    final: bool = False

    @staticmethod
    def _from_tritonserver_inference_response(
        model: _model.Model,
        request: TRITONSERVER_InferenceRequest,
        response,
        flags: TRITONSERVER_ResponseCompleteFlag,
    ):
        result = InferenceResponse(
            model,
            request.id,
            final=(flags == TRITONSERVER_ResponseCompleteFlag.FINAL),
        )

        try:
            if response is None:
                return result

            try:
                response.throw_if_response_error()
            except TritonError as error:
                error.args += (result,)
                result.error = error

            name, version = response.model
            result.model.name = name
            result.model.version = version
            result.request_id = response.id
            parameters = {}
            for parameter_index in range(response.parameter_count):
                name, type_, value = response.parameter(parameter_index)
                parameters[name] = value
            result.parameters = parameters
            outputs = {}
            for output_index in range(response.output_count):
                (
                    name,
                    data_type,
                    shape,
                    _data_ptr,
                    _byte_size,
                    _memory_type,
                    _memory_type_id,
                    memory_buffer,
                ) = response.output(output_index)
                tensor = Tensor(data_type, shape, memory_buffer)

                outputs[name] = tensor
            result.outputs = outputs
        except Exception as e:
            error = InternalError(f"Unexpected error in creating response object: {e}")
            error.args += (result,)
            result.error = error

        # TODO: support classification
        # values["classification_label"] = response.output_classification_label()

        return result
