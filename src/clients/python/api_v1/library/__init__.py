# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

from builtins import range
from enum import IntEnum
from functools import partial
from future.utils import iteritems
from ctypes import *
import numpy as np
from numpy.ctypeslib import ndpointer
import pkg_resources
import struct
import threading
from google.protobuf import text_format
import tensorrtserver.api.model_config_pb2
from tensorrtserver.api.server_status_pb2 import ModelRepositoryIndex
from tensorrtserver.api.server_status_pb2 import ServerStatus
from tensorrtserver.api.server_status_pb2 import SharedMemoryStatus
from tensorrtserver.api.api_pb2 import *

class _utf8(object):
    @classmethod
    def from_param(cls, value):
        if value is None:
            return None
        elif isinstance(value, bytes):
            return value
        else:
            return value.encode('utf8')

import os
_request_lib = "request" if os.name == 'nt' else 'librequest.so'
_crequest_lib = "crequest" if os.name == 'nt' else 'libcrequest.so'
_request_path = pkg_resources.resource_filename('tensorrtserver.api', _request_lib)
_request = cdll.LoadLibrary(_request_path)
_crequest_path = pkg_resources.resource_filename('tensorrtserver.api', _crequest_lib)
_crequest = cdll.LoadLibrary(_crequest_path)

_crequest_error_new = _crequest.ErrorNew
_crequest_error_new.restype = c_void_p
_crequest_error_new.argtypes = [_utf8]
_crequest_error_del = _crequest.ErrorDelete
_crequest_error_del.argtypes = [c_void_p]
_crequest_error_isok = _crequest.ErrorIsOk
_crequest_error_isok.restype = c_bool
_crequest_error_isok.argtypes = [c_void_p]
_crequest_error_msg = _crequest.ErrorMessage
_crequest_error_msg.restype = c_char_p
_crequest_error_msg.argtypes = [c_void_p]
_crequest_error_serverid = _crequest.ErrorServerId
_crequest_error_serverid.restype = c_char_p
_crequest_error_serverid.argtypes = [c_void_p]
_crequest_error_requestid = _crequest.ErrorRequestId
_crequest_error_requestid.restype = c_int64
_crequest_error_requestid.argtypes = [c_void_p]

_crequest_health_ctx_new = _crequest.ServerHealthContextNew
_crequest_health_ctx_new.restype = c_void_p
_crequest_health_ctx_new.argtypes = [POINTER(c_void_p), _utf8, c_int,
                                     POINTER(c_char_p), c_int, c_bool]
_crequest_health_ctx_del = _crequest.ServerHealthContextDelete
_crequest_health_ctx_del.argtypes = [c_void_p]
_crequest_health_ctx_ready = _crequest.ServerHealthContextGetReady
_crequest_health_ctx_ready.restype = c_void_p
_crequest_health_ctx_ready.argtypes = [c_void_p, POINTER(c_bool)]
_crequest_health_ctx_live = _crequest.ServerHealthContextGetLive
_crequest_health_ctx_live.restype = c_void_p
_crequest_health_ctx_live.argtypes = [c_void_p, POINTER(c_bool)]

_crequest_status_ctx_new = _crequest.ServerStatusContextNew
_crequest_status_ctx_new.restype = c_void_p
_crequest_status_ctx_new.argtypes = [POINTER(c_void_p), _utf8, c_int,
                                     POINTER(c_char_p), c_int, _utf8, c_bool]
_crequest_status_ctx_del = _crequest.ServerStatusContextDelete
_crequest_status_ctx_del.argtypes = [c_void_p]
_crequest_status_ctx_get = _crequest.ServerStatusContextGetServerStatus
_crequest_status_ctx_get.restype = c_void_p
_crequest_status_ctx_get.argtypes = [c_void_p, POINTER(c_char_p), POINTER(c_uint32)]

_crequest_repository_ctx_new = _crequest.ModelRepositoryContextNew
_crequest_repository_ctx_new.restype = c_void_p
_crequest_repository_ctx_new.argtypes = [POINTER(c_void_p), _utf8, c_int,
                                         POINTER(c_char_p), c_int, c_bool]
_crequest_repository_ctx_del = _crequest.ModelRepositoryContextDelete
_crequest_repository_ctx_del.argtypes = [c_void_p]
_crequest_repository_ctx_get = _crequest.ModelRepositoryContextGetModelRepositoryIndex
_crequest_repository_ctx_get.restype = c_void_p
_crequest_repository_ctx_get.argtypes = [c_void_p, POINTER(c_char_p), POINTER(c_uint32)]

_crequest_model_control_ctx_new = _crequest.ModelControlContextNew
_crequest_model_control_ctx_new.restype = c_void_p
_crequest_model_control_ctx_new.argtypes = [POINTER(c_void_p), _utf8, c_int,
                                     POINTER(c_char_p), c_int, c_bool]
_crequest_model_control_ctx_del = _crequest.ModelControlContextDelete
_crequest_model_control_ctx_del.argtypes = [c_void_p]
_crequest_model_control_ctx_load = _crequest.ModelControlContextLoad
_crequest_model_control_ctx_load.restype = c_void_p
_crequest_model_control_ctx_load.argtypes = [c_void_p, _utf8]
_crequest_model_control_ctx_unload = _crequest.ModelControlContextUnload
_crequest_model_control_ctx_unload.restype = c_void_p
_crequest_model_control_ctx_unload.argtypes = [c_void_p, _utf8]

if os.name != 'nt':
    _crequest_shm_control_ctx_new = _crequest.SharedMemoryControlContextNew
    _crequest_shm_control_ctx_new.restype = c_void_p
    _crequest_shm_control_ctx_new.argtypes = [POINTER(c_void_p), _utf8, c_int, c_bool]
    _crequest_shm_control_ctx_del = _crequest.SharedMemoryControlContextDelete
    _crequest_shm_control_ctx_del.argtypes = [c_void_p]
    _crequest_shm_control_ctx_register = _crequest.SharedMemoryControlContextRegister
    _crequest_shm_control_ctx_register.restype = c_void_p
    _crequest_shm_control_ctx_register.argtypes = [c_void_p, c_void_p]
    _crequest_shm_control_ctx_cuda_register = _crequest.SharedMemoryControlContextCudaRegister
    _crequest_shm_control_ctx_cuda_register.restype = c_void_p
    _crequest_shm_control_ctx_cuda_register.argtypes = [c_void_p, c_void_p]
    _crequest_shm_control_ctx_unregister = _crequest.SharedMemoryControlContextUnregister
    _crequest_shm_control_ctx_unregister.restype = c_void_p
    _crequest_shm_control_ctx_unregister.argtypes = [c_void_p, c_void_p]
    _crequest_shm_control_ctx_unregister_all = _crequest.SharedMemoryControlContextUnregisterAll
    _crequest_shm_control_ctx_unregister_all.restype = c_void_p
    _crequest_shm_control_ctx_unregister_all.argtypes = [c_void_p]
    _crequest_shm_control_ctx_get_status = _crequest.SharedMemoryControlContextGetStatus
    _crequest_shm_control_ctx_get_status.restype = c_void_p
    _crequest_shm_control_ctx_get_status.argtypes =  [c_void_p, POINTER(c_char_p), POINTER(c_uint32)]

_crequest_infer_ctx_new = _crequest.InferContextNew
_crequest_infer_ctx_new.restype = c_void_p
_crequest_infer_ctx_new.argtypes = [POINTER(c_void_p), _utf8, c_int,
                                    POINTER(c_char_p), c_int, _utf8, c_int64,
                                    c_uint64, c_bool, c_bool]
_crequest_infer_ctx_del = _crequest.InferContextDelete
_crequest_infer_ctx_del.argtypes = [c_void_p]
_crequest_infer_ctx_set_options = _crequest.InferContextSetOptions
_crequest_infer_ctx_set_options.restype = c_void_p
_crequest_infer_ctx_set_options.argtypes = [c_void_p, c_void_p]
_crequest_infer_ctx_run = _crequest.InferContextRun
_crequest_infer_ctx_run.restype = c_void_p
_crequest_infer_ctx_run.argtypes = [c_void_p]
_async_run_callback_prototype = CFUNCTYPE(None, c_void_p, c_uint64)
_crequest_infer_ctx_async_run = _crequest.InferContextAsyncRun
_crequest_infer_ctx_async_run.restype = c_void_p
_crequest_infer_ctx_async_run.argtypes = [c_void_p, _async_run_callback_prototype]
_crequest_infer_ctx_get_async_run_results = _crequest.InferContextGetAsyncRunResults
_crequest_infer_ctx_get_async_run_results.restype = c_void_p
_crequest_infer_ctx_get_async_run_results.argtypes = [c_void_p, c_uint64]

_crequest_infer_ctx_options_new = _crequest.InferContextOptionsNew
_crequest_infer_ctx_options_new.restype = c_void_p
_crequest_infer_ctx_options_new.argtypes = [POINTER(c_void_p), c_uint32, c_uint64, c_uint64, c_uint32, c_uint64]
_crequest_infer_ctx_options_del = _crequest.InferContextOptionsDelete
_crequest_infer_ctx_options_del.argtypes = [c_void_p]
_crequest_infer_ctx_options_add_raw = _crequest.InferContextOptionsAddRaw
_crequest_infer_ctx_options_add_raw.restype = c_void_p
_crequest_infer_ctx_options_add_raw.argtypes = [c_void_p, c_void_p, _utf8]
_crequest_infer_ctx_options_add_class = _crequest.InferContextOptionsAddClass
_crequest_infer_ctx_options_add_class.restype = c_void_p
_crequest_infer_ctx_options_add_class.argtypes = [c_void_p, c_void_p, _utf8, c_uint64]
_crequest_infer_ctx_options_add_shared_memory = _crequest.InferContextOptionsAddSharedMemory
_crequest_infer_ctx_options_add_shared_memory.restype = c_void_p
_crequest_infer_ctx_options_add_shared_memory.argtypes = [c_void_p, c_void_p, _utf8, c_void_p]
_crequest_correlation_id = _crequest.CorrelationId
_crequest_correlation_id.restype = c_uint64
_crequest_correlation_id.argtypes = [c_void_p]

_crequest_infer_ctx_input_new = _crequest.InferContextInputNew
_crequest_infer_ctx_input_new.restype = c_void_p
_crequest_infer_ctx_input_new.argtypes = [POINTER(c_void_p), c_void_p, _utf8]
_crequest_infer_ctx_input_del = _crequest.InferContextInputDelete
_crequest_infer_ctx_input_del.argtypes = [c_void_p]
_crequest_infer_ctx_input_set_shape = _crequest.InferContextInputSetShape
_crequest_infer_ctx_input_set_shape.restype = c_void_p
_crequest_infer_ctx_input_set_shape.argtypes = [c_void_p,
                                                ndpointer(c_int64, flags="C_CONTIGUOUS"),
                                                c_uint64]
_crequest_infer_ctx_input_set_raw = _crequest.InferContextInputSetRaw
_crequest_infer_ctx_input_set_raw.restype = c_void_p
_crequest_infer_ctx_input_set_raw.argtypes = [c_void_p, c_void_p, c_uint64]

_crequest_infer_ctx_input_set_shared_memory = _crequest.InferContextInputSetSharedMemory
_crequest_infer_ctx_input_set_shared_memory.restype = c_void_p
_crequest_infer_ctx_input_set_shared_memory.argtypes = [c_void_p, c_void_p]

_crequest_infer_ctx_result_new = _crequest.InferContextResultNew
_crequest_infer_ctx_result_new.restype = c_void_p
_crequest_infer_ctx_result_new.argtypes = [POINTER(c_void_p), c_void_p, _utf8]
_crequest_infer_ctx_async_result_new = _crequest.InferContextAsyncResultNew
_crequest_infer_ctx_async_result_new.restype = c_void_p
_crequest_infer_ctx_async_result_new.argtypes = [POINTER(c_void_p), c_void_p, c_uint64, _utf8]
_crequest_infer_ctx_result_del = _crequest.InferContextResultDelete
_crequest_infer_ctx_result_del.argtypes = [c_void_p]
_crequest_infer_ctx_result_modelname = _crequest.InferContextResultModelName
_crequest_infer_ctx_result_modelname.restype = c_void_p
_crequest_infer_ctx_result_modelname.argtypes = [c_void_p, POINTER(c_char_p)]
_crequest_infer_ctx_result_modelver = _crequest.InferContextResultModelVersion
_crequest_infer_ctx_result_modelver.restype = c_void_p
_crequest_infer_ctx_result_modelver.argtypes = [c_void_p, POINTER(c_int64)]
_crequest_infer_ctx_result_dtype = _crequest.InferContextResultDataType
_crequest_infer_ctx_result_dtype.restype = c_void_p
_crequest_infer_ctx_result_dtype.argtypes = [c_void_p, POINTER(c_uint32)]
_crequest_infer_ctx_result_shape = _crequest.InferContextResultShape
_crequest_infer_ctx_result_shape.restype = c_void_p
_crequest_infer_ctx_result_shape.argtypes = [c_void_p, c_uint64,
                                            ndpointer(c_int64, flags="C_CONTIGUOUS"),
                                            POINTER(c_uint64)]
_crequest_infer_ctx_result_next_raw = _crequest.InferContextResultNextRaw
_crequest_infer_ctx_result_next_raw.restype = c_void_p
_crequest_infer_ctx_result_next_raw.argtypes = [c_void_p, c_uint64, POINTER(c_char_p),
                                                POINTER(c_uint64)]
_crequest_infer_ctx_result_class_cnt = _crequest.InferContextResultClassCount
_crequest_infer_ctx_result_class_cnt.restype = c_void_p
_crequest_infer_ctx_result_class_cnt.argtypes = [c_void_p, c_uint64, POINTER(c_uint64)]
_crequest_infer_ctx_result_next_class = _crequest.InferContextResultNextClass
_crequest_infer_ctx_result_next_class.restype = c_void_p
_crequest_infer_ctx_result_next_class.argtypes = [c_void_p, c_uint64, POINTER(c_uint64),
                                                  POINTER(c_float), POINTER(c_char_p)]
_crequest_get_shared_memory_handle_info = _crequest.SharedMemoryControlContextGetSharedMemoryHandleInfo
_crequest_get_shared_memory_handle_info.restype = c_void_p
_crequest_get_shared_memory_handle_info.argtypes = [c_void_p, POINTER(c_char_p), POINTER(c_char_p),
                                                    POINTER(c_int), POINTER(c_uint64), POINTER(c_uint64)]
_crequest_shared_memory_handle_release_buffer = _crequest.SharedMemoryControlContextReleaseBuffer
_crequest_shared_memory_handle_release_buffer.restype = c_void_p
_crequest_shared_memory_handle_release_buffer.argtypes = [c_void_p, c_char_p]

_crequest_infer_ctx_get_stat = _crequest.InferContextGetStat
_crequest_infer_ctx_get_stat.restype = c_void_p
_crequest_infer_ctx_get_stat.argtypes = [c_void_p, POINTER(c_uint64), POINTER(c_uint64),
                                        POINTER(c_uint64), POINTER(c_uint64)]

def _raise_if_error(err):
    """
    Raise InferenceServerException if 'err' is non-success.
    Otherwise return the request ID.
    """
    if err.value is not None:
        ex = InferenceServerException(err)
        isok = _crequest_error_isok(err)
        _crequest_error_del(err)
        if not isok:
            raise ex
        return ex.request_id()
    return 0

def _raise_error(msg):
    err = c_void_p(_crequest_error_new(msg))
    ex = InferenceServerException(err)
    _crequest_error_del(err)
    raise ex


def serialize_string_tensor(input_tensor):
    """
    Serializes a string tensor into a flat numpy array of length prepend strings.
    Can pass string tensor as numpy array of bytes with dtype of np.bytes_,
    numpy strings with dtype of np.str_ or python strings with dtype of np.object.

    Parameters
    ----------
    input_tensor : np.array
        The string tensor to serialize.

    Returns
    -------
    serialized_string_tensor : np.array
        The 1-D numpy array of type uint8 containing the serialized string in 'C' order.

    Raises
    ------
    InferenceServerException
        If unable to serialize the given tensor.
    """

    if not isinstance(input_tensor, (np.ndarray,)):
        _raise_error("input must be a numpy array")

    if input_tensor.size == 0:
        _raise_error("input cannot be empty")

    # If the input is a tensor of string objects, then must flatten those into
    # a 1-dimensional array containing the 4-byte string length followed by the
    # actual string characters. All strings are concatenated together in "C"
    # order.
    if (input_tensor.dtype == np.object) or (input_tensor.dtype.type == np.bytes_):
        flattened = bytes()
        for obj in np.nditer(input_tensor, flags=["refs_ok"], order='C'):
            # If directly passing bytes to STRING type,
            # don't convert it to str as Python will encode the
            # bytes which may distort the meaning
            if obj.dtype.type == np.bytes_:
                if type(obj.item()) == bytes:
                    s = obj.item()
                else:
                    s = bytes(obj)
            else:
                s = str(obj).encode('utf-8')
            flattened += struct.pack("<I", len(s))
            flattened += s
        return np.asarray(flattened)
    else:
        _raise_error("cannot serialize string tensor: invalid datatype")
    return None

class ProtocolType(IntEnum):
    """Protocol types supported by the client API

    HTTP
        The HTTP protocol.
    GRPC
        The GRPC protocol.

    """
    HTTP = 0
    GRPC = 1

    @classmethod
    def from_str(cls, value):
        """Convert a string to the corresponding ProtocolType.

        Parameters
        ----------
        value : str
            The string value to convert.

        Returns
        -------
        ProtocolType
            The ProtocolType corresponding to 'value'.

        Raises
        ------
        Exception
            If 'value' is an unknown protocol.

        """
        if value.lower() == 'http':
            return ProtocolType.HTTP
        elif value.lower() == 'grpc':
            return ProtocolType.GRPC
        raise Exception("unexpected protocol: " + value +
                        ", expecting HTTP or gRPC")
        return ProtocolType.HTTP

class InferenceServerException(Exception):
    """Exception indicating non-Success status.

    Parameters
    ----------
    err : c_void_p
        Pointer to an Error that should be used to initialize the exception.

    """
    def __init__(self, err):
        self._msg = None
        self._server_id = None
        self._request_id = 0
        if (err is not None) and (err.value is not None):
            self._msg = _crequest_error_msg(err)
            if self._msg is not None:
                self._msg = self._msg.decode('utf-8')
            self._server_id = _crequest_error_serverid(err)
            if self._server_id is not None:
                self._server_id = self._server_id.decode('utf-8')
            self._request_id = _crequest_error_requestid(err)

    def __str__(self):
        msg = super().__str__() if self._msg is None else self._msg
        if self._server_id is not None:
            msg = '[' + self._server_id + ' ' + str(self._request_id) + '] ' + msg
        return msg

    def message(self):
        """Get the exception message.

        Returns
        -------
        str
            The message associated with this exception, or None if no message.

        """
        return self._msg

    def server_id(self):
        """Get the ID of the server associated with this exception.

        Returns
        -------
        str
            The ID of the server associated with this exception, or
            None if no server is associated.

        """
        return self._server_id

    def request_id(self):
        """Get the ID of the request with this exception.

        Returns
        -------
        int
            The ID of the request associated with this exception, or
            0 (zero) if no request is associated.

        """
        return self._request_id

class ServerHealthContext:
    """Performs a health request to an inference server.

    Parameters
    ----------
    url : str
        The inference server URL, e.g. localhost:8000.

    protocol : ProtocolType
        The protocol used to communicate with the server.

    verbose : bool
        If True generate verbose output.

    http_headers : list of strings
        HTTP headers to send with request. Ignored for GRPC
        protocol. Each header must be specified as "Header:Value".

    """
    def __init__(self, url, protocol, verbose=False, http_headers=[]):
        self._last_request_id = 0
        self._ctx = c_void_p()

        b_http_headers = list()
        if http_headers is not None:
            for hh in http_headers:
                b_http_headers.append(hh.encode('utf-8'))

        http_headers_arr = (c_char_p * len(b_http_headers))()
        http_headers_arr[:] = b_http_headers

        _raise_if_error(
            c_void_p(
                _crequest_health_ctx_new(
                    byref(self._ctx), url, int(protocol),
                    http_headers_arr, len(b_http_headers), verbose)))

    def __del__(self):
        # when module is unloading may get called after
        # _crequest_health_ctx_del has been released
        if _crequest_health_ctx_del is not None:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        """Close the context. Any future calls to is_ready() or is_live() will
        result in an Error.

        """
        _crequest_health_ctx_del(self._ctx)
        self._ctx = None

    def is_ready(self):
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
        self._last_request_id = None
        if self._ctx is None:
            _raise_error("ServerHealthContext is closed")

        cready = c_bool()
        self._last_request_id = _raise_if_error(
            c_void_p(_crequest_health_ctx_ready(self._ctx, byref(cready))))
        return cready.value

    def is_live(self):
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
        self._last_request_id = None
        if self._ctx is None:
            _raise_error("ServerHealthContext is closed")

        clive = c_bool()
        self._last_request_id = _raise_if_error(
            c_void_p(_crequest_health_ctx_live(self._ctx, byref(clive))))
        return clive.value

    def get_last_request_id(self):
        """Get the request ID of the most recent is_ready() or is_live()
        request.

        Returns
        -------
        int
            The request ID, or None if a request has not yet been made
            or if the last request was not successful.

        """
        return self._last_request_id


class ServerStatusContext:
    """Performs a status request to an inference server.

    A request can be made to get status for the server and all models
    managed by the server, or to get status foronly a single model.

    Parameters
    ----------
    url : str
        The inference server URL, e.g. localhost:8000.

    protocol : ProtocolType
        The protocol used to communicate with the server.

    model_name : str
        The name of the model to get status for, or None to get status
        for all models managed by the server.

    verbose : bool
        If True generate verbose output.

    http_headers : list of strings
        HTTP headers to send with request. Ignored for GRPC
        protocol. Each header must be specified as "Header:Value".

    """
    def __init__(self, url, protocol, model_name=None, verbose=False, http_headers=[]):
        self._last_request_id = 0
        self._ctx = c_void_p()

        b_http_headers = list()
        if http_headers is not None:
            for hh in http_headers:
                b_http_headers.append(hh.encode('utf-8'))

        http_headers_arr = (c_char_p * len(b_http_headers))()
        http_headers_arr[:] = b_http_headers

        _raise_if_error(
            c_void_p(
                _crequest_status_ctx_new(
                    byref(self._ctx), url, int(protocol), http_headers_arr, len(b_http_headers),
                    model_name, verbose)))

    def __del__(self):
        # when module is unloading may get called after
        # _crequest_status_ctx_del has been released
        if _crequest_status_ctx_del is not None:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        """Close the context. Any future calls to get_server_status() will
        result in an Error.

        """
        _crequest_status_ctx_del(self._ctx)
        self._ctx = None

    def get_server_status(self):
        """Contact the inference server and get status.

        Returns
        -------
        ServerStatus
            The ServerStatus protobuf containing the status.

        Raises
        ------
        InferenceServerException
            If unable to get status.

        """
        self._last_request_id = None
        if self._ctx is None:
            _raise_error("ServerStatusContext is closed")

        cstatus = c_char_p()
        cstatus_len = c_uint32()
        self._last_request_id = _raise_if_error(
            c_void_p(_crequest_status_ctx_get(
                self._ctx, byref(cstatus), byref(cstatus_len))))

        status = text_format.Parse(cstatus.value.decode(), ServerStatus())
        return status

    def get_last_request_id(self):
        """Get the request ID of the most recent get_server_status() request.

        Returns
        -------
        int
            The request ID, or None if a request has not yet been made
            or if the last request was not successful.

        """
        return self._last_request_id


class ModelRepositoryContext:
    """Performs a model repository request to an inference server.

    A request can be made to get model repository information of the server

    Parameters
    ----------
    url : str
        The inference server URL, e.g. localhost:8000.

    protocol : ProtocolType
        The protocol used to communicate with the server.

    verbose : bool
        If True generate verbose output.

    http_headers : list of strings
        HTTP headers to send with request. Ignored for GRPC
        protocol. Each header must be specified as "Header:Value".

    """
    def __init__(self, url, protocol, verbose=False, http_headers=[]):
        self._last_request_id = 0
        self._ctx = c_void_p()

        b_http_headers = list()
        if http_headers is not None:
            for hh in http_headers:
                b_http_headers.append(hh.encode('utf-8'))

        http_headers_arr = (c_char_p * len(b_http_headers))()
        http_headers_arr[:] = b_http_headers

        _raise_if_error(
            c_void_p(
                _crequest_repository_ctx_new(
                    byref(self._ctx), url, int(protocol), http_headers_arr, len(b_http_headers), verbose)))

    def __del__(self):
        # when module is unloading may get called after
        # _crequest_status_ctx_del has been released
        if _crequest_repository_ctx_del is not None:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        """Close the context. Any future calls to get_model_repository_index()
        will result in an Error.

        """
        _crequest_status_ctx_del(self._ctx)
        self._ctx = None

    def get_model_repository_index(self):
        """Contact the inference server and get the index of the model repository.

        Returns
        -------
        ModelRepositoryIndex
            The ModelRepositoryIndex protobuf containing the index.

        Raises
        ------
        InferenceServerException
            If unable to get index.

        """
        self._last_request_id = None
        if self._ctx is None:
            _raise_error("ModelRepositoryContext is closed")

        cindex = c_char_p()
        cindex_len = c_uint32()
        self._last_request_id = _raise_if_error(
            c_void_p(_crequest_repository_ctx_get(
                self._ctx, byref(cindex), byref(cindex_len))))
        index_buf = cast(cindex, POINTER(c_byte * cindex_len.value))[0]

        index = ModelRepositoryIndex()
        index.ParseFromString(index_buf)
        return index

    def get_last_request_id(self):
        """Get the request ID of the most recent get_model_repository_index() request.

        Returns
        -------
        int
            The request ID, or None if a request has not yet been made
            or if the last request was not successful.

        """
        return self._last_request_id


class ModelControlContext:
    """Performs a model control request to an inference server.

    Parameters
    ----------
    url : str
        The inference server URL, e.g. localhost:8000.

    protocol : ProtocolType
        The protocol used to communicate with the server.

    verbose : bool
        If True generate verbose output.

    http_headers : list of strings
        HTTP headers to send with request. Ignored for GRPC
        protocol. Each header must be specified as "Header:Value".

    """
    def __init__(self, url, protocol, verbose=False, http_headers=[]):
        self._last_request_id = 0
        self._ctx = c_void_p()

        b_http_headers = list()
        if http_headers is not None:
            for hh in http_headers:
                b_http_headers.append(hh.encode('utf-8'))

        http_headers_arr = (c_char_p * len(b_http_headers))()
        http_headers_arr[:] = b_http_headers

        _raise_if_error(
            c_void_p(
                _crequest_model_control_ctx_new(
                    byref(self._ctx), url, int(protocol),
                    http_headers_arr, len(b_http_headers), verbose)))

    def __del__(self):
        # when module is unloading may get called after
        # _crequest_model_control_ctx_del has been released
        if _crequest_model_control_ctx_del is not None:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        """Close the context. Any future calls to load() or unload() will
        result in an Error.

        """
        _crequest_model_control_ctx_del(self._ctx)
        self._ctx = None

    def load(self, model_name):
        """Request the inference server to load specified model.

        Parameters
        ----------
        model_name : str
            The name of the model to be loaded.

        Raises
        ------
        InferenceServerException
            If unable to load the model.

        """
        self._last_request_id = None
        if self._ctx is None:
            _raise_error("ModelControlContext is closed")

        self._last_request_id = _raise_if_error(
            c_void_p(_crequest_model_control_ctx_load(self._ctx, model_name)))
        return

    def unload(self, model_name):
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
        self._last_request_id = None
        if self._ctx is None:
            _raise_error("ModelControlContext is closed")

        self._last_request_id = _raise_if_error(
            c_void_p(_crequest_model_control_ctx_unload(self._ctx, model_name)))
        return

    def get_last_request_id(self):
        """Get the request ID of the most recent load() or unload()
        request.

        Returns
        -------
        int
            The request ID, or None if a request has not yet been made
            or if the last request was not successful.

        """
        return self._last_request_id


if os.name != 'nt':
    class SharedMemoryControlContext:
        """Performs a shared memory control request to an inference server.

        Parameters
        ----------
        url : str
            The inference server URL, e.g. localhost:8000.

        protocol : ProtocolType
            The protocol used to communicate with the server.

        verbose : bool
            If True generate verbose output.

        http_headers : list of strings
            HTTP headers to send with request. Ignored for GRPC
            protocol. Each header must be specified as "Header:Value".

        """
        def __init__(self, url, protocol, verbose=False, http_headers=[]):
            self._last_request_id = 0
            self._ctx = c_void_p()

            b_http_headers = list()
            if http_headers is not None:
                for hh in http_headers:
                    b_http_headers.append(hh.encode('utf-8'))

            http_headers_arr = (c_char_p * len(b_http_headers))()
            http_headers_arr[:] = b_http_headers

            _raise_if_error(
                c_void_p(
                    _crequest_shm_control_ctx_new(
                        byref(self._ctx), url, int(protocol),
                        http_headers_arr, len(b_http_headers), verbose)))

        def __del__(self):
            if _crequest_shm_control_ctx_del is not None:
                self.close()

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            self.close()

        def close(self):
            """Close the context. Any future calls to register() or unregister()
            will result in an Error.

            """
            _crequest_shm_control_ctx_del(self._ctx)
            self._ctx = None

        def register(self, shm_handle):
            """Request the inference server to register specified shared memory region.

            Parameters
            ----------
            shm_handle : c_void_p
                The handle for the shared memory region.

            Raises
            ------
            InferenceServerException
                If unable to register the shared memory region.

            """
            self._last_request_id = None
            if self._ctx is None:
                _raise_error("SharedMemoryControlContext is closed")

            self._last_request_id = _raise_if_error(
                c_void_p(_crequest_shm_control_ctx_register(self._ctx, shm_handle)))
            return

        def cuda_register(self, cuda_shm_handle):
            """Request the inference server to register specified shared memory region.

            Parameters
            ----------
            cuda_shm_handle : c_void_p
                The handle for the CUDA shared memory region.

            Raises
            ------
            InferenceServerException
                If unable to register the shared memory region.

            """
            self._last_request_id = None
            if self._ctx is None:
                _raise_error("SharedMemoryControlContext is closed")

            self._last_request_id = _raise_if_error(
                c_void_p(_crequest_shm_control_ctx_cuda_register(self._ctx, cuda_shm_handle)))
            return

        def unregister(self, shm_handle):
            """Request the inference server to unregister specified shared memory region.

            Parameters
            ----------
            shm_handle : c_void_p
                The handle for the shared memory region.

            Raises
            ------
            InferenceServerException
                If unable to unregister the shared memory region.

            """
            self._last_request_id = None
            if self._ctx is None:
                _raise_error("SharedMemoryControlContext is closed")

            self._last_request_id = _raise_if_error(
                c_void_p(_crequest_shm_control_ctx_unregister(self._ctx, shm_handle)))
            return

        def unregister_all(self):
            """Request the inference server to unregister all shared memory regions.

            Raises
            ------
            InferenceServerException
                If unable to unregister any shared memory regions.

            """
            self._last_request_id = None
            if self._ctx is None:
                _raise_error("SharedMemoryControlContext is closed")

            self._last_request_id = _raise_if_error(
                c_void_p(_crequest_shm_control_ctx_unregister_all(self._ctx)))
            return

        def get_shared_memory_status(self):
            """Contact the inference server and get status.

            Returns
            -------
            SharedMemoryStatus
                The SharedMemoryStatus protobuf containing the status.

            Raises
            ------
            InferenceServerException
                If unable to get status.

            """
            self._last_request_id = None
            if self._ctx is None:
                _raise_error("SharedMemoryControlContext is closed")

            cstatus = c_char_p()
            cstatus_len = c_uint32()
            self._last_request_id = _raise_if_error(
                c_void_p(_crequest_shm_control_ctx_get_status(
                    self._ctx, byref(cstatus), byref(cstatus_len))))

            status = text_format.Parse(cstatus.value.decode(), SharedMemoryStatus())
            return status

        def get_last_request_id(self):
            """Get the request ID of the most recent register() or unregister()
            request.

            Returns
            -------
            int
                The request ID, or None if a request has not yet been made
                or if the last request was not successful.

            """
            return self._last_request_id


class InferContext:
    """An InferContext object is used to run inference on an inference
    server for a specific model.

    Once created an InferContext object can be used repeatedly to
    perform inference using the model.

    Parameters
    ----------
    url : str
        The inference server URL, e.g. localhost:8000.

    protocol : ProtocolType
        The protocol used to communicate with the server.

    model_name : str
        The name of the model to use for inference.

    model_version : int
        The version of the model to use for inference,
        or None to indicate that the latest (i.e. highest version number)
        version should be used.

    verbose : bool
        If True generate verbose output.

    correlation_id : int
        The correlation ID for the inference. If not specified (or if
        specified as 0), the inference will have no correlation ID.

    streaming : bool
        If True create streaming context. Streaming is only allowed with
        gRPC protocol.

    http_headers : list of strings
        HTTP headers to send with request. Ignored for GRPC
        protocol. Each header must be specified as "Header:Value".

    """
    class ResultFormat:
        """Formats for output tensor results.

        RAW
            All values of the output are returned as an numpy array
            of the appropriate type.

        CLASS
            Specified as tuple (CLASS, k). Top 'k' results
            are returned as an array of (index, value, label) tuples.

        """
        RAW = 1,
        CLASS = 2

    def __init__(self, url, protocol, model_name, model_version=None,
                 verbose=False, correlation_id=0, streaming=False, http_headers=[]):
        self._correlation_id = correlation_id
        self._last_request_id = None
        self._last_request_model_name = None
        self._last_request_model_version = None
        self._requested_outputs_dict = dict()
        # Similar to _requested_outputs_dict, but contains other resources
        # that has to be kept for callback
        self._callback_resources_dict = dict()
        self._callback_resources_dict_id = 0
        self._ctx = c_void_p()
        # Lock for the thread-safety across asynchronous requests
        self._lock = threading.Lock()

        b_http_headers = list()
        if http_headers is not None:
            for hh in http_headers:
                b_http_headers.append(hh.encode('utf-8'))

        http_headers_arr = (c_char_p * len(b_http_headers))()
        http_headers_arr[:] = b_http_headers

        imodel_version = -1 if model_version is None else model_version
        _raise_if_error(
            c_void_p(
                _crequest_infer_ctx_new(
                    byref(self._ctx), url, int(protocol),
                    http_headers_arr, len(b_http_headers),
                    model_name, imodel_version, correlation_id,
                    streaming, verbose)))

    def __del__(self):
        # when module is unloading may get called after
        # _crequest_infer_ctx_del has been released
        if _crequest_infer_ctx_del is not None:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def _async_callback_wrapper(self, dict_id, callback, cb_ctx, cb_request_id):
        # By this point, request id is known and '_requested_outputs_dict'
        # can be set to retrieve results properly
        self._requested_outputs_dict[cb_request_id] = dict_id
        # 'ctx_obj' is captured by partial and used for Python API,
        # 'cb_ctx' is placeholder as crequest API provides pointer to the struct
        # directly to be consistent with the C++ API
        callback(self, cb_request_id)
        return None

    def _get_result_numpy_dtype(self, result):
        ctype = c_uint32()
        _raise_if_error(c_void_p(_crequest_infer_ctx_result_dtype(result, byref(ctype))))
        if ctype.value == model_config_pb2.TYPE_BOOL:
            return np.bool_
        elif ctype.value == model_config_pb2.TYPE_UINT8:
            return np.uint8
        elif ctype.value == model_config_pb2.TYPE_UINT16:
            return np.uint16
        elif ctype.value == model_config_pb2.TYPE_UINT32:
            return np.uint32
        elif ctype.value == model_config_pb2.TYPE_UINT64:
            return np.uint64
        elif ctype.value == model_config_pb2.TYPE_INT8:
            return np.int8
        elif ctype.value == model_config_pb2.TYPE_INT16:
            return np.int16
        elif ctype.value == model_config_pb2.TYPE_INT32:
            return np.int32
        elif ctype.value == model_config_pb2.TYPE_INT64:
            return np.int64
        elif ctype.value == model_config_pb2.TYPE_FP16:
            return np.float16
        elif ctype.value == model_config_pb2.TYPE_FP32:
            return np.float32
        elif ctype.value == model_config_pb2.TYPE_FP64:
            return np.float64
        elif ctype.value == model_config_pb2.TYPE_STRING:
            return np.dtype(object)
        _raise_error("unknown result datatype " + ctype.value)

    def _prepare_request(self, inputs, outputs,
                         flags, batch_size, corr_id, priority, timeout_us,
                         contiguous_input_values):
        # Make sure each input is given as a list (one entry per
        # batch). It is a common error when using batch-size 1 to
        # specify an input directly as an array instead of as a list
        # containing one array.
        # An input's data may be specified as a list of numpy arrays,
        # or as a shared memory handle or as a tuple of a shared memory
        # handle and the shape of the input tensor.
        for inp_name, inp in inputs.items():
            if (not isinstance(inp, (list, tuple))) and (type(inp) != c_void_p):
                _raise_error("input '" + inp_name +
                             "' values must be specified as a list of numpy arrays" \
                             " or as a single c_void_p (representing the shared memory handle)" \
                             " or as a tuple of c_void_p (representing the shared memory handle)" \
                             " and list (representing the shape of the input tensor)")
            if type(inp) != c_void_p:
                # Skip further checks for this input if it is a tuple of shared memory and shape
                # of the form (c_void_p, list)
                if (len(inp) == 2) and (type(inp[0]) == c_void_p) and (isinstance(inp[1], (list, tuple))):
                    continue
                for ip in inp:
                    if not isinstance(ip, (np.ndarray, tuple)):
                        _raise_error("input '" + inp_name +
                                     "' values must be specified as a list of numpy arrays" \
                             " or as a single c_void_p (representing the shared memory handle)" \
                             " or as a tuple of c_void_p (representing the shared memory handle)" \
                             " and list (representing the shape of the input tensor)")
        # Set run options using formats specified in 'outputs'
        # An output format may be may be specified as a RAW or (CLASS, cnt)
        # or as a (RAW, shared_memory_handle).
        options = c_void_p()
        try:
            _raise_if_error(c_void_p(
                _crequest_infer_ctx_options_new(
                    byref(options), flags, batch_size, corr_id, priority, timeout_us)))

            for (output_name, output_format) in iteritems(outputs):
                if len(output_format) == 2 and isinstance(output_format, (list, tuple)) \
                    and output_format[0] == InferContext.ResultFormat.RAW:
                    if type(output_format[1]) != c_void_p:
                        _raise_error("shared memory requires tuple of size 2" \
                                    " - output_format(RAW), shared_memory_handle(c_void_p)")
                    _raise_if_error(
                        c_void_p(
                            _crequest_infer_ctx_options_add_shared_memory(
                                self._ctx, options, output_name, output_format[1])))
                elif output_format == InferContext.ResultFormat.RAW:
                    _raise_if_error(
                        c_void_p(
                            _crequest_infer_ctx_options_add_raw(self._ctx, options, output_name)))
                elif (isinstance(output_format, (list, tuple)) and
                      (output_format[0] == InferContext.ResultFormat.CLASS)):
                    _raise_if_error(
                        c_void_p(
                            _crequest_infer_ctx_options_add_class(
                                self._ctx, options, output_name, c_uint64(output_format[1]))))
                else:
                    _raise_error("unrecognized output format")

            _raise_if_error(c_void_p(_crequest_infer_ctx_set_options(self._ctx, options)))

        finally:
            _crequest_infer_ctx_options_del(options)

        # Set the input tensors
        for (input_name, input_values) in iteritems(inputs):
            input = c_void_p()
            try:
                _raise_if_error(
                    c_void_p(_crequest_infer_ctx_input_new(byref(input), self._ctx, input_name)))

                # Set the input shape
                if isinstance(input_values, (list, tuple)):
                    if len(input_values) > 0:
                        if isinstance(input_values[0], (np.ndarray,)):
                            shape_value = np.asarray(input_values[0].shape, dtype=np.int64)
                            _raise_if_error(
                                c_void_p(
                                    _crequest_infer_ctx_input_set_shape(
                                           input, shape_value, c_uint64(shape_value.size))))

                    # use values if numpy array, reference if shared memory
                    if isinstance(input_values[0], (np.ndarray,)):
                        for input_value in input_values:
                            # If the input tensor is empty then avoid going
                            # through the more complicated logic since
                            # creating the buffer for string objects results
                            # is a size-1 array instead of 0.
                            if input_value.size == 0:
                                _raise_if_error(
                                    c_void_p(
                                        _crequest_infer_ctx_input_set_raw(input, 0, 0)))
                            else:
                                # If the input is a tensor of string objects,
                                # then must flatten those into a 1-dimensional
                                # array containing the 4-byte string length
                                # followed by the actual string characters.
                                # All strings are concatenated together in "C"
                                # order.
                                if (input_value.dtype == np.object) or (input_value.dtype.type == np.bytes_):
                                    input_value = serialize_string_tensor(input_value)

                                if not input_value.flags['C_CONTIGUOUS']:
                                    input_value = np.ascontiguousarray(input_value)
                                contiguous_input_values.append(input_value)
                                _raise_if_error(
                                    c_void_p(
                                        _crequest_infer_ctx_input_set_raw(
                                            input, input_value.ctypes.data_as(c_void_p),
                                            c_uint64(input_value.size * input_value.itemsize))))
                    # For variable size tensors, need the shape as well as the
                    # shared memory handle
                    elif isinstance(input_values[1], (list, tuple)) and (type(input_values[0]) == c_void_p):
                        shape_value = np.asarray(input_values[1], dtype=np.int64)
                        _raise_if_error(
                            c_void_p(
                                _crequest_infer_ctx_input_set_shape(
                                       input, shape_value, c_uint64(shape_value.size))))
                        _raise_if_error(
                            c_void_p(
                                _crequest_infer_ctx_input_set_shared_memory(
                                    input, input_values[0])))
                else:
                    _raise_if_error(
                        c_void_p(
                            _crequest_infer_ctx_input_set_shared_memory(
                                input, input_values)))

            finally:
                _crequest_infer_ctx_input_del(input)

    def _get_results(self, outputs, batch_size, request_id=None):
        # Create the result map.
        results = dict()
        for (output_name, output_format) in iteritems(outputs):
            result = c_void_p()
            try:
                if request_id is None:
                    _raise_if_error(
                        c_void_p(_crequest_infer_ctx_result_new(byref(result), self._ctx, output_name)))
                else:
                    _raise_if_error(
                        c_void_p(_crequest_infer_ctx_async_result_new(byref(result), self._ctx, request_id, output_name)))

                # The model name and version are the same for every
                # result so only set once
                if self._last_request_model_name is None:
                    cmodelname = c_char_p()
                    _raise_if_error(
                        c_void_p(
                            _crequest_infer_ctx_result_modelname(result, byref(cmodelname))))
                    if cmodelname.value is not None:
                        self._last_request_model_name = cmodelname.value.decode('utf-8')
                if self._last_request_model_version is None:
                    cmodelver = c_int64()
                    _raise_if_error(
                        c_void_p(
                            _crequest_infer_ctx_result_modelver(result, byref(cmodelver))))
                    self._last_request_model_version = cmodelver.value

                result_dtype = self._get_result_numpy_dtype(result)
                results[output_name] = list()
                if output_format == InferContext.ResultFormat.RAW:
                    # Get the shape of each result tensor
                    max_shape_dims = 16
                    shape_array = np.zeros(max_shape_dims, dtype=np.int64)
                    shape_len = c_uint64()
                    _raise_if_error(
                        c_void_p(
                            _crequest_infer_ctx_result_shape(
                                result, c_uint64(max_shape_dims),
                                shape_array, byref(shape_len))))
                    shape = np.resize(shape_array, shape_len.value).tolist()

                    for b in range(batch_size):
                        # Get the result value into a 1-dim np array
                        # of the appropriate type
                        cval = c_char_p()
                        cval_len = c_uint64()
                        _raise_if_error(
                            c_void_p(
                                _crequest_infer_ctx_result_next_raw(
                                    result, b, byref(cval), byref(cval_len))))
                        if cval_len.value == 0:
                            val = np.empty(shape, dtype=result_dtype)
                            results[output_name].append(val)
                        else:
                            val_buf = cast(cval, POINTER(c_byte * cval_len.value))[0]

                            # If the result is not a string datatype
                            # then convert directly. Otherwise parse
                            # 'val_buf' into an array of strings and
                            # from that into a numpy array of string
                            # objects.
                            if result_dtype != np.object:
                                val = np.frombuffer(val_buf, dtype=result_dtype)
                            else:
                                # String results contain a 4-byte
                                # string length followed by the actual
                                # string characters.
                                strs = list()
                                offset = 0
                                while offset < len(val_buf):
                                    l = struct.unpack_from("<I", val_buf, offset)[0]
                                    offset += 4
                                    sb = struct.unpack_from("<{}s".format(l), val_buf, offset)[0]
                                    offset += l
                                    strs.append(sb)
                                val = np.array(strs, dtype=object)

                            # Reshape the result to the appropriate shape
                            shaped = np.reshape(np.copy(val), shape)
                            results[output_name].append(shaped)

                elif (isinstance(output_format, (list, tuple)) and
                      (output_format[0] == InferContext.ResultFormat.CLASS)):
                    for b in range(batch_size):
                        classes = list()
                        ccnt = c_uint64()
                        _raise_if_error(
                           c_void_p(_crequest_infer_ctx_result_class_cnt(result, b, byref(ccnt))))
                        for cc in range(ccnt.value):
                            cidx = c_uint64()
                            cprob = c_float()
                            clabel = c_char_p()
                            _raise_if_error(
                                c_void_p(
                                    _crequest_infer_ctx_result_next_class(
                                        result, b, byref(cidx), byref(cprob), byref(clabel))))
                            label = None if clabel.value is None else clabel.value.decode('utf-8')
                            classes.append((cidx.value, cprob.value, label))
                        results[output_name].append(classes)
                elif (isinstance(output_format, (list, tuple)) and
                    (output_format[0] == InferContext.ResultFormat.RAW) and (len(output_format) == 2)):
                    if os.name == 'nt':
                        _raise_error("Shared memory not supported on Windows")
                    else:
                        # Get the shape of each result tensor
                        max_shape_dims = 16
                        shape_array = np.zeros(max_shape_dims, dtype=np.int64)
                        shape_len = c_uint64()
                        _raise_if_error(
                            c_void_p(
                                _crequest_infer_ctx_result_shape(
                                    result, c_uint64(max_shape_dims),
                                    shape_array, byref(shape_len))))
                        shape = np.resize(shape_array, shape_len.value).tolist()

                        # get info for shared memory regions and read results
                        shm_fd = c_int()
                        offset = c_uint64()
                        byte_size = c_uint64()
                        shm_addr = c_char_p()
                        shm_key = c_char_p()
                        try:
                            _raise_if_error(
                                c_void_p(_crequest_get_shared_memory_handle_info(output_format[1], \
                                        byref(shm_addr), byref(shm_key), byref(shm_fd), \
                                        byref(offset), byref(byte_size))))
                            if (np.prod(shape) * np.dtype(result_dtype).itemsize) < int(byte_size.value/batch_size):
                                element_byte_size = np.prod(shape) * np.dtype(result_dtype).itemsize
                            else:
                                element_byte_size = int(byte_size.value/batch_size)
                            start_pos = offset.value
                            if result_dtype != np.object:
                                cval = shm_addr
                                for b in range(batch_size):
                                    cval_len = start_pos + element_byte_size
                                    if cval_len == 0:
                                        val = np.empty(shape, dtype=result_dtype)
                                        results[output_name].append(val)
                                    else:
                                        val_buf = cast(cval, POINTER(c_byte * cval_len))[0]
                                        val = np.frombuffer(val_buf, dtype=result_dtype, offset=start_pos)
                                    start_pos += element_byte_size

                                    # Reshape the result to the appropriate shape.
                                    # This copy is only needed for CUDA shared memory
                                    # since the temporary CPU buffer is cleared later
                                    # by _crequest_shared_memory_handle_release_buffer
                                    shaped = np.reshape(np.copy(val), shape)

                                    results[output_name].append(shaped)
                            else:
                                cval = shm_addr
                                str_offset = start_pos
                                val_buf = cast(cval, POINTER(c_byte * byte_size.value))[0]
                                b = 0
                                while b < batch_size:
                                    ii = 0
                                    strs = list()
                                    while (ii % np.prod(shape) != 0) or (ii == 0):
                                        l = struct.unpack_from("<I", val_buf, str_offset)[0]
                                        str_offset += 4
                                        sb = struct.unpack_from("<{}s".format(l), val_buf, str_offset)[0]
                                        str_offset += l
                                        strs.append(sb)
                                        ii+=1
                                    b+=1
                                    val = np.array(strs, dtype=object)

                                    # Reshape the result to the appropriate shape.
                                    shaped = np.reshape(val, shape)

                                    results[output_name].append(shaped)
                        finally:
                            _raise_if_error(
                                c_void_p(_crequest_shared_memory_handle_release_buffer(output_format[1], shm_addr)))
                else:
                    _raise_error("unrecognized output format")
            finally:
                _crequest_infer_ctx_result_del(result)

        return results

    def close(self):
        """Close the context. Any future calls to object will result in an
        Error.

        """
        _crequest_infer_ctx_del(self._ctx)
        self._ctx = None

    def correlation_id(self):
        """Get the correlation ID associated with the context.

        Returns
        -------
        int
            The correlation ID.

        """
        return _crequest_correlation_id(self._ctx)

    def run(self, inputs, outputs, batch_size=1, flags=0, corr_id=0,
            priority=0, timeout_us=0):
        """Run inference using the supplied 'inputs' to calculate the outputs
        specified by 'outputs'.

        Parameters
        ----------
        inputs : dict
            Dictionary from input name to the value(s) for that
            input. An input value is specified as a numpy array. Each
            input in the dictionary maps to a list of values (i.e. a
            list of numpy array objects), where the length of the list
            must equal the 'batch_size'.
            However, for shape tensor input the list should contain
            just a single tensor.

        outputs : dict
            Dictionary from output name to a value indicating the
            ResultFormat that should be used for that output. For RAW
            the value should be ResultFormat.RAW. For CLASS the value
            should be a tuple (ResultFormat.CLASS, k), where 'k'
            indicates how many classification results should be
            returned for the output.

        batch_size : int
            The batch size of the inference. Each input must provide
            an appropriately sized batch of inputs.

        flags : int
            The flags to use for the inference. The bitwise-or of
            InferRequestHeader.Flag values.

        corr_id : int
            The correlation id of the inference. Used to differentiate
            sequences.

        priority : int
            The priority of the inference.

        timeout_us : int
            The timeout of the inference, in microseconds.

        Returns
        -------
        dict
            A dictionary from output name to the list of values for
            that output (one list element for each entry of the
            batch). The format of a value returned for an output
            depends on the output format specified in 'outputs'. For
            format RAW a value is a numpy array of the appropriate
            type and shape for the output. For format CLASS a value is
            the top 'k' output values returned as an array of (class
            index, class value, class label) tuples.

        Raises
        ------
        InferenceServerException
            If all inputs are not specified, if the size of input data
            does not match expectations, if unknown output names are
            specified or if server fails to perform inference.

        """
        self._last_request_id = None
        self._last_request_model_name = None
        self._last_request_model_version = None

        # The input values must be contiguous and the lifetime of those
        # contiguous copies must span until the inference completes
        # so grab a reference to them at this scope.
        contiguous_input = list()

        # Set run option and input values
        self._prepare_request(
            inputs, outputs, flags, batch_size, corr_id, priority, timeout_us, contiguous_input)

        # Run inference...
        self._last_request_id = _raise_if_error(c_void_p(_crequest_infer_ctx_run(self._ctx)))

        return self._get_results(outputs, batch_size)

    def async_run(self, callback, inputs, outputs, batch_size=1, flags=0, corr_id=0,
                  priority=0, timeout_us=0):
        """Run inference using the supplied 'inputs' to calculate the outputs
        specified by 'outputs'.

        Once the request is completed, the InferContext object and the integer
        identifier will be passed to the provided 'callback' function. It is the
        function caller's choice on either retrieving the results inside the
        callback function or deferring it to a different thread so that the
        InferContext is unblocked.

        Parameters
        ----------
        callback : function
            Python function that accepts an InferContext object that sends the
            request and an integer identifier as arguments. This function will
            be invoked once the request is completed.

        inputs : dict
            Dictionary from input name to the value(s) for that
            input. An input value is specified as a numpy array. Each
            input in the dictionary maps to a list of values (i.e. a
            list of numpy array objects), where the length of the list
            must equal the 'batch_size'.
            However, for shape tensor input the list should contain
            just a single tensor.

        outputs : dict
            Dictionary from output name to a value indicating the
            ResultFormat that should be used for that output. For RAW
            the value should be ResultFormat.RAW. For CLASS the value
            should be a tuple (ResultFormat.CLASS, k), where 'k'
            indicates how many classification results should be
            returned for the output.

        batch_size : int
            The batch size of the inference. Each input must provide
            an appropriately sized batch of inputs.

        corr_id : int
            The correlation id of the inference. If non-zero this
            correlation ID overrides the context's correlation ID for
            all subsequent inference requests, else the inference
            request uses the context's correlation ID.

        flags : int
            The flags to use for the inference. The bitwise-or of
            InferRequestHeader.Flag values.

        priority : int
            The priority of the inference.

        timeout_us : int
            The timeout of the inference, in microseconds.

        Raises
        ------
        InferenceServerException
            If all inputs are not specified, if the size of input data
            does not match expectations, if unknown output names are
            specified or if server fails to perform inference.

        """
        # Same situation as in run(), but the list will be kept inside
        # the object given that the request is asynchronous
        contiguous_input = list()

        # Set run option and input values
        self._prepare_request(
            inputs, outputs, flags, batch_size, corr_id, priority, timeout_us, contiguous_input)

        # Wrap over the provided callback
        wrapped_cb = partial(self._async_callback_wrapper, self._callback_resources_dict_id, callback)
        c_cb = _async_run_callback_prototype(wrapped_cb)

        with self._lock:
            # Run asynchronous inference...
            _raise_if_error(
                c_void_p(
                    _crequest_infer_ctx_async_run(self._ctx, c_cb)))

            self._callback_resources_dict[self._callback_resources_dict_id] = \
                (outputs, batch_size, contiguous_input, c_cb, wrapped_cb)
            self._callback_resources_dict_id += 1

    def get_async_run_results(self, request_id):
        """Retrieve the results of a previous async_run() using the supplied
        'request_id'

        Parameters
        ----------
        request_id : int
            The integer ID of the asynchronous request exposed in the
            callback function of the async_run.

        Returns
        -------
        dict
            A dictionary from output name to the list of values for that
            output (one list element for each entry of the batch). The
            format of a value returned for an output depends on the
            output format specified in 'outputs'. For format RAW a
            value is a numpy array of the appropriate type and shape
            for the output. For format CLASS a value is the top 'k'
            output values returned as an array of (class index, class
            value, class label) tuples.

        Raises
        ------
        InferenceServerException
            If the request ID supplied is not valid, or if the server
            fails to perform inference.

        """
        # Get async run results
        err = c_void_p(_crequest_infer_ctx_get_async_run_results(
            self._ctx, request_id))

        with self._lock:
            requested_outputs = self._requested_outputs_dict[request_id]
            if isinstance(requested_outputs, int):
                idx = requested_outputs
                requested_outputs = self._callback_resources_dict[idx]
                del self._callback_resources_dict[idx]
            del self._requested_outputs_dict[request_id]

        self._last_request_id = _raise_if_error(err)

        return self._get_results(requested_outputs[0], requested_outputs[1], request_id)

    def get_last_request_id(self):
        """Get the request ID of the most recent run() request.

        Returns
        -------
        int
            The request ID, or None if a request has not yet been made
            or if the last request was not successful.

        """
        return self._last_request_id

    def get_last_request_model_name(self):
        """Get the model name used in the most recent run() request.

        Returns
        -------
        str
            The model name, or None if a request has not yet been made
            or if the last request was not successful.

        """
        return self._last_request_model_name

    def get_last_request_model_version(self):
        """Get the model version used in the most recent run() request.

        Returns
        -------
        int
            The model version, or None if a request has not yet been made
            or if the last request was not successful.

        """
        return self._last_request_model_version

    def get_stat(self):
        """Get the current statistics of the InferContext.

        Returns
        -------
        dict
            Containing the completed_request_count,
            cumulative_total_request_time_ns, cumulative_send_time_ns
            and cumulative_receive_time_ns with their respective keys.

        Raises
        ------
        InferenceServerException
            If fails to retrieve the statistics.

        """
        stat = dict()
        completed_request_count = c_uint64()
        cumulative_total_request_time_ns = c_uint64()
        cumulative_send_time_ns = c_uint64()
        cumulative_receive_time_ns = c_uint64()
        _raise_if_error(c_void_p(_crequest_infer_ctx_get_stat(
                self._ctx, byref(completed_request_count),
                byref(cumulative_total_request_time_ns),
                byref(cumulative_send_time_ns),
                byref(cumulative_receive_time_ns))))
        # Populate the dictionary with the values
        stat["completed_request_count"] = completed_request_count.value
        stat["cumulative_total_request_time_ns"] = cumulative_total_request_time_ns.value
        stat["cumulative_send_time_ns"] = cumulative_send_time_ns.value
        stat["cumulative_receive_time_ns"] = cumulative_receive_time_ns.value

        return stat
