# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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
import tensorrtserver.api.model_config_pb2
from tensorrtserver.api.server_status_pb2 import ServerStatus
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
_cshmwrap_lib = "shmwrap" if os.name == 'nt' else 'libcshmwrap.so'
_request_path = pkg_resources.resource_filename('tensorrtserver.api', _request_lib)
_request = cdll.LoadLibrary(_request_path)
_crequest_path = pkg_resources.resource_filename('tensorrtserver.api', _crequest_lib)
_crequest = cdll.LoadLibrary(_crequest_path)
_cshmwrap_path = pkg_resources.resource_filename('tensorrtserver.api', _cshmwrap_lib)
# _cshmwrap = cdll.LoadLibrary("/workspace/build/trtis-clients/src/clients/python/libcshmwrap.so")
_cshmwrap = cdll.LoadLibrary(_cshmwrap_path)

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

_crequest_shm_control_ctx_new = _crequest.SharedMemoryControlContextNew
_crequest_shm_control_ctx_new.restype = c_void_p
_crequest_shm_control_ctx_new.argtypes = [POINTER(c_void_p), _utf8, c_int, c_bool]
_crequest_shm_control_ctx_del = _crequest.SharedMemoryControlContextDelete
_crequest_shm_control_ctx_del.argtypes = [c_void_p]
_crequest_shm_control_ctx_register = _crequest.SharedMemoryControlContextRegister
_crequest_shm_control_ctx_register.restype = c_void_p
_crequest_shm_control_ctx_register.argtypes = [c_void_p, _utf8]
_crequest_shm_control_ctx_unregister = _crequest.SharedMemoryControlContextUnregister
_crequest_shm_control_ctx_unregister.restype = c_void_p
_crequest_shm_control_ctx_unregister.argtypes = [c_void_p, _utf8]

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
_crequest_infer_ctx_async_run = _crequest.InferContextAsyncRun
_crequest_infer_ctx_async_run.restype = c_void_p
_crequest_infer_ctx_async_run.argtypes = [c_void_p, POINTER(c_uint64)]
_async_run_callback_prototype = CFUNCTYPE(None, c_void_p, c_uint64)
_crequest_infer_ctx_async_run_with_cb = _crequest.InferContextAsyncRunWithCallback
_crequest_infer_ctx_async_run_with_cb.restype = c_void_p
_crequest_infer_ctx_async_run_with_cb.argtypes = [c_void_p, _async_run_callback_prototype]
_crequest_infer_ctx_get_async_run_results = _crequest.InferContextGetAsyncRunResults
_crequest_infer_ctx_get_async_run_results.restype = c_void_p
_crequest_infer_ctx_get_async_run_results.argtypes = [c_void_p, POINTER(c_bool), c_uint64, c_bool]
_crequest_infer_ctx_get_ready_async_request = _crequest.InferContextGetReadyAsyncRequest
_crequest_infer_ctx_get_ready_async_request.restype = c_void_p
_crequest_infer_ctx_get_ready_async_request.argtypes = [c_void_p, POINTER(c_bool), POINTER(c_uint64), c_bool]

_crequest_infer_ctx_options_new = _crequest.InferContextOptionsNew
_crequest_infer_ctx_options_new.restype = c_void_p
_crequest_infer_ctx_options_new.argtypes = [POINTER(c_void_p), c_uint32, c_uint64]
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
_crequest_infer_ctx_options_add_shared_memory.argtypes = [c_void_p, c_void_p, _utf8, _utf8, c_uint64, c_uint64]

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
_crequest_infer_ctx_input_set_shared_memory.argtypes = [c_void_p, _utf8, c_uint64, c_uint64]

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

_cshmwrap_create_shared_memory_region = _cshmwrap.CreateSharedMemoryRegion
_cshmwrap_create_shared_memory_region.restype = c_void_p
_cshmwrap_create_shared_memory_region.argtypes = [_utf8, c_uint64, POINTER(c_int)]
_cshmwrap_open_shared_memory_region = _cshmwrap.OpenSharedMemoryRegion
_cshmwrap_open_shared_memory_region.restype = c_void_p
_cshmwrap_open_shared_memory_region.argtypes = [POINTER(c_char_p), POINTER(c_int)]
_cshmwrap_close_shared_memory_region = _cshmwrap.CloseSharedMemoryRegion
_cshmwrap_close_shared_memory_region.restype = c_void_p
_cshmwrap_close_shared_memory_region.argtypes = [c_uint64]
_cshmwrap_set_shared_memory_region_data = _cshmwrap.SetSharedMemoryRegionData
_cshmwrap_set_shared_memory_region_data.restype = c_void_p
_cshmwrap_set_shared_memory_region_data.argtypes = [c_int, c_uint64, c_uint64, POINTER(c_void_p)]
_cshmwrap_read_shared_memory_region_data = _cshmwrap.ReadSharedMemoryRegionData
_cshmwrap_read_shared_memory_region_data.restype = c_void_p
_cshmwrap_read_shared_memory_region_data.argtypes = [c_int, c_uint64, c_uint64, POINTER(c_void_p)]
_cshmwrap_unlink_shared_memory_region = _cshmwrap.UnlinkSharedMemoryRegion
_cshmwrap_unlink_shared_memory_region.restype = c_void_p
_cshmwrap_unlink_shared_memory_region.argtypes = [POINTER(c_char_p)]

_cshmwrap_unmap_shared_memory_region = _cshmwrap.UnmapSharedMemory
_cshmwrap_unmap_shared_memory_region.restype = c_void_p
_cshmwrap_unmap_shared_memory_region.argtypes = [c_void_p, c_uint64]

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

        if http_headers is None:
            http_headers = list()

        http_headers_arr = (c_char_p * len(http_headers))()
        http_headers_arr[:] = http_headers

        _raise_if_error(
            c_void_p(
                _crequest_health_ctx_new(
                    byref(self._ctx), url, int(protocol),
                    http_headers_arr, len(http_headers), verbose)))

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

        if http_headers is None:
            http_headers = list()

        http_headers_arr = (c_char_p * len(http_headers))()
        http_headers_arr[:] = http_headers

        _raise_if_error(
            c_void_p(
                _crequest_status_ctx_new(
                    byref(self._ctx), url, int(protocol), http_headers_arr, len(http_headers),
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
        status_buf = cast(cstatus, POINTER(c_byte * cstatus_len.value))[0]

        status = ServerStatus()
        status.ParseFromString(status_buf)
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

        if http_headers is None:
            http_headers = list()

        http_headers_arr = (c_char_p * len(http_headers))()
        http_headers_arr[:] = http_headers

        _raise_if_error(
            c_void_p(
                _crequest_model_control_ctx_new(
                    byref(self._ctx), url, int(protocol),
                    http_headers_arr, len(http_headers), verbose)))

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

        if http_headers is None:
            http_headers = list()

        http_headers_arr = (c_char_p * len(http_headers))()
        http_headers_arr[:] = http_headers

        _raise_if_error(
            c_void_p(
                _crequest_shm_control_ctx_new(
                    byref(self._ctx), url, int(protocol),
                    http_headers_arr, len(http_headers), verbose)))

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

    def create_shared_memory_region(self, shm_key, byte_size):
        """Creates a shared memory region with the specified name and size.

        Parameters
        ----------
        shm_key : str
            The unique key of the shared memory object.
        byte_size : int
            The size in bytes of the shared memory region to be created.

        Returns
        -------
        shm_fd : int
            The unique shared memory region identifier.

        Raises
        ------
        InferenceServerException
            If unable to create the shared memory region.
        """

        if self._ctx is None:
            _raise_error("SharedMemoryControlContext is closed")

        shm_fd = c_int()
        _raise_if_error(
            c_void_p(_cshmwrap_create_shared_memory_region(shm_key, byte_size, byref(shm_fd))))

        return shm_fd

    def open_shared_memory_region(self, shm_key):
        """Opens a shared memory region with the specified name.

        Parameters
        ----------
        shm_key : str
            The unique key of the shared memory object.

        Returns
        -------
        shm_fd : int
            The unique shared memory region descriptor.

        Raises
        ------
        InferenceServerException
            If unable to open the shared memory region.
        """

        shm_fd = c_int()
        _raise_if_error(
            c_void_p(_cshmwrap_open_shared_memory_region(shm_key, byref(shm_fd))))

        return shm_fd

    def close_shared_memory_region(self, shm_fd):
        """Closes a shared memory region with the specified identifier.

        Parameters
        ----------
        shm_fd : int
            The unique shared memory region identifier.

        Raises
        ------
        InferenceServerException
            If unable to close the shared memory region.
        """

        _raise_if_error(
            c_void_p(_cshmwrap_close_shared_memory_region(shm_fd)))
        return

    def set_shared_memory_region_data(self, shm_fd, offset, input_values):
        """Copy the contents of the numpy array into a shared memory region with
        the specified identifier, offset and size.

        Parameters
        ----------
        shm_fd : int
            The unique shared memory region identifier.
        offset : int
            The offset from the start of the shared shared memory region.
        input_values : void*
            The pointer to the values in the numpy array to be copied into the
            shared memory region.

        Raises
        ------
        InferenceServerException
            If unable to mmap or set values in the shared memory region.
        """

        if not isinstance(input_values, (np.ndarray,)):
            _raise_error("input values must be specified as a list of numpy arrays")

        input_values = np.ascontiguousarray(input_values)
        _raise_if_error(
            c_void_p(_cshmwrap_set_shared_memory_region_data(shm_fd, offset,
                c_uint64(input_values.size * input_values.itemsize),
                input_values.ctypes.data_as(c_void_p))))
        return

    def read_shared_memory_region_data(self, shm_fd, offset, byte_size, result_dtype, shape, batch_size):
        """Copy the contents of the numpy array into a shared memory region with
        the specified identifier, offset and size.

        Parameters
        ----------
        shm_fd : int
            The unique shared memory region identifier.
        offset : int
            The offset from the start of the shared shared memory region.
        byte_size : int
            The size in bytes of the data to be read from the shared memory region.

        Returns
        -------
        output_values : list
            The list of numpy arrays read from the shared memory region.

        Raises
        ------
        InferenceServerException
            If unable to mmap or set values in the shared memory region.
        """

        output_base_pointer = c_void_p()
        _raise_if_error(
            c_void_p(_cshmwrap_read_shared_memory_region_data(shm_fd, offset, byte_size, byref(output_base_pointer))))

        output_values = list()
        for b in range(batch_size):
            # Get the result value into a 1-dim np array
            # of the appropriate type
            cval = output_base_pointer
            cval_len = byte_size

            val_buf = cast(cval, POINTER(c_byte * cval_len))[0]

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
            # output_val = np.copy(val)
            # del val
            # del val_buf
            shaped = np.reshape(np.copy(val), shape)
            output_values.append(shaped)

        return output_values

    def unlink_shared_memory_region(self, shm_key):
        """Unlink a shared memory region with the specified name.

        Parameters
        ----------
        shm_key : str
            The unique key of the shared memory object.

        Raises
        ------
        InferenceServerException
            If unable to unlink the shared memory region.
        """

        _raise_if_error(
            c_void_p(_cshmwrap_unlink_shared_memory_region(shm_key)))
        return

    def unmap_shared_memory_region(self, shm_addr, byte_size):
        """Unmap a shared memory region with the specified name and size.

        Parameters
        ----------
        shm_addr : void*
            The base address of the shared memory region.
        byte_size : int
            The size in bytes of the data in the shared memory region.

        Raises
        ------
        InferenceServerException
            If unable to munmap the shared memory region.
        """

        _raise_if_error(
            c_void_p(_cshmwrap_unmap_shared_memory_region(shm_addr, byte_size)))
        return

    def register(self, name, shm_key, offset, byte_size):
        """Request the inference server to register specified shared memory region.

        Parameters
        ----------
        name : str
            The name of the shared memory region to be registered.
        shm_key : str
            The unique key of the shared memory object.
        offset : int
            The offset from the start of the shared shared memory region.
        byte_size : int
            The size in bytes of the data to be read / written in the shared
            memory region.

        Raises
        ------
        InferenceServerException
            If unable to register the shared memory region.

        """
        self._last_request_id = None
        if self._ctx is None:
            _raise_error("SharedMemoryControlContext is closed")

        self._last_request_id = _raise_if_error(
            c_void_p(_crequest_shm_control_ctx_register(self._ctx, name, shm_key, offset, byte_size)))
        return

    def unregister(self, name):
        """Request the inference server to unregister specified shared memory region.

        Parameters
        ----------
        name : str
            The name of the shared memory region to be unregistered.

        Raises
        ------
        InferenceServerException
            If unable to unregister the shared memory region.

        """
        self._last_request_id = None
        if self._ctx is None:
            _raise_error("SharedMemoryControlContext is closed")

        self._last_request_id = _raise_if_error(
            c_void_p(_crequest_shm_control_ctx_unregister(self._ctx, name)))
        return

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

        if http_headers is None:
            http_headers = list()

        http_headers_arr = (c_char_p * len(http_headers))()
        http_headers_arr[:] = http_headers

        imodel_version = -1 if model_version is None else model_version
        _raise_if_error(
            c_void_p(
                _crequest_infer_ctx_new(
                    byref(self._ctx), url, int(protocol),
                    http_headers_arr, len(http_headers),
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
                         flags, batch_size, contiguous_input_values):
        # Make sure each input is given as a list (one entry per
        # batch). It is a common error when using batch-size 1 to
        # specify an input directly as an array instead of as a list
        # containing one array.
        # Each element in the list must be a 'numpy array' if inputs are being
        # passed directly and a 'tuple' if passing a reference to shared memory
        for inp_name, inp in inputs.items():
            if not isinstance(inp, (list, tuple)):
                _raise_error("input '" + inp_name +
                             "' values must be specified as a list of numpy arrays" \
                             " or list of tuples representing location in shared memory")
            if isinstance(inp[0], (np.ndarray,)):
                for ip in inp:
                    if not isinstance(ip, (np.ndarray, tuple)):
                        _raise_error("input '" + inp_name +
                                     "' values must be specified as a list of numpy arrays" \
                                     " or list of tuples representing location in shared memory")
            else:
                if (len(inp) != 3) or (type(inp[0]) != str) \
                    or (type(inp[1]) != int) or (type(inp[2]) != int):
                    _raise_error("shared memory requires tuple of size 3" \
                        + " - shm_key (string), offset (int), size (int)")
        # Set run options using formats specified in 'outputs'
        options = c_void_p()
        try:
            _raise_if_error(c_void_p(
                _crequest_infer_ctx_options_new(byref(options), flags, batch_size)))

            for (output_name, output_format) in iteritems(outputs):
                if len(output_format) == 4 and isinstance(output_format, (list, tuple)):
                    if output_format[0] != InferContext.ResultFormat.RAW \
                    or (type(output_format[1]) != str) or (type(output_format[2]) != int) \
                        or (type(output_format[3]) != int):
                        _raise_error("shared memory requires tuple of size 4" \
                            + " - output_format(RAW), shm_key (string), offset (int), size (int)")
                    _raise_if_error(
                        c_void_p(
                            _crequest_infer_ctx_options_add_shared_memory(
                                self._ctx, options, output_name, output_format[1], c_uint64(output_format[2]),
                                c_uint64(output_format[3]))))
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
                if len(input_values) > 0:
                    if isinstance(input_values[0], (np.ndarray,)):
                        shape_value = np.asarray(input_values[0].shape, dtype=np.int64)
                        _raise_if_error(
                            c_void_p(
                                _crequest_infer_ctx_input_set_shape(
                                       input, shape_value, c_uint64(shape_value.size))))

               # # use values if numpy array, reference if shared memory
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
                            if input_value.dtype == np.object or input_value.dtype.type == np.bytes_:
                                flattened = bytes()
                                for obj in np.nditer(input_value, flags=["refs_ok"], order='C'):
                                    # If directly passing bytes to STRING type,
                                    # don't convert it to str as Python will encode the
                                    # bytes which may distort the meaning
                                    if obj.dtype.type == np.bytes_:
                                        s = bytes(obj)
                                    else:
                                        s = str(obj).encode('utf-8')
                                    flattened += struct.pack("<I", len(s))
                                    flattened += s
                                input_value = np.asarray(flattened)

                            if not input_value.flags['C_CONTIGUOUS']:
                                input_value = np.ascontiguousarray(input_value)
                            contiguous_input_values.append(input_value)
                            _raise_if_error(
                                c_void_p(
                                    _crequest_infer_ctx_input_set_raw(
                                        input, input_value.ctypes.data_as(c_void_p),
                                        c_uint64(input_value.size * input_value.itemsize))))
                else:
                    _raise_if_error(
                        c_void_p(
                            _crequest_infer_ctx_input_set_shared_memory(
                                input, input_values[0], c_uint64(input_values[1]),
                                c_uint64(input_values[2]))))

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
                    (output_format[0] == InferContext.ResultFormat.RAW) and (len(output_format) == 4)):
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
                    results[output_name].append(result_dtype)
                    results[output_name].append(shape)
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
        return self._correlation_id

    def run(self, inputs, outputs, batch_size=1, flags=0):
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
        self._prepare_request(inputs, outputs, flags, batch_size, contiguous_input)

        # Run inference...
        self._last_request_id = _raise_if_error(c_void_p(_crequest_infer_ctx_run(self._ctx)))

        return self._get_results(outputs, batch_size)

    def async_run(self, inputs, outputs, batch_size=1, flags=0):
        """DEPRECATED: This function is deprecated and will be removed in
        a future version of this API. Instead use async_run_with_cb().

        Run inference using the supplied 'inputs' to calculate the outputs
        specified by 'outputs'.

        Unlike run(), async_run() returns immediately after sending
        the inference request to the server. The returned integer
        identifier must be used subsequently to wait on and retrieve
        the actual inference results.

        Parameters
        ----------
        inputs : dict
            Dictionary from input name to the value(s) for that
            input. An input value is specified as a numpy array. Each
            input in the dictionary maps to a list of values (i.e. a
            list of numpy array objects), where the length of the list
            must equal the 'batch_size'.

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

        Returns
        -------
        int
            Integer identifier which must be passed to
            get_async_run_results() to wait on and retrieve the
            inference results.

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
        self._prepare_request(inputs, outputs, flags, batch_size, contiguous_input)

        # Run asynchronous inference...
        c_request_id = c_uint64()
        _raise_if_error(
            c_void_p(
                _crequest_infer_ctx_async_run(self._ctx, byref(c_request_id))))

        self._requested_outputs_dict[c_request_id.value] = (outputs, batch_size, contiguous_input)

        return c_request_id.value

    def async_run_with_cb(self, callback, inputs, outputs, batch_size=1, flags=0):
        """Run inference using the supplied 'inputs' to calculate the outputs
        specified by 'outputs'.

        Similar to AsyncRun() above. However, this function does not return the
        integer identifier. Instead, once the request is completed, the InferContext
        object and the integer identifier will be passed to the provided 'callback'
        function. It is the function caller's choice on either retrieving the results
        inside the callback function or deferring it to a different thread so that
        the InferContext is unblocked.

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

        Returns
        -------
        int
            Integer identifier which must be passed to
            get_async_run_results() to wait on and retrieve the
            inference results.

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
        self._prepare_request(inputs, outputs, flags, batch_size, contiguous_input)

        # Wrap over the provided callback
        wrapped_cb = partial(self._async_callback_wrapper, self._callback_resources_dict_id, callback)
        c_cb = _async_run_callback_prototype(wrapped_cb)

        # Run asynchronous inference...
        _raise_if_error(
            c_void_p(
                _crequest_infer_ctx_async_run_with_cb(self._ctx, c_cb)))

        self._callback_resources_dict[self._callback_resources_dict_id] = \
            (outputs, batch_size, contiguous_input, c_cb, wrapped_cb)
        self._callback_resources_dict_id += 1

    def get_async_run_results(self, request_id, wait):
        """Retrieve the results of a previous async_run() using the supplied
        'request_id'

        Parameters
        ----------
        request_id : int
            The integer ID of the asynchronous request returned by async_run().

        wait : bool
            If True block until the request results are ready. If False return
            immediately even if results are not ready.

        Returns
        -------
        dict
            None if the results are not ready and 'wait' is False. A
            dictionary from output name to the list of values for that
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
        c_is_ready = c_bool()
        err = c_void_p(_crequest_infer_ctx_get_async_run_results(
            self._ctx, byref(c_is_ready), request_id, wait))


        self._last_request_id = _raise_if_error(err)

        if not c_is_ready.value:
            return None

        requested_outputs = self._requested_outputs_dict[request_id]
        if isinstance(requested_outputs, int):
            idx = requested_outputs
            requested_outputs = self._callback_resources_dict[idx]
            del self._callback_resources_dict[idx]
        del self._requested_outputs_dict[request_id]

        return self._get_results(requested_outputs[0], requested_outputs[1], request_id)

    def get_ready_async_request(self, wait):
        """DEPRECATED: This function is deprecated and will be removed in
        a future version of this API. This function is only useful with
        the deprecated version of async_run(). Instead use async_run_with_cb().

        Get the request ID of an async_run() request that has completed but
        not yet had results read with get_async_run_results().

        Parameters
        ----------
        wait : bool
            If True block until an async request is ready. If False return
            immediately even if results are not ready.

        Returns
        -------
        int
            None if no asynchronous results are ready and 'wait' is
            False. An integer identifier which must be passed to
            get_async_run_results() to wait on and retrieve the
            inference results.

        Raises
        ------
        InferenceServerException
            If no asynchronous request is in flight or completed.

        """
        # Get async run results
        c_is_ready = c_bool()
        c_request_id = c_uint64()
        err = c_void_p(_crequest_infer_ctx_get_ready_async_request(
            self._ctx, byref(c_is_ready), byref(c_request_id), wait))

        _raise_if_error(err)

        if not c_is_ready.value:
            return None

        return c_request_id.value

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
