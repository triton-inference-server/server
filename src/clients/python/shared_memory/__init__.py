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
_cshm_lib = "shmwrap" if os.name == 'nt' else 'libcshm.so'
_cshm_path = pkg_resources.resource_filename('tensorrtserver.shared_memory', _cshm_lib)
_cshm = cdll.LoadLibrary(_cshm_path)

_cshm_shared_memory_region_create = _cshm.SharedMemoryRegionCreate
_cshm_shared_memory_region_create.restype = c_void_p
_cshm_shared_memory_region_create.argtypes = [_utf8, c_uint64, POINTER(c_void_p)]
_cshm_shared_memory_region_set = _cshm.SharedMemoryRegionSet
_cshm_shared_memory_region_set.restype = c_void_p
_cshm_shared_memory_region_set.argtypes = [c_void_p, c_uint64, c_uint64, c_void_p]
_cshm_shared_memory_region_destroy = _cshm.SharedMemoryRegionDestroy
_cshm_shared_memory_region_destroy.restype = c_void_p
_cshm_shared_memory_region_destroy.argtypes = [c_char_p]

_cshm_error_new =  _cshm.ErrorNew
_cshm_error_new.restype = c_void_p
_cshm_error_new.argtypes = [_utf8]
_cshm_error_del =  _cshm.ErrorDelete
_cshm_error_del.argtypes = [c_void_p]
_cshm_error_isok =  _cshm.ErrorIsOk
_cshm_error_isok.restype = c_bool
_cshm_error_isok.argtypes = [c_void_p]
_cshm_error_msg =  _cshm.ErrorMessage
_cshm_error_msg.restype = c_char_p
_cshm_error_msg.argtypes = [c_void_p]
_cshm_error_serverid =  _cshm.ErrorServerId
_cshm_error_serverid.restype = c_char_p
_cshm_error_serverid.argtypes = [c_void_p]
_cshm_error_requestid =  _cshm.ErrorRequestId
_cshm_error_requestid.restype = c_int64
_cshm_error_requestid.argtypes = [c_void_p]

def _raise_if_error(err):
    """
    Raise InferenceServerException if 'err' is non-success.
    Otherwise return the request ID.
    """
    if err.value is not None:
        ex = InferenceServerException(err)
        isok = _cshm_error_isok(err)
        _cshm_error_del(err)
        if not isok:
            raise ex
        return ex.request_id()
    return 0

def _raise_error(msg):
    err = c_void_p(_cshm_error_new(msg))
    ex = InferenceServerException(err)
    _cshm_error_del(err)
    raise ex


class SharedMemoryHelper:
    """Client helper functions for using shared memory in python.
    """

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
        shm_handle : c_void_p
            The handle for the shared memory region.

        Raises
        ------
        InferenceServerException
            If unable to create the shared memory region.
        """

        shm_handle = c_void_p()
        _raise_if_error(
            c_void_p(_cshm_shared_memory_region_create(shm_key, byte_size, byref(shm_handle))))

        return shm_handle

    def set_shared_memory_region(self, shm_handle, offset, input_values):
        """Copy the contents of the numpy array into a shared memory region with
        the specified identifier, offset and size.

        Parameters
        ----------
        shm_handle : c_void_p
            The handle for the shared memory region.
        offset : int
            The offset from the start of the shared shared memory region.
        input_values : np.array
            The list of numpy arrays to be copied into the shared memory region.

        Raises
        ------
        InferenceServerException
            If unable to mmap or set values in the shared memory region.
        """

        if not isinstance(input_values, (list,tuple)):
            _raise_error("input_values must be specified as a numpy array")
        for input_value in input_values:
            if not isinstance(input_value, (np.ndarray,)):
                _raise_error("input_values must be specified as a list/tuple of numpy arrays")

        offset_current = offset
        for input_value in input_values:
            input_value = np.ascontiguousarray(input_value).flatten()
            byte_size = input_value.size * input_value.itemsize
            _raise_if_error(
                c_void_p(_cshm_shared_memory_region_set(shm_handle, c_uint64(offset_current), \
                    c_uint64(byte_size), input_value.ctypes.data_as(c_void_p))))
            offset_current += byte_size
        return

    def destroy_shared_memory_region(self, shm_key):
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
            c_void_p(_cshm_shared_memory_region_destroy(shm_key)))
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
            c_void_p(_cshm_unmap_shared_memory_region(shm_addr, byte_size)))
        return


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
            self._msg = _cshm_error_msg(err)
            if self._msg is not None:
                self._msg = self._msg.decode('utf-8')
            self._server_id = _cshm_error_serverid(err)
            if self._server_id is not None:
                self._server_id = self._server_id.decode('utf-8')
            self._request_id = _cshm_error_requestid(err)

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
