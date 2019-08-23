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
_cshmwrap_lib = "shmwrap" if os.name == 'nt' else 'libcshmwrap.so'
_cshmwrap_path = pkg_resources.resource_filename('tensorrtserver.shared_memory', _cshmwrap_lib)
_cshmwrap = cdll.LoadLibrary(_cshmwrap_path)

_cshmwrap_create_shared_memory_region = _cshmwrap.CreateSharedMemoryRegion
_cshmwrap_create_shared_memory_region.restype = c_void_p
_cshmwrap_create_shared_memory_region.argtypes = [_utf8, c_uint64, POINTER(c_int)]
_cshmwrap_open_shared_memory_region = _cshmwrap.OpenSharedMemoryRegion
_cshmwrap_open_shared_memory_region.restype = c_void_p
_cshmwrap_open_shared_memory_region.argtypes = [c_char_p, POINTER(c_int)]
_cshmwrap_close_shared_memory_region = _cshmwrap.CloseSharedMemoryRegion
_cshmwrap_close_shared_memory_region.restype = c_void_p
_cshmwrap_close_shared_memory_region.argtypes = [c_uint64]
_cshmwrap_set_shared_memory_region_data = _cshmwrap.SetSharedMemoryRegionData
_cshmwrap_set_shared_memory_region_data.restype = c_void_p
_cshmwrap_set_shared_memory_region_data.argtypes = [c_void_p, c_uint64, c_uint64, c_void_p]
_cshmwrap_map_shared_memory_region = _cshmwrap.MapSharedMemoryRegion
_cshmwrap_map_shared_memory_region.restype = c_void_p
_cshmwrap_map_shared_memory_region.argtypes = [c_int, c_uint64, c_uint64, POINTER(c_void_p)]
_cshmwrap_unlink_shared_memory_region = _cshmwrap.UnlinkSharedMemoryRegion
_cshmwrap_unlink_shared_memory_region.restype = c_void_p
_cshmwrap_unlink_shared_memory_region.argtypes = [c_char_p]
_cshmwrap_create_shared_memory_handle = _cshmwrap.CreateSharedMemoryHandle
_cshmwrap_create_shared_memory_handle.restype = c_void_p
_cshmwrap_create_shared_memory_handle.argtypes = [c_void_p, _utf8, c_int, POINTER(c_void_p)]
_cshmwrap_get_shared_memory_handle_info = _cshmwrap.GetSharedMemoryHandleInfo
_cshmwrap_get_shared_memory_handle_info.restype = c_void_p
_cshmwrap_get_shared_memory_handle_info.argtypes = [c_void_p, POINTER(c_void_p), POINTER(c_char_p), POINTER(c_int)]
_cshmwrap_unmap_shared_memory_region = _cshmwrap.UnmapSharedMemoryRegion
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

        shm_fd = c_int()
        _raise_if_error(
            c_void_p(_cshmwrap_create_shared_memory_region(shm_key, byte_size, byref(shm_fd))))

        shm_addr = c_void_p()
        _raise_if_error(
            c_void_p(_cshmwrap_map_shared_memory_region(shm_fd, 0, byte_size, byref(shm_addr))))

        shm_handle = c_void_p()
        _raise_if_error(
            c_void_p(_cshmwrap_create_shared_memory_handle(shm_addr, shm_key, shm_fd, byref(shm_handle))))

        return shm_handle

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

    def set_shared_memory_region_data(self, shm_handle, offset, input_values):
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

        shm_fd = c_int()
        shm_addr = c_void_p()
        shm_key = c_char_p()
        _raise_if_error(
            c_void_p(_cshmwrap_get_shared_memory_handle_info(shm_handle, byref(shm_addr), byref(shm_key), byref(shm_fd))))

        offset_current = offset
        for input_value in input_values:
            input_value = np.ascontiguousarray(input_value).flatten()
            byte_size = input_value.size * input_value.itemsize
            _raise_if_error(
                c_void_p(_cshmwrap_set_shared_memory_region_data(shm_addr, c_uint64(offset_current), \
                    c_uint64(byte_size), input_value.ctypes.data_as(c_void_p))))
            offset_current += byte_size
        return

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

    def map_shared_memory_region(self, shm_fd, offset, byte_size):
        """Unmap a shared memory region with the specified name and size.

        Parameters
        ----------
        shm_fd : int
            The unique shared memory region identifier.
        offset : int
            The offset from the start of the shared shared memory region.
        byte_size : int
            The size in bytes of the shared memory region.

        Returns
        -------
        shm_addr : c_void_p
            The base address of the shared memory region.

        Raises
        ------
        InferenceServerException
            If unable to munmap the shared memory region.
        """
        shm_addr = c_void_p()
        _raise_if_error(
            c_void_p(_cshmwrap_map_shared_memory_region(shm_fd, offset, byte_size, byref(shm_addr))))

        return shm_addr

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
