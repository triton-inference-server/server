# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
_cshm_lib = "cshm" if os.name == 'nt' else 'libcshm.so'
_cshm_path = pkg_resources.resource_filename('tensorrtserver.shared_memory', _cshm_lib)
_cshm = cdll.LoadLibrary(_cshm_path)

_cshm_shared_memory_region_create = _cshm.SharedMemoryRegionCreate
_cshm_shared_memory_region_create.restype = c_int
_cshm_shared_memory_region_create.argtypes = [_utf8, _utf8, c_uint64, POINTER(c_void_p)]
_cshm_shared_memory_region_set = _cshm.SharedMemoryRegionSet
_cshm_shared_memory_region_set.restype = c_int
_cshm_shared_memory_region_set.argtypes = [c_void_p, c_uint64, c_uint64, c_void_p]
_cshm_shared_memory_region_destroy = _cshm.SharedMemoryRegionDestroy
_cshm_shared_memory_region_destroy.restype = c_int
_cshm_shared_memory_region_destroy.argtypes = [c_void_p]

def _raise_if_error(errno):
    """
    Raise SharedMemoryException if 'err' is non-success.
    Otherwise return nothing.
    """
    if errno.value != 0:
        ex = SharedMemoryException(errno)
        raise ex
    return

def _raise_error(msg):
    ex = SharedMemoryException(msg)
    raise ex

def create_shared_memory_region(trtis_shm_name, shm_key, byte_size):
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
    SharedMemoryException
        If unable to create the shared memory region.
    """

    shm_handle = c_void_p()
    _raise_if_error(
        c_int(_cshm_shared_memory_region_create(trtis_shm_name, shm_key, byte_size, byref(shm_handle))))

    return shm_handle

def set_shared_memory_region(shm_handle, input_values):
    """Copy the contents of the numpy array into a shared memory region with
    the specified identifier, offset and size.

    Parameters
    ----------
    shm_handle : c_void_p
        The handle for the shared memory region.
    input_values : np.array
        The list of numpy arrays to be copied into the shared memory region.

    Raises
    ------
    SharedMemoryException
        If unable to mmap or set values in the shared memory region.
    """

    if not isinstance(input_values, (list,tuple)):
        _raise_error("input_values must be specified as a numpy array")
    for input_value in input_values:
        if not isinstance(input_value, (np.ndarray,)):
            _raise_error("input_values must be specified as a list/tuple of numpy arrays")

    offset_current = 0
    for input_value in input_values:
        input_value = np.ascontiguousarray(input_value).flatten()
        byte_size = input_value.size * input_value.itemsize
        _raise_if_error(
            c_int(_cshm_shared_memory_region_set(shm_handle, c_uint64(offset_current), \
                c_uint64(byte_size), input_value.ctypes.data_as(c_void_p))))
        offset_current += byte_size
    return

def destroy_shared_memory_region(shm_handle):
    """Unlink a shared memory region with the specified name.

    Parameters
    ----------
    shm_handle : c_void_p
        The handle for the shared memory region.

    Raises
    ------
    SharedMemoryException
        If unable to unlink the shared memory region.
    """

    _raise_if_error(
        c_int(_cshm_shared_memory_region_destroy(shm_handle)))
    return

class SharedMemoryException(Exception):
    """Exception indicating non-Success status.

    Parameters
    ----------
    err : c_void_p
        Pointer to an Error that should be used to initialize the exception.

    """
    def __init__(self, err):
        self.err_code_map = { -2: "unable to get shared memory descriptor",
                            -3: "unable to initialize the size",
                            -4: "unable to read/mmap the shared memory region",
                            -5: "unable to unlink the shared memory region"}
        self._msg = None
        if err.value != 0 and err.value in self.err_code_map:
            self._msg = self.err_code_map[err.value]

    def __str__(self):
        msg = super().__str__() if self._msg is None else self._msg
        return msg
