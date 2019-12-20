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
_ccudashm_lib = "ccudashm" if os.name == 'nt' else 'libccudashm.so'
_ccudashm_path = pkg_resources.resource_filename('tensorrtserver.cuda_shared_memory', _ccudashm_lib)
_ccudashm = cdll.LoadLibrary(_ccudashm_path)

_ccudashm_shared_memory_region_create = _ccudashm.CudaSharedMemoryRegionCreate
_ccudashm_shared_memory_region_create.restype = c_int
_ccudashm_shared_memory_region_create.argtypes = [_utf8, c_uint64, c_uint64, POINTER(c_void_p)]
_ccudashm_shared_memory_region_set = _ccudashm.CudaSharedMemoryRegionSet
_ccudashm_shared_memory_region_set.restype = c_int
_ccudashm_shared_memory_region_set.argtypes = [c_void_p, c_uint64, c_uint64, c_void_p]
_ccudashm_shared_memory_region_destroy = _ccudashm.CudaSharedMemoryRegionDestroy
_ccudashm_shared_memory_region_destroy.restype = c_int
_ccudashm_shared_memory_region_destroy.argtypes = [c_void_p]

def _raise_if_error(errno):
    """
    Raise CudaSharedMemoryException if 'err' is non-success.
    Otherwise return nothing.
    """
    if errno.value != 0:
        ex = CudaSharedMemoryException(errno)
        raise ex
    return

def _raise_error(msg):
    ex = CudaSharedMemoryException(msg)
    raise ex

def create_shared_memory_region(trtis_shm_name, byte_size, device_id):
    """Creates a shared memory region with the specified name and size.

    Parameters
    ----------
    trtis_shm_name : str
        The unique name of the cuda shared memory region to be created.
    byte_size : int
        The size in bytes of the cuda shared memory region to be created.
    device_id : int
        The GPU device ID of the cuda shared memory region to be created.
    Returns
    -------
    cuda_shm_handle : c_void_p
        The handle for the cuda shared memory region.

    Raises
    ------
    CudaSharedMemoryException
        If unable to create the cuda shared memory region on the specified device.
    """

    cuda_shm_handle = c_void_p()
    _raise_if_error(
        c_int(_ccudashm_shared_memory_region_create(trtis_shm_name, byte_size, device_id, byref(cuda_shm_handle))))

    return cuda_shm_handle

def set_shared_memory_region(cuda_shm_handle, input_values):
    """Copy the contents of the numpy array into a shared memory region with
    the specified identifier, offset and size.

    Parameters
    ----------
    cuda_shm_handle : c_void_p
        The handle for the cuda shared memory region.
    input_values : np.array
        The list of numpy arrays to be copied into the shared memory region.

    Raises
    ------
    CudaSharedMemoryException
        If unable to set values in the cuda shared memory region.
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
            c_int(_ccudashm_shared_memory_region_set(cuda_shm_handle, c_uint64(offset_current), \
                c_uint64(byte_size), input_value.ctypes.data_as(c_void_p))))
        offset_current += byte_size
    return

def destroy_shared_memory_region(cuda_shm_handle):
    """Close a cuda shared memory region with the specified handle.

    Parameters
    ----------
    cuda_shm_handle : c_void_p
        The handle for the cuda shared memory region.

    Raises
    ------
    CudaSharedMemoryException
        If unable to close the cuda_shm_handle shared memory region and free the device memory.
    """

    _raise_if_error(
        c_int(_ccudashm_shared_memory_region_destroy(cuda_shm_handle)))
    return

class CudaSharedMemoryException(Exception):
    """Exception indicating non-Success status.

    Parameters
    ----------
    err : c_void_p
        Pointer to an Error that should be used to initialize the exception.

    """
    def __init__(self, err):
        self.err_code_map = { -1: "unable to set device successfully",
                            -2: "unable to create cuda shared memory handle",
                            -3: "unable to set values in cuda shared memory",
                            -4: "unable to free GPU device memory"}
        self._msg = None
        if type(err) == str:
            self._msg = err
        elif err.value != 0 and err.value in self.err_code_map:
            self._msg = self.err_code_map[err.value]

    def __str__(self):
        msg = super().__str__() if self._msg is None else self._msg
        return msg
