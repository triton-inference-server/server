# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

import os
from ctypes import *
import numpy as np
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


_cshm_lib = "cshm" if os.name == 'nt' else 'libcshm.so'
_cshm_path = pkg_resources.resource_filename('tritonclient.utils.shared_memory',
                                             _cshm_lib)
_cshm = cdll.LoadLibrary(_cshm_path)

_cshm_shared_memory_region_create = _cshm.SharedMemoryRegionCreate
_cshm_shared_memory_region_create.restype = c_int
_cshm_shared_memory_region_create.argtypes = [
    _utf8, _utf8, c_uint64, POINTER(c_void_p)
]
_cshm_shared_memory_region_set = _cshm.SharedMemoryRegionSet
_cshm_shared_memory_region_set.restype = c_int
_cshm_shared_memory_region_set.argtypes = [
    c_void_p, c_uint64, c_uint64, c_void_p
]
_cshm_get_shared_memory_handle_info = _cshm.GetSharedMemoryHandleInfo
_cshm_get_shared_memory_handle_info.restype = c_int
_cshm_get_shared_memory_handle_info.argtypes = [
    c_void_p,
    POINTER(c_char_p),
    POINTER(c_char_p),
    POINTER(c_int),
    POINTER(c_uint64),
    POINTER(c_uint64)
]
_cshm_shared_memory_region_destroy = _cshm.SharedMemoryRegionDestroy
_cshm_shared_memory_region_destroy.restype = c_int
_cshm_shared_memory_region_destroy.argtypes = [c_void_p]

mapped_shm_regions = []


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


def create_shared_memory_region(triton_shm_name, shm_key, byte_size):
    """Creates a system shared memory region with the specified name and size.

    Parameters
    ----------
    triton_shm_name : str
        The unique name of the shared memory region to be created.
    shm_key : str
        The unique key of the shared memory object.
    byte_size : int
        The size in bytes of the shared memory region to be created.

    Returns
    -------
    shm_handle : c_void_p
        The handle for the system shared memory region.

    Raises
    ------
    SharedMemoryException
        If unable to create the shared memory region.
    """

    shm_handle = c_void_p()
    _raise_if_error(
        c_int(
            _cshm_shared_memory_region_create(triton_shm_name, shm_key,
                                              byte_size, byref(shm_handle))))
    mapped_shm_regions.append(shm_key)

    return shm_handle


def set_shared_memory_region(shm_handle, input_values):
    """Copy the contents of the numpy array into the system shared memory region.

    Parameters
    ----------
    shm_handle : c_void_p
        The handle for the system shared memory region.
    input_values : list
        The list of numpy arrays to be copied into the shared memory region.

    Raises
    ------
    SharedMemoryException
        If unable to mmap or set values in the system shared memory region.
    """

    if not isinstance(input_values, (list, tuple)):
        _raise_error(
            "input_values must be specified as a list/tuple of numpy arrays")
    for input_value in input_values:
        if not isinstance(input_value, np.ndarray):
            _raise_error("each element of input_values must be a numpy array")

    offset_current = 0
    for input_value in input_values:
        input_value = np.ascontiguousarray(input_value).flatten()
        if input_value.dtype == np.object_:
            input_value = input_value.item()
            byte_size = np.dtype(np.byte).itemsize * len(input_value)
            _raise_if_error(
                c_int(_cshm_shared_memory_region_set(shm_handle, c_uint64(offset_current), \
                    c_uint64(byte_size), cast(input_value, c_void_p))))
        else:
            byte_size = input_value.size * input_value.itemsize
            _raise_if_error(
                c_int(_cshm_shared_memory_region_set(shm_handle, c_uint64(offset_current), \
                    c_uint64(byte_size), input_value.ctypes.data_as(c_void_p))))
        offset_current += byte_size
    return


def get_contents_as_numpy(shm_handle, datatype, shape):
    """Generates a numpy array using the data stored in the system shared memory
    region specified with the handle.

    Parameters
    ----------
    shm_handle : c_void_p
        The handle for the system shared memory region.
    datatype : np.dtype
        The datatype of the array to be returned.
    shape : list
        The list of int describing the shape of the array to be returned.

    Returns
    -------
    np.array
        The numpy array generated using the contents of the specified shared
        memory region.
    """
    shm_fd = c_int()
    offset = c_uint64()
    byte_size = c_uint64()
    shm_addr = c_char_p()
    shm_key = c_char_p()
    _raise_if_error(
            c_int(_cshm_get_shared_memory_handle_info(shm_handle, byref(shm_addr), byref(shm_key), byref(shm_fd), \
                                    byref(offset), byref(byte_size))))
    start_pos = offset.value
    if (datatype != np.object_) and (datatype != np.bytes_):
        requested_byte_size = np.prod(shape) * np.dtype(datatype).itemsize
        cval_len = start_pos + requested_byte_size
        if byte_size.value < cval_len:
            _raise_error(
                "The size of the shared memory region is unsufficient to provide numpy array with requested size"
            )
        if cval_len == 0:
            result = np.empty(shape, dtype=datatype)
        else:
            val_buf = cast(shm_addr, POINTER(c_byte * cval_len))[0]
            val = np.frombuffer(val_buf, dtype=datatype, offset=start_pos)

            # Reshape the result to the appropriate shape.
            result = np.reshape(val, shape)
    else:
        str_offset = start_pos
        val_buf = cast(shm_addr, POINTER(c_byte * byte_size.value))[0]
        ii = 0
        strs = list()
        while (ii % np.prod(shape) != 0) or (ii == 0):
            l = struct.unpack_from("<I", val_buf, str_offset)[0]
            str_offset += 4
            sb = struct.unpack_from("<{}s".format(l), val_buf, str_offset)[0]
            str_offset += l
            strs.append(sb)
            ii += 1

        val = np.array(strs, dtype=object)

        # Reshape the result to the appropriate shape.
        result = np.reshape(val, shape)

    return result


def mapped_shared_memory_regions():
    """Return all system shared memory regions that were mapped but not unmapped/destoryed.

    Returns
    -------
    list
        The list of mapped system shared memory regions.
    """

    return mapped_shm_regions


def destroy_shared_memory_region(shm_handle):
    """Unlink a system shared memory region with the specified handle.

    Parameters
    ----------
    shm_handle : c_void_p
        The handle for the system shared memory region.

    Raises
    ------
    SharedMemoryException
        If unable to unlink the shared memory region.
    """

    _raise_if_error(c_int(_cshm_shared_memory_region_destroy(shm_handle)))

    shm_fd = c_int()
    offset = c_uint64()
    byte_size = c_uint64()
    shm_addr = c_char_p()
    shm_key = c_char_p()
    _raise_if_error(
            c_int(_cshm_get_shared_memory_handle_info(shm_handle, byref(shm_addr), byref(shm_key), byref(shm_fd), \
                                    byref(offset), byref(byte_size))))
    mapped_shm_regions.remove(shm_key.value.decode("utf-8"))

    return


class SharedMemoryException(Exception):
    """Exception indicating non-Success status.

    Parameters
    ----------
    err : c_void_p
        Pointer to an Error that should be used to initialize the exception.

    """

    def __init__(self, err):
        self.err_code_map = {
            -2: "unable to get shared memory descriptor",
            -3: "unable to initialize the size",
            -4: "unable to read/mmap the shared memory region",
            -5: "unable to unlink the shared memory region",
            -6: "unable to munmap the shared memory region"
        }
        self._msg = None
        if type(err) == str:
            self._msg = err
        elif err.value != 0 and err.value in self.err_code_map:
            self._msg = self.err_code_map[err.value]

    def __str__(self):
        msg = super().__str__() if self._msg is None else self._msg
        return msg
