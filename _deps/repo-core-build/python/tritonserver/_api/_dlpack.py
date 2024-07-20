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

################################################################################
# This file contains the DLPack API wrapped in Python style (see
# 'dlpack.h' for detail) and the utilities for Triton client to interact
# with DLPack
#
# Ref:
# https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h
# https://github.com/dmlc/dlpack/blob/main/apps/numpy_dlpack/dlpack/from_numpy.py
################################################################################

import ctypes

# Need to explicit set the res / arg types for pythonapi functions to
# work properly
ctypes.pythonapi.PyMem_RawMalloc.restype = ctypes.c_void_p
ctypes.pythonapi.PyMem_RawFree.argtypes = [ctypes.c_void_p]

ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
ctypes.pythonapi.PyCapsule_New.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_void_p,
]

ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

c_str_dltensor = b"dltensor"


class DLDeviceType(ctypes.c_int):
    kDLCPU = 1
    kDLCUDA = 2
    kDLCUDAHost = 3
    kDLOpenCL = 4
    kDLVulkan = 7
    kDLMetal = 8
    kDLVPI = 9
    kDLROCM = 10
    kDLROCMHost = 11
    kDLExtDev = 12
    kDLCUDAManaged = 13
    kDLOneAPI = 14
    kDLWebGPU = 15
    kDLHexagon = 16


class DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", ctypes.c_int),
        ("device_id", ctypes.c_int),
    ]


class DLDataTypeCode(ctypes.c_uint8):
    kDLInt = 0
    kDLUInt = 1
    kDLFloat = 2
    kDLOpaquePointer = 3
    kDLBfloat = 4
    kDLComplex = 5
    kDLBool = 6


class DLDataType(ctypes.Structure):
    _fields_ = [
        ("type_code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
    ]


# Utilities


def _raise_error(msg):
    """
    Raise error with the provided message
    """
    raise Exception(msg=msg) from None


# Use as managed context in DLPack that doesn't hold ownership of the
# data content.
class DataViewContext:
    def __init__(self, shape) -> None:
        # Convert the Python object to ctypes objects expected by
        # DLPack
        self._shape = (ctypes.c_int64 * len(shape))(*shape)
        # No strides: compact and row-major
        self._strides = ctypes.POINTER(ctypes.c_int64)()

    def as_manager_ctx(self) -> ctypes.c_void_p:
        py_obj = ctypes.py_object(self)
        py_obj_ptr = ctypes.pointer(py_obj)
        ctypes.pythonapi.Py_IncRef(py_obj)
        ctypes.pythonapi.Py_IncRef(ctypes.py_object(py_obj_ptr))
        return ctypes.cast(py_obj_ptr, ctypes.c_void_p)


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def managed_tensor_deleter(handle: ctypes.c_void_p) -> None:
    dl_managed_tensor = DLManagedTensor.from_address(handle)
    py_obj_ptr = ctypes.cast(
        dl_managed_tensor.manager_ctx, ctypes.POINTER(ctypes.py_object)
    )
    py_obj = py_obj_ptr.contents
    ctypes.pythonapi.Py_DecRef(py_obj)
    ctypes.pythonapi.Py_DecRef(ctypes.py_object(py_obj_ptr))
    ctypes.pythonapi.PyMem_RawFree(handle)


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def pycapsule_deleter(handle: ctypes.c_void_p) -> None:
    pycapsule: ctypes.py_object = ctypes.cast(handle, ctypes.py_object)
    if ctypes.pythonapi.PyCapsule_IsValid(pycapsule, c_str_dltensor):
        dl_managed_tensor = ctypes.pythonapi.PyCapsule_GetPointer(
            pycapsule, c_str_dltensor
        )
        managed_tensor_deleter(dl_managed_tensor)
        ctypes.pythonapi.PyCapsule_SetDestructor(pycapsule, None)


def triton_to_dlpack_dtype(dtype):
    if dtype == "BOOL":
        type_code = DLDataTypeCode.kDLBool
        bits = 8
    elif dtype == "INT8":
        type_code = DLDataTypeCode.kDLInt
        bits = 8
    elif dtype == "INT16":
        type_code = DLDataTypeCode.kDLInt
        bits = 16
    elif dtype == "INT32":
        type_code = DLDataTypeCode.kDLInt
        bits = 32
    elif dtype == "INT64":
        type_code = DLDataTypeCode.kDLInt
        bits = 64
    elif dtype == "UINT8":
        type_code = DLDataTypeCode.kDLUInt
        bits = 8
    elif dtype == "UINT16":
        type_code = DLDataTypeCode.kDLUInt
        bits = 16
    elif dtype == "UINT32":
        type_code = DLDataTypeCode.kDLUInt
        bits = 32
    elif dtype == "UINT64":
        type_code = DLDataTypeCode.kDLUInt
        bits = 64
    elif dtype == "FP16":
        type_code = DLDataTypeCode.kDLFloat
        bits = 16
    elif dtype == "FP32":
        type_code = DLDataTypeCode.kDLFloat
        bits = 32
    elif dtype == "FP64":
        type_code = DLDataTypeCode.kDLFloat
        bits = 64
    elif dtype == "BF16":
        type_code = DLDataTypeCode.kDLBfloat
        bits = 16
    elif dtype == "BYTES":
        _raise_error("DLPack currently doesn't support BYTES type")
    else:
        _raise_error(
            "Can not convert unknown data type '{}' to DLPack data type".format(dtype)
        )
    return DLDataType(type_code, bits, 1)


def is_contiguous_data(
    ndim: ctypes.c_int,
    shape: ctypes.POINTER(ctypes.c_int64),
    stride: ctypes.POINTER(ctypes.c_int64),
):
    # If 'stride' doesn't capture valid value
    if (stride is None) or (not bool(stride)):
        return True
    calculated_stride = 1
    # iterate stride in reverse order [ndim-1, -1)
    for i in reversed(range(ndim)):
        if stride[i] != calculated_stride:
            return False
        calculated_stride *= shape[i]
    return True


def get_byte_size(
    dtype: DLDataType, ndim: ctypes.c_int, shape: ctypes.POINTER(ctypes.c_int64)
):
    element_byte_size = dtype.bits * dtype.lanes // 8  # Assume 8 bits in a byte
    for i in range(ndim):
        element_byte_size *= shape[i]
    return element_byte_size


def get_dlpack_capsule(dlpack_obj, stream=None):
    # Extract PyCapsule of the DLPack object
    if hasattr(dlpack_obj, "__dlpack__"):
        if not hasattr(dlpack_obj, "__dlpack_device__"):
            _raise_error(
                "DLPack expects '__dlpack_device__' if '__dlpack__' has been defined"
            )
        device = dlpack_obj.__dlpack_device__()
        # Have to condition on the device type as, using numpy as example,
        # some DLPack implementation doesn't accept 'stream' as arguments
        if device != DLDeviceType.kDLCUDA:
            return dlpack_obj.__dlpack__()
        else:
            return dlpack_obj.__dlpack__(stream)
    else:
        # Old interface where PyCapsule object is passed directly
        return dlpack_obj


def get_dlpack_device(dlpack_obj):
    if hasattr(dlpack_obj, "__dlpack_device__"):
        return dlpack_obj.__dlpack_device__()
    return None


def get_managed_tensor(dlcapsule):
    ptr = ctypes.pythonapi.PyCapsule_GetPointer(dlcapsule, c_str_dltensor)
    return DLManagedTensor.from_address(ptr)
