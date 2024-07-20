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

from __future__ import annotations

import ctypes
import struct
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, ClassVar, Optional, Sequence, Type

import numpy
from tritonserver._c import InvalidArgumentError
from tritonserver._c import TRITONSERVER_DataType as DataType
from tritonserver._c import TRITONSERVER_MemoryType as MemoryType
from tritonserver._c import TRITONSERVER_ResponseAllocator, UnsupportedError

from . import _dlpack

try:
    import cupy
except ImportError:
    cupy = None

DeviceOrMemoryType = (
    tuple[MemoryType, int] | MemoryType | tuple[_dlpack.DLDeviceType, int] | str
)


class CustomKeyErrorDict(dict):
    def __init__(
        self,
        from_name: str,
        to_name: str,
        *args,
        exception: Type[Exception] = InvalidArgumentError,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._to_name = to_name
        self._from_name = from_name
        self._exception = exception

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            raise self._exception(
                f"Unsupported {self._from_name}. Can't convert {key} to {self._to_name}"
            ) from None


STRING_TO_TRITON_MEMORY_TYPE: dict[str, MemoryType] = CustomKeyErrorDict(
    "Memory Type String",
    "Triton server memory type",
    {"CPU": MemoryType.CPU, "CPU_PINNED": MemoryType.CPU_PINNED, "GPU": MemoryType.GPU},
)


DLPACK_DEVICE_TYPE_TO_TRITON_MEMORY_TYPE: dict[
    _dlpack.DLDeviceType, MemoryType
] = CustomKeyErrorDict(
    "DLPack device type",
    "Triton server memory type",
    {
        _dlpack.DLDeviceType.kDLCUDA: MemoryType.GPU,
        _dlpack.DLDeviceType.kDLCPU: MemoryType.CPU,
    },
)

TRITON_MEMORY_TYPE_TO_DLPACK_DEVICE_TYPE: dict[
    MemoryType, _dlpack.DLDeviceType
] = CustomKeyErrorDict(
    "Triton server memory type",
    "DLPack device type",
    {
        **{
            value: key
            for key, value in DLPACK_DEVICE_TYPE_TO_TRITON_MEMORY_TYPE.items()
        },
        **{MemoryType.CPU_PINNED: _dlpack.DLDeviceType.kDLCPU},
    },
)

DLPACK_TO_TRITON_DTYPE: dict[
    tuple[_dlpack.DLDataTypeCode, int], DataType
] = CustomKeyErrorDict(
    "DLPack data type",
    "Triton server data type",
    {
        (_dlpack.DLDataTypeCode.kDLBool, 8): DataType.BOOL,
        (_dlpack.DLDataTypeCode.kDLInt, 8): DataType.INT8,
        (
            _dlpack.DLDataTypeCode.kDLInt,
            16,
        ): DataType.INT16,
        (
            _dlpack.DLDataTypeCode.kDLInt,
            32,
        ): DataType.INT32,
        (
            _dlpack.DLDataTypeCode.kDLInt,
            64,
        ): DataType.INT64,
        (
            _dlpack.DLDataTypeCode.kDLUInt,
            8,
        ): DataType.UINT8,
        (
            _dlpack.DLDataTypeCode.kDLUInt,
            16,
        ): DataType.UINT16,
        (
            _dlpack.DLDataTypeCode.kDLUInt,
            32,
        ): DataType.UINT32,
        (
            _dlpack.DLDataTypeCode.kDLUInt,
            64,
        ): DataType.UINT64,
        (
            _dlpack.DLDataTypeCode.kDLFloat,
            16,
        ): DataType.FP16,
        (
            _dlpack.DLDataTypeCode.kDLFloat,
            32,
        ): DataType.FP32,
        (
            _dlpack.DLDataTypeCode.kDLFloat,
            64,
        ): DataType.FP64,
        (
            _dlpack.DLDataTypeCode.kDLBfloat,
            16,
        ): DataType.BF16,
    },
)

TRITON_TO_DLPACK_DTYPE: dict[DataType, _dlpack.DLDataType] = CustomKeyErrorDict(
    "Triton server data type",
    "DLPack data type",
    {
        value: _dlpack.DLDataType(type_code=key[0], bits=key[1], lanes=1)
        for key, value in DLPACK_TO_TRITON_DTYPE.items()
    },
)

NUMPY_TO_TRITON_DTYPE: dict[type, DataType] = CustomKeyErrorDict(
    "Numpy data type",
    "Triton server data type",
    {
        bool: DataType.BOOL,
        numpy.bool_: DataType.BOOL,
        numpy.int8: DataType.INT8,
        numpy.int16: DataType.INT16,
        numpy.int32: DataType.INT32,
        numpy.int64: DataType.INT64,
        numpy.uint8: DataType.UINT8,
        numpy.uint16: DataType.UINT16,
        numpy.uint32: DataType.UINT32,
        numpy.uint64: DataType.UINT64,
        numpy.float16: DataType.FP16,
        numpy.float32: DataType.FP32,
        numpy.float64: DataType.FP64,
        numpy.bytes_: DataType.BYTES,
        numpy.str_: DataType.BYTES,
        numpy.object_: DataType.BYTES,
    },
)

TRITON_TO_NUMPY_DTYPE: dict[DataType, type] = CustomKeyErrorDict(
    "Triton data type",
    "Numpy data type",
    {
        **{value: key for key, value in NUMPY_TO_TRITON_DTYPE.items()},
        **{DataType.BYTES: numpy.object_},
        **{DataType.BOOL: numpy.bool_},
    },
)


def parse_device_or_memory_type(
    device_or_memory_type: DeviceOrMemoryType,
) -> tuple[MemoryType, int]:
    if isinstance(device_or_memory_type, tuple):
        if isinstance(device_or_memory_type[0], MemoryType):
            memory_type = device_or_memory_type[0]
            memory_type_id = device_or_memory_type[1]
        elif isinstance(device_or_memory_type[0], _dlpack.DLDeviceType):
            memory_type = DLPACK_DEVICE_TYPE_TO_TRITON_MEMORY_TYPE[
                device_or_memory_type[0]
            ]
            memory_type_id = device_or_memory_type[1]
        else:
            raise InvalidArgumentError(f"Invalid memory type {device_or_memory_type}")
    elif isinstance(device_or_memory_type, MemoryType):
        memory_type = device_or_memory_type
        memory_type_id = 0
    elif isinstance(device_or_memory_type, str):
        memory_str_tuple = device_or_memory_type.split(":")
        if len(memory_str_tuple) > 2:
            raise InvalidArgumentError(
                f"Invalid memory type string {device_or_memory_type}"
            )
        memory_type = STRING_TO_TRITON_MEMORY_TYPE[memory_str_tuple[0].upper()]
        if len(memory_str_tuple) == 2:
            try:
                memory_type_id = int(memory_str_tuple[1])
            except ValueError:
                raise InvalidArgumentError(
                    f"Invalid memory type string {device_or_memory_type}"
                ) from None
        else:
            memory_type_id = 0
    return (memory_type, memory_type_id)


class DLPackObject:
    def __init__(self, value) -> None:
        try:
            stream = None
            device, device_id = value.__dlpack_device__()
            if device == _dlpack.DLDeviceType.kDLCUDA:
                if cupy is None:
                    raise UnsupportedError(
                        f"DLPack synchronization on device {device,device_id} not supported"
                    )
                with cupy.cuda.Device(device_id):
                    stream = 1  # legacy default stream
                    self._capsule = _dlpack.get_dlpack_capsule(value, stream)
                    self._tensor = _dlpack.get_managed_tensor(self._capsule).dl_tensor
            else:
                self._capsule = _dlpack.get_dlpack_capsule(value)
                self._tensor = _dlpack.get_managed_tensor(self._capsule).dl_tensor
        except Exception as e:
            raise InvalidArgumentError(
                f"Object does not support DLPack protocol: {e}"
            ) from None

    def __eq__(self, other) -> bool:
        if not isinstance(other, DLPackObject):
            return False
        if self.byte_size != other.byte_size:
            return False
        if self.memory_type != other.memory_type:
            return False
        if self.memory_type_id != other.memory_type_id:
            return False
        if self.shape != other.shape:
            return False
        if self.data_ptr != other.data_ptr:
            return False
        if self.contiguous != other.contiguous:
            return False
        if self.triton_data_type != other.triton_data_type:
            return False
        return True

    @property
    def byte_size(self) -> int:
        return _dlpack.get_byte_size(
            self._tensor.dtype, self._tensor.ndim, self._tensor.shape
        )

    @property
    def memory_type(self) -> MemoryType:
        return DLPACK_DEVICE_TYPE_TO_TRITON_MEMORY_TYPE[self._tensor.device.device_type]

    @property
    def memory_type_id(self) -> int:
        return self._tensor.device.device_id

    @property
    def shape(self) -> list[int]:
        return [self._tensor.shape[i] for i in range(self._tensor.ndim)]

    @property
    def triton_data_type(self) -> DataType:
        return DLPACK_TO_TRITON_DTYPE[self.data_type]

    @property
    def data_type(self) -> tuple[_dlpack.DLDataTypeCode, int]:
        return (self._tensor.dtype.type_code, self._tensor.dtype.bits)

    @property
    def data_ptr(self) -> ctypes.c_void_p:
        return self._tensor.data + self._tensor.byte_offset

    @property
    def contiguous(self) -> bool:
        return _dlpack.is_contiguous_data(
            self._tensor.ndim, self._tensor.shape, self._tensor.strides
        )
