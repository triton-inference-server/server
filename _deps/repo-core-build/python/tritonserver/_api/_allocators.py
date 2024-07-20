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

"""Default / Example Allocators for Tensor Memory"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy
from tritonserver._api._datautils import (
    DeviceOrMemoryType,
    DLPackObject,
    parse_device_or_memory_type,
)
from tritonserver._api._logging import LogLevel, LogMessage
from tritonserver._c import TRITONSERVER_ResponseAllocator
from tritonserver._c.triton_bindings import (
    InvalidArgumentError,
    TRITONSERVER_BufferAttributes,
)
from tritonserver._c.triton_bindings import TRITONSERVER_MemoryType as MemoryType

try:
    import cupy
except ImportError:
    cupy = None


default_memory_allocators: dict[MemoryType, MemoryAllocator] = dict({})


@dataclass
class MemoryBuffer:
    """Memory allocated for a Tensor.

    This object does not own the memory but holds a reference to the
    owner.

    Parameters
    ----------
    data_ptr : int
        Pointer to the allocated memory.
    memory_type : MemoryType
        memory type
    memory_type_id : int
        memory type id (typically the same as device id)
    size : int
        Size of the allocated memory in bytes.
    owner : Any
        Object that owns or manages the memory buffer.  Allocated
        memory must not be freed while a reference to the owner is
        held.

    Examples
    --------
    >>> buffer = MemoryBuffer.from_dlpack(numpy.array([100],dtype=numpy.uint8))

    """

    data_ptr: int
    memory_type: MemoryType
    memory_type_id: int
    size: int
    owner: Any

    @staticmethod
    def from_dlpack(owner: Any) -> MemoryBuffer:
        if not hasattr(owner, "__dlpack__"):
            raise InvalidArgumentError("Object does not support DLpack protocol")

        dlpack_object = DLPackObject(owner)

        return MemoryBuffer._from_dlpack_object(owner, dlpack_object)

    @staticmethod
    def _from_dlpack_object(owner: Any, dlpack_object: DLPackObject) -> MemoryBuffer:
        if not dlpack_object.contiguous:
            raise InvalidArgumentError("Only contiguous memory is supported")

        return MemoryBuffer(
            int(dlpack_object.data_ptr),
            dlpack_object.memory_type,
            dlpack_object.memory_type_id,
            dlpack_object.byte_size,
            owner,
        )

    def _create_tritonserver_buffer_attributes(self) -> TRITONSERVER_BufferAttributes:
        buffer_attributes = TRITONSERVER_BufferAttributes()
        buffer_attributes.memory_type = self.memory_type
        buffer_attributes.memory_type_id = self.memory_type_id
        buffer_attributes.byte_size = self.size
        # TODO: Support allocation / use of cuda shared memory
        #        buffer_attributes.cuda_ipc_handle = None
        return buffer_attributes


class MemoryAllocator(ABC):
    """Abstract interface to allow for custom memory allocation strategies

    Classes implementing the MemoryAllocator interface have to provide
    an allocate method returning MemoryBuffer objects.  A memory
    allocator implementation does not need to match the requested
    memory type or memory type id.



    Examples
    --------

    class TorchAllocator(tritonserver.MemoryAllocator):
        def allocate(self,
                     size,
                     memory_type,
                     memory_type_id):

            device = "cpu"

            if memory_type == tritonserver.MemoryType.GPU:
                device = "cuda"

            tensor = torch.zeros(size,dtype=torch.uint8,device=device)
            return tritonserver.MemoryBuffer.from_dlpack(tensor)

    """

    @abstractmethod
    def allocate(
        self, size: int, memory_type: MemoryType, memory_type_id: int
    ) -> MemoryBuffer:
        """Allocate memory buffer for tensor.

        Note: A memory allocator implementation does not need to honor
        the requested memory type or memory type id

        Parameters
        ----------
        size : int
            number of bytes requested
        memory_type : MemoryType
            type of memory requested (CPU, GPU, etc.)
        memory_type_id : int
            memory type id requested (typically device id)

        Returns
        -------
        MemoryBuffer
            memory buffer with requested size

        Examples
        --------
        memory_buffer = allocator.allocate(100, MemoryType.CPU, 0)

        """

        pass


class CPUAllocator(MemoryAllocator):
    def __init__(self):
        pass

    def allocate(
        self, size: int, memory_type: MemoryType, memory_type_id: int
    ) -> MemoryBuffer:
        ndarray = numpy.empty(size, numpy.byte)
        return MemoryBuffer.from_dlpack(ndarray)


default_memory_allocators[MemoryType.CPU] = CPUAllocator()

if cupy is not None:

    class GPUAllocator(MemoryAllocator):
        def __init__(self):
            pass

        def allocate(
            self,
            size: int,
            memory_type: MemoryType,
            memory_type_id: int,
        ) -> MemoryBuffer:
            with cupy.cuda.Device(memory_type_id):
                ndarray = cupy.empty(size, cupy.byte)

            return MemoryBuffer.from_dlpack(ndarray)

    default_memory_allocators[MemoryType.GPU] = GPUAllocator()


class ResponseAllocator:
    def __init__(
        self,
        memory_allocator: Optional[MemoryAllocator] = None,
        device_or_memory_type: Optional[DeviceOrMemoryType] = None,
    ):
        self._memory_allocator = memory_allocator
        self._memory_type: Optional[MemoryType] = None
        self._memory_type_id: int = 0
        self._response_allocator = None
        if device_or_memory_type is not None:
            self._memory_type, self._memory_type_id = parse_device_or_memory_type(
                device_or_memory_type
            )
        if (
            self._memory_type is not None
            and self._memory_allocator is None
            and self._memory_type not in default_memory_allocators
        ):
            raise InvalidArgumentError(
                f"Memory type {self._memory_type} not supported by default_memory_allocators: {default_memory_allocators}"
            )

    def allocate(
        self,
        _allocator,
        tensor_name,
        byte_size,
        memory_type,
        memory_type_id,
        _user_object,
    ):
        try:
            if self._memory_type is not None:
                memory_type = self._memory_type
                memory_type_id = self._memory_type_id

            memory_allocator = self._memory_allocator
            if memory_allocator is None:
                if memory_type in default_memory_allocators:
                    memory_allocator = default_memory_allocators[memory_type]
                else:
                    LogMessage(
                        LogLevel.WARN,
                        f"Requested memory type {memory_type} is not supported, falling back to {MemoryType.CPU}",
                    )
                    memory_type = MemoryType.CPU
                    memory_type_id = 0
                    memory_allocator = default_memory_allocators[memory_type]

            memory_buffer = memory_allocator.allocate(
                byte_size, memory_type, memory_type_id
            )

            return (
                memory_buffer.data_ptr,
                memory_buffer,
                memory_buffer.memory_type,
                memory_buffer.memory_type_id,
            )
        except Exception as e:
            message = f"Catastrophic failure in allocator: {e}, returning NULL"
            LogMessage(LogLevel.ERROR, message)
            return (0, None, MemoryType.CPU, 0)

    def release(
        self,
        _allocator,
        _buffer_,
        _buffer_user_object,
        _byte_size,
        _memory_type,
        _memory_type_id,
    ):
        pass

    def start(self, _allocator, _user_object):
        pass

    def query_preferred_memory_type(
        self,
        _allocator,
        _user_object,
        _tensor_name,
        _byte_size,
        memory_type: MemoryType,
        memory_type_id,
    ):
        if self._memory_type is not None:
            memory_type = self._memory_type
            memory_type_id = self._memory_type_id

        return (memory_type, memory_type_id)

    def set_buffer_attributes(
        self,
        _allocator,
        _tensor_name,
        buffer_attributes,
        _user_object,
        _buffer_user_object,
    ):
        if self._memory_type is not None:
            buffer_attributes.memory_type = self._memory_type
            buffer_attributes.memory_type_id = self._memory_type_id
        return buffer_attributes

    def create_tritonserver_response_allocator(self):
        self._response_allocator = TRITONSERVER_ResponseAllocator(
            self.allocate, self.release, self.start
        )
        self._response_allocator.set_query_function(self.query_preferred_memory_type)

        self._response_allocator.set_buffer_attributes_function(
            self.set_buffer_attributes
        )
        return self._response_allocator
