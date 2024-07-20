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
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Sequence

import numpy
from tritonserver._api._allocators import MemoryBuffer
from tritonserver._api._datautils import (
    NUMPY_TO_TRITON_DTYPE,
    TRITON_MEMORY_TYPE_TO_DLPACK_DEVICE_TYPE,
    TRITON_TO_DLPACK_DTYPE,
    DLPackObject,
    parse_device_or_memory_type,
)
from tritonserver._api._dlpack import (
    DLDevice,
    DLDeviceType,
    DLManagedTensor,
    c_str_dltensor,
)
from tritonserver._c.triton_bindings import (
    InvalidArgumentError,
    TRITONSERVER_BufferAttributes,
)
from tritonserver._c.triton_bindings import TRITONSERVER_DataType as DataType
from tritonserver._c.triton_bindings import TRITONSERVER_MemoryType as MemoryType
from tritonserver._c.triton_bindings import UnsupportedError

DeviceOrMemoryType = (
    tuple[MemoryType, int] | MemoryType | tuple[DLDeviceType, int] | str
)

try:
    import cupy
except ImportError:
    cupy = None


@dataclass
class Tensor:
    """Class representing a Tensor.

    Parameters
    ----------
    data_type : DataType
        Data type of the tensor.
    shape : Sequence[int]
        Shape of the tensor.
    memory_buffer : MemoryBuffer
        Memory buffer containing the tensor data.
    """

    data_type: DataType
    shape: Sequence[int]
    memory_buffer: MemoryBuffer

    @property
    def data_ptr(self) -> int:
        """Get the pointer to the tensor's data.

        Returns
        -------
        int
            The pointer to the tensor's data.
        """

        return self.memory_buffer.data_ptr

    @property
    def memory_type(self) -> MemoryType:
        """Get the memory type of the tensor.

        Returns
        -------
        MemoryType
            The memory type of the tensor.
        """

        return self.memory_buffer.memory_type

    @property
    def memory_type_id(self) -> int:
        """Get the ID representing the memory type of the tensor.

        Returns
        -------
        int
            The ID representing the memory type of the tensor.
        """

        return self.memory_buffer.memory_type_id

    @property
    def size(self) -> int:
        """Get the size of the tensor's data in bytes.

        Returns
        -------
        int
            The size of the tensor's data in bytes.
        """

        return self.memory_buffer.size

    def _sync_on_requested_stream(self, requested_stream_ptr):
        """Stream synchronization based on cupy implementation.

        Record event on requested stream if different from current
        stream.

        See
        `https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html`
        and
        `https://github.com/cupy/cupy/blob/f9563bcd5c674623f80ce975b590f9c860b44ed6/cupy/_core/core.pyx#L281`.
        for more details.

        Parameters
        ----------
        requested_stream_ptr :
            requested stream as defined by DLPack protocol

        Raises
        ------
        unsupported
            If synchronization can not be done

        """

        current_stream_ptr = None
        current_stream = None

        unsupported = UnsupportedError(
            f"DLPack stream synchronization on memory type {self.memory_type} and stream {requested_stream_ptr} not supported"
        )

        if requested_stream_ptr is not None and not isinstance(
            requested_stream_ptr, int
        ):
            raise unsupported

        if self.memory_type != MemoryType.GPU:
            if requested_stream_ptr not in (None, 0):
                raise unsupported
            return

        if cupy is None:
            raise unsupported

        # NOTE: Technically this is not required by the protocol. It is the
        #       responsibility of the caller(consumer) to ensure that
        #       we are on the correct device. Added to ensure
        #       the semantics are correct - but should be a no-op.
        #       May be removed in the future.
        with cupy.cuda.Device(self.memory_type_id):
            current_stream = cupy.cuda.get_current_stream()
            curr_stream_ptr = current_stream.ptr

            # Based on cupy documentation
            # cupy.cuda.Stream.null.ptr is legacy default stream
            # cupy.cuda.Stream.ptds.ptr is per thread default stream

            if curr_stream_ptr == 0:
                curr_stream_ptr = cupy.cuda.Stream.null.ptr

            if requested_stream_ptr in (None, 0, 1):
                requested_stream_ptr = cupy.cuda.Stream.null.ptr

            if requested_stream_ptr in (2,):
                requested_stream_ptr = cupy.cuda.Stream.ptds.ptr

            if requested_stream_ptr >= 0 and curr_stream_ptr != requested_stream_ptr:
                next_stream = cupy.cuda.ExternalStream(requested_stream_ptr)
                event = current_stream.record()
                next_stream.wait_event(event)

    def __dlpack__(self, *, stream=None):
        """Convert the tensor to a DLPack-compatible object.

        Parameters
        ----------
        stream : Any, optional
            Currently Ignored parameter, by default None.

        Returns
        -------
        Any
            A DLPack-compatible object representing the tensor.
        """

        self._sync_on_requested_stream(stream)

        dl_managed_tensor = Tensor._create_managed_tensor()
        dl_managed_tensor.dl_tensor.data = self.data_ptr
        dl_managed_tensor.dl_tensor.device = DLDevice(
            TRITON_MEMORY_TYPE_TO_DLPACK_DEVICE_TYPE[self.memory_type],
            self.memory_type_id,
        )

        dl_managed_tensor.dl_tensor.dtype = TRITON_TO_DLPACK_DTYPE[self.data_type]
        dl_managed_tensor.dl_tensor.ndim = len(self.shape)
        dl_managed_tensor.dl_tensor.shape = (ctypes.c_int64 * len(self.shape))(
            *self.shape
        )
        dl_managed_tensor.dl_tensor.strides = ctypes.POINTER(ctypes.c_int64)()
        dl_managed_tensor.dl_tensor.byte_offset = 0
        dl_managed_tensor.deleter = Tensor._managed_tensor_deleter

        self._set_dlpack_manager_ctx(dl_managed_tensor)
        pycapsule = ctypes.pythonapi.PyCapsule_New(
            ctypes.byref(dl_managed_tensor),
            c_str_dltensor,
            Tensor._pycapsule_deleter,
        )
        return pycapsule

    def __dlpack_device__(self) -> tuple[DLDeviceType, int]:
        """Get the DLPack device information for the tensor.

        Returns
        -------
        tuple[DLDeviceType, int]
            A tuple representing the DLPack device information (device type, device ID).
        """

        return (
            TRITON_MEMORY_TYPE_TO_DLPACK_DEVICE_TYPE[self.memory_type],
            self.memory_type_id,
        )

    def to_string_array(self) -> numpy.ndarray:
        """Deserialize Triton BYTES Tensor into numpy array of strings.

        If memory is not on the host the tensor data will be copied to
        the host before deserialization. For more details on the
        format of Triton BYTES Tensors please see Triton Inference
        Server documentation.

        Returns
        -------
        numpy.ndarray
            A numpy array of objects representing the BYTES tensor.

        Examples
        --------

        numpy_ndarray = response.outputs["text_output"].to_string_array()

        """
        return self.to_bytes_array().astype(str)

    def to_bytes_array(self) -> numpy.ndarray:
        """Deserialize Triton BYTES Tensor into numpy array.

        If memory is not on the host the tensor data will be copied to
        the host before deserialization. For more details on the
        format of Triton BYTES Tensors please see Triton Inference
        Server documentation.

        Returns
        -------
        numpy.ndarray
            A numpy array of objects representing the BYTES tensor.

        Examples
        --------

        numpy_ndarray = response.outputs["text_output"].to_bytes_array()

        """
        if self.data_type != DataType.BYTES:
            raise InvalidArgumentError(
                f"Tensor has data type {self.data_type} not {DataType.BYTES}"
            )

        # Reshape into 1d array of bytes on host
        original_data_type = self.data_type
        original_shape = self.shape
        self.data_type = DataType.UINT8
        self.shape = [self.size]
        numpy_ndarray = self._to_numpy_on_host()

        # Deserialize bytes array and reshape
        self.shape = original_shape
        self.data_type = original_data_type
        return Tensor._deserialize_bytes_array(numpy_ndarray).reshape(self.shape)

    @staticmethod
    def from_string_array(string_array: list[str] | numpy.ndarray) -> Tensor:
        """Create Triton BYTES Tensor from numpy array of strings or list of strings.

        Creates a Triton tensor of type BYTES from a list of strings,
        or numpy array of type str_. The
        method allocates new host memory to store the serialized
        tensor. For more details on the format of Triton BYTES Tensors
        please see Triton Inference Server documentation.

        Parameters
        ----------
        string_array : list[str] | numpy.ndarray
            an array like object to convert

        Returns
        -------
        Tensor

        Raises
        ------
        InvalidArgumentError
            If the given object can not be converted.

        Examples
        --------

        tensor = Tensor.from_string_array(numpy.array(["hello"]))

        tensor = Tensor.from_string_array(["hello"])

        """
        return Tensor.from_bytes_array(string_array)

    @staticmethod
    def from_bytes_array(
        bytes_array: list[str] | list[bytes] | numpy.ndarray,
    ) -> Tensor:
        """Create Triton BYTES Tensor from numpy array or list

        Creates a Triton tensor of type BYTES from a list of strings,
        bytes or a numpy array of type object_, bytes_, or str_. The
        method allocates new host memory to store the serialized
        tensor. For more details on the format of Triton BYTES Tensors
        please see Triton Inference Server documentation.

        Parameters
        ----------
        bytes_array : list[str | bytes] | numpy.ndarray
            an array like object to convert

        Returns
        -------
        Tensor

        Raises
        ------
        InvalidArgumentError
            If the given object can not be converted.

        Examples
        --------

        tensor = Tensor.from_bytes_array(numpy.array(["hello"]))

        tensor = Tensor.from_bytes_array(["hello"])

        """
        result = Tensor._from_object(bytes_array)
        if result.data_type != DataType.BYTES:
            raise InvalidArgumentError(
                f"Unsupported conversion from {bytes_array} to Triton BYTES Tensor. Got {result.data_type}"
            )
        return result

    @staticmethod
    def _from_object(obj: list[Any] | numpy.ndarray | Any) -> Tensor:
        """Create a tensor from an object.

        Creates a tensor from an object using specific conversion
        methods if available or falls back to using __from_dlpack__.

        Specific conversions are currently supported for:

        list[obj: Any] : implicitly converted to numpy.array()
        numpy.ndarray : serialized if required to BYTES tensor

        Parameters
        ----------
        obj : list[Any] | numpy.ndarray | Any
            The input object to create the tensor from.

        Returns
        -------
        Tensor
            A new tensor created from the specified object.

        Examples
        --------

        tensor = Tensor.from_object(numpy.array(["hello"]))

        tensor = Tensor.from_object(["hello"])


        """
        if type(obj) in Tensor._from_converters:
            return Tensor._from_converters[type(obj)](obj)
        elif hasattr(obj, "__dlpack__"):
            return Tensor.from_dlpack(obj)
        else:
            raise InvalidArgumentError(
                f"Input type {type(obj)} not supported. Must be one of {list(Tensor._from_converters.keys())} or the type must support __dlpack__"
            )

    @staticmethod
    def from_dlpack(obj: Any) -> Tensor:
        """Create a tensor from a DLPack-compatible object.

        Parameters
        ----------
        obj : Any
            The DLPack-compatible object.

        Returns
        -------
        Tensor
            A new tensor created from the DLPack-compatible object.

        Examples
        --------

        tensor = Tensor.from_dlpack(numpy.array([0,1,2], dtype=numpy.float16))

        tensor = Tensor.from_dlpack(torch.zeros(100, dtype=torch.float16))

        """
        dlpack_object = DLPackObject(obj)
        data_type = dlpack_object.triton_data_type
        shape = dlpack_object.shape
        memory_buffer = MemoryBuffer._from_dlpack_object(
            obj, dlpack_object=dlpack_object
        )
        return Tensor(data_type, shape, memory_buffer)

    def to_host(self) -> Tensor:
        """Move the tensor to CPU memory from device memory

        Returns
        -------
        Tensor
            The tensor moved to the CPU.

        Examples
        --------

        tensor = Tensor.from_dlpack(torch.zeros(100, dtype=torch.float16).to("cuda"))

        numpy_nd_array = numpy.array(tensor.to_host())

        """
        return self.to_device("cpu")

    def to_device(self, device: DeviceOrMemoryType) -> Tensor:
        """Move the tensor to the specified device.

        Parameters
        ----------
        device : DeviceOrMemoryType
            The target device. Device can be specified as a string,
            MemoryType, tuple [MemoryType, memory_type__id], or
            tuple[DLDeviceType, device_id].

        Returns
        -------
        Tensor
            The tensor moved to the specified device.

        Examples
        --------

        tensor_cpu = tritonserver.Tensor.from_dlpack(numpy.array([0,1,2], dtype=numpy.float16))

        # Different ways to specify the device

        tensor_gpu = tensor_cpu.to_device(MemoryType.GPU)

        tensor_gpu = tensor_cpu.to_device((MemoryType.GPU,0))

        tensor_gpu = tensor_cpu.to_device((DLDeviceType.kDLCUDA,0))

        tensor_gpu = tensor_cpu.to_device("gpu")

        tensor_gpu = tensor_cpu.to_device("gpu:0")

        ndarray_gpu = cupy.from_dlpack(tensor_gpu)

        ndarray_gpu[0] = ndarray_gpu.mean()

        tensor_cpu = tensor_gpu.to_device("cpu")

        ndarray_cpu = numpy.from_dlpack(tensor_cpu)

        assert ndarray_cpu[0] == ndarray_gpu[0]

        """
        memory_type, memory_type_id = parse_device_or_memory_type(device)
        if self.memory_type == memory_type and self.memory_type_id == memory_type_id:
            return self
        if self.memory_type == MemoryType.CPU_PINNED and memory_type == MemoryType.CPU:
            return self
        if cupy is not None:
            if self.memory_type in (MemoryType.CPU, MemoryType.CPU_PINNED):
                ndarray = numpy.from_dlpack(self)
            else:
                ndarray = cupy.from_dlpack(self)

            if memory_type == MemoryType.CPU:
                return Tensor.from_dlpack(cupy.asnumpy(ndarray))
            if memory_type == MemoryType.GPU:
                with cupy.cuda.Device(memory_type_id):
                    return Tensor.from_dlpack(cupy.asarray(ndarray))

        raise UnsupportedError(
            f"Conversion from {(self.memory_type,self.memory_type_id)} to {(memory_type, memory_type_id)} not supported."
        )

    def _to_numpy_on_host(self) -> numpy.ndarray:
        if self.memory_type in (MemoryType.CPU, MemoryType.CPU_PINNED):
            return numpy.from_dlpack(self)

        if cupy is not None:
            return cupy.asnumpy(cupy.from_dlpack(self))

        raise UnsupportedError(
            f"Conversion from {self.memory_type} to numpy array not supported."
        )

    @staticmethod
    def _deserialize_bytes_array(numpy_ndarray: numpy.ndarray) -> numpy.ndarray:
        result = []
        _buffer = memoryview(numpy_ndarray)
        offset = 0
        while offset < len(_buffer):
            (item_length,) = struct.unpack_from("@I", _buffer, offset)
            offset += 4
            result.append(bytes(_buffer[offset : offset + item_length]))
            offset += item_length
        return numpy.array(result, dtype=numpy.object_)

    @staticmethod
    def _serialize_numpy_bytes_array(array: numpy.ndarray) -> numpy.ndarray:
        result = []
        for array_item in numpy.nditer(array, flags=["refs_ok"], order="C"):
            item = array_item.item()
            if not isinstance(item, bytes):
                item = str(item).encode("utf-8")
            result.append(struct.pack("@I", len(item)))
            result.append(item)
        return numpy.frombuffer(b"".join(result), dtype=numpy.byte)

    @staticmethod
    def _from_list(obj: list[Any]) -> Tensor:
        try:
            return Tensor._from_numpy(numpy.array(obj))
        except Exception as e:
            raise InvalidArgumentError(
                f"Conversion from {obj} to tensor not supported."
            ) from e

    @staticmethod
    def _from_numpy(obj: numpy.ndarray | numpy.generic) -> Tensor:
        data_type = NUMPY_TO_TRITON_DTYPE[obj.dtype.type]
        shape = obj.shape

        if isinstance(obj, numpy.generic):
            obj = numpy.asarray(obj)

        if data_type == DataType.BYTES:
            obj = Tensor._serialize_numpy_bytes_array(obj)

        memory_buffer = MemoryBuffer(
            data_ptr=obj.ctypes.data,
            memory_type=MemoryType.CPU,
            memory_type_id=0,
            size=obj.itemsize * obj.size,
            owner=obj,
        )

        return Tensor(data_type, shape, memory_buffer)

    @staticmethod
    def _create_managed_tensor():
        size = ctypes.c_size_t(ctypes.sizeof(DLManagedTensor))
        address = ctypes.pythonapi.PyMem_RawMalloc(size)
        return DLManagedTensor.from_address(address)

    @staticmethod
    @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    def _managed_tensor_deleter(handle: int) -> None:
        dl_managed_tensor = DLManagedTensor.from_address(handle)
        tensor_obj_ptr = ctypes.cast(
            dl_managed_tensor.manager_ctx, ctypes.POINTER(ctypes.py_object)
        )
        tensor_obj = tensor_obj_ptr.contents
        ctypes.pythonapi.Py_DecRef(tensor_obj)
        shape_obj = ctypes.py_object(dl_managed_tensor.dl_tensor.shape)
        ctypes.pythonapi.Py_DecRef(shape_obj)
        ctypes.pythonapi.PyMem_RawFree(handle)

    @staticmethod
    @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    def _pycapsule_deleter(handle: ctypes.c_void_p) -> None:
        try:
            pycapsule: ctypes.py_object = ctypes.cast(handle, ctypes.py_object)
            if ctypes.pythonapi.PyCapsule_IsValid(pycapsule, c_str_dltensor):
                dl_managed_tensor = ctypes.pythonapi.PyCapsule_GetPointer(
                    pycapsule, c_str_dltensor
                )

                Tensor._managed_tensor_deleter(dl_managed_tensor)

                ctypes.pythonapi.PyCapsule_SetDestructor(pycapsule, None)
        except Exception as e:
            print(f"Exception occurred while deleting capsule: {e}")
            raise e

    def _set_dlpack_manager_ctx(self, dl_managed_tensor):
        tensor_obj = ctypes.py_object(self)
        tensor_obj_ptr = ctypes.pointer(tensor_obj)
        dl_managed_tensor.manager_ctx = ctypes.cast(tensor_obj_ptr, ctypes.c_void_p)
        shape_obj = ctypes.py_object(dl_managed_tensor.dl_tensor.shape)
        ctypes.pythonapi.Py_IncRef(tensor_obj)
        ctypes.pythonapi.Py_IncRef(shape_obj)

    _from_converters: ClassVar[dict[type, Callable[[Any], Tensor]]] = dict(
        {numpy.ndarray: _from_numpy, numpy.generic: _from_numpy, list: _from_list},
    )
