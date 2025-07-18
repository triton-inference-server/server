// Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

#include <dlpack/dlpack.h>

#ifdef TRITON_PB_STUB
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

#include <functional>
#include <string>

#include "pb_memory.h"
#include "pb_string.h"
#include "pb_utils.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_memory.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace python {

//
// Represents a Tensor object in shared memory.
//
struct TensorShm {
  // Handle for the pointer data in shared memory.
  bi::managed_external_buffer::handle_t memory;
  TRITONSERVER_DataType dtype;
  size_t dims_count;
};

// PbTensor class is the representation of Triton tensors inside Python backend.
class PbTensor {
 public:
#ifdef TRITON_PB_STUB
  /// Create a PbTensor using a numpy array
  /// \param name The name of the tensor
  /// \param numpy_array Numpy array to use for the initialization of the tensor
  PbTensor(const std::string& name, py::array& numpy_array);

  /// Create a PbTensor using a numpy array. This constructor is used for types
  /// that are not natively available in C++ such as float16. This constructor
  /// will fix the type of the NumPy array to match the Triton dtype.
  /// \param name The name of the tensor
  /// \param numpy_array Numpy array to use for the initialization of the tensor
  /// \param dtype The triton dtype
  PbTensor(
      const std::string& name, py::array& numpy_array,
      TRITONSERVER_DataType dtype);
#endif

  /// Create a PbTensor from raw pointer. This constructor is used for
  /// interfacing with DLPack tensors.
  /// \param name The name of the tensor
  /// \param dims Tensor dimensions
  /// \param dtype Triton dtype
  /// \param memory_type The memory type of the tensor
  /// \param memory_type_id The memory type_id of the tensor
  /// \param memory_ptr Pointer to the location of the data. Data must be
  /// contiguous and in C-order.
  /// \param byte_size Total number of bytes that the tensor uses.
  /// \param shm_handle The shared memory handle of pointer if it is stored in
  /// shared memory.
  PbTensor(
      const std::string& name, const std::vector<int64_t>& dims,
      TRITONSERVER_DataType dtype, TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_id, void* memory_ptr, uint64_t byte_size,
      DLManagedTensor* dl_managed_tensor = nullptr);

  /// This constructor is used when loading the tensor from shared memory.
  /// \param tensor_shm The name of the tensor
  /// \param dims_shm Tensor dimensions
  /// \param pb_string Triton dtype
  PbTensor(
      AllocatedSharedMemory<char>& tensor_shm,
      std::unique_ptr<PbString>& name_shm,
      std::unique_ptr<PbMemory>& pb_memory);

  // Copying tensor objects is not allowed.
  DISALLOW_COPY_AND_ASSIGN(PbTensor);

#ifdef TRITON_PB_STUB
  /// Construct a Python backend tensor from an
  /// external tensor.
  /// \param dlpack source dlpack tensor
  /// \param name name of the tensor
  static std::shared_ptr<PbTensor> FromDLPack(
      const std::string& name, const py::object& dlpack);

  /// Construct a Python backend tensor using a DLPack
  /// capsule.
  static std::shared_ptr<PbTensor> FromDLPackCapsule(
      const std::string& name, const py::capsule& dlpack);

  /// Construct a Python backend tensor using a NumPy object.
  /// \param numpy_array Numpy array
  /// \param name name of the tensor
  static std::shared_ptr<PbTensor> FromNumpy(
      const std::string& name, py::array& numpy_array);

  /// Get device type in DLPack format.
  DLDeviceType DeviceType();

  /// Exports tensor for consumption by `from_dlpack()` as a DLPack capsule.
  /// \param stream  a Python integer representing a pointer to a stream,
  ///                on devices that support streams
  /// \return Capsule object containing pointer to a DLPack object.
  py::capsule DLPack(const py::object& stream);

  /// Get a PyCapsule object containing the DLPack representation of the tensor.
  /// \return Capsule object containing pointer to a DLPack object.
  py::capsule ToDLPack();

  /// Returns device type and device ID.
  /// Meant for use within `from_dlpack()`.
  /// \return a pair (device_type, device_id).
  std::pair<int32_t, int64_t> DLPackDevice();
#endif

  /// Get the name of the tensor
  /// \return name of the tensor.
  const std::string& Name() const;

  /// Set the name of the tensor
  /// \param name Name of the tensor.
  void SetName(const std::string& name);

  /// Get the shared memory handle corresponding to this tensor
  /// \return returns the shared memory handle.
  bi::managed_external_buffer::handle_t ShmHandle();

  /// Load the tensor object from shared memory.
  /// \param shm_pool The shared memory manager object
  /// \param tensor_handle The handle of the object in shared memory.
  /// \param open_cuda_handle If the tensor is in GPU, setting this option to
  /// true will call cudaIpcOpenMemHandle on it. In the main process this option
  /// should be set to false because we never want to call cudaIpcOpenMemHandle
  /// in the main process.
  /// \return returns the tensor loaded from shared memory.
  static std::unique_ptr<PbTensor> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t tensor_handle,
      bool open_cuda_handle);

#ifdef TRITON_PB_STUB
  /// Get NumPy representation of the tensor.
  /// \throw If the tensor is stored in GPU, an exception is thrown
  /// \return NumPy representation of the Tensor
  const py::array* AsNumpy() const;
#endif

  /// Save tensor inside shared memory.
  void SaveToSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool, bool copy_gpu);

  /// Get the triton dtype
  /// \return Triton dtype
  TRITONSERVER_DataType TritonDtype() const;

  /// Get the data ptr
  /// \return Get the raw pointer.
  void* DataPtr();

  /// This function will be automatically called by the stub when the tensor is
  /// no longer required.
  void DeleteDLPack();

  /// Tells whether the Tensor is stored in CPU or not.
  /// \return A boolean value indicating whether the tensor is stored in CPU
  /// or not.
  bool IsCPU() const;

  /// Get the total byte size of the tensor.
  uint64_t ByteSize() const;

  /// Get the triton memory type of the Tensor.
  /// \return the memory type of the tensor.
  TRITONSERVER_MemoryType MemoryType() const;

  /// Get a mutable reference to the MemoryType.
  /// \return the pointer to the memory type of the tensor.
  TRITONSERVER_MemoryType* MutableMemoryType();

  /// Get the triton memory type of the Tensor.
  /// \return the memory type of the tensor.
  int64_t MemoryTypeId() const;

  /// Get the dimensions of the tensor
  /// \return A vector containing the tensor dimensions.
  const std::vector<int64_t>& Dims() const;

  /// Get the underlying memory
  std::unique_ptr<PbMemory>& Memory();

  /// Set the underlying memory
  void SetMemory(std::unique_ptr<PbMemory>&& memory);

  PbTensor();

  /// Destructor
  ~PbTensor() noexcept(false);

 private:
  std::string name_;
#ifdef TRITON_PB_STUB
  py::array numpy_array_;
  // Storing the serialized version of the numpy array
  py::array numpy_array_serialized_;
#endif
  TRITONSERVER_DataType dtype_;
  void* memory_ptr_;
  int64_t memory_type_id_;
  std::vector<int64_t> dims_;
  TRITONSERVER_MemoryType memory_type_;
  uint64_t byte_size_;
  DLManagedTensor* dl_managed_tensor_;

  bi::managed_external_buffer::handle_t shm_handle_;

  AllocatedSharedMemory<char> tensor_shm_;
  TensorShm* tensor_shm_ptr_;
  int64_t* dims_shm_ptr_;
  std::unique_ptr<PbString> name_shm_;

  // The pointer is null when the object is not stored in shared memory.
  std::unique_ptr<PbMemory> pb_memory_;
};
}}}  // namespace triton::backend::python
