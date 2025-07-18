// Copyright 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifdef TRITON_ENABLE_GPU
#include <cuda.h>
#endif  // TRITON_ENABLE_GPU

#ifdef TRITON_PB_STUB
#include "pb_stub.h"
#include "pb_stub_utils.h"
namespace py = pybind11;
#endif
#include "pb_tensor.h"

// WAR for undefined ssize_t on Windows: https://stackoverflow.com/a/35368387
#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

namespace triton { namespace backend { namespace python {

#ifdef TRITON_PB_STUB
PbTensor::PbTensor(const std::string& name, py::array& numpy_array)
    : name_(name)
{
  if (name == "") {
    throw PythonBackendException("Tensor name cannot be an empty string.");
  }

  dtype_ = numpy_to_triton_type(numpy_array.attr("dtype"));
  memory_type_ = TRITONSERVER_MEMORY_CPU;
  memory_type_id_ = 0;
  dl_managed_tensor_ = nullptr;

  bool is_contiguous =
      numpy_array.attr("data").attr("c_contiguous").cast<bool>();
  if (!is_contiguous) {
    py::module numpy = py::module::import("numpy");
    numpy_array = numpy.attr("ascontiguousarray")(numpy_array);
  }
  numpy_array_ = numpy_array;

  if (dtype_ == TRITONSERVER_TYPE_BYTES) {
    py::module triton_pb_utils =
        py::module::import("triton_python_backend_utils");
    numpy_array_serialized_ =
        triton_pb_utils.attr("serialize_byte_tensor")(numpy_array);
    memory_ptr_ = numpy_array_serialized_.request().ptr;
    byte_size_ = numpy_array_serialized_.nbytes();
  } else {
    memory_ptr_ = numpy_array_.request().ptr;
    byte_size_ = numpy_array_.nbytes();
  }

  // Initialize tensor dimension
  size_t dims_count = numpy_array_.ndim();

  const ssize_t* numpy_shape = numpy_array_.shape();
  for (size_t i = 0; i < dims_count; i++) {
    dims_.push_back(numpy_shape[i]);
  }
}

PbTensor::PbTensor(
    const std::string& name, py::array& numpy_array,
    TRITONSERVER_DataType dtype)
    : name_(name)
{
  if (name == "") {
    throw PythonBackendException("Tensor name cannot be an empty string.");
  }

  if (numpy_to_triton_type(numpy_array.attr("dtype")) != dtype) {
    numpy_array = numpy_array.attr("view")(triton_to_numpy_type(dtype));
  }
  bool is_contiguous =
      numpy_array.attr("data").attr("c_contiguous").cast<bool>();
  if (!is_contiguous) {
    py::module numpy = py::module::import("numpy");
    numpy_array = numpy.attr("ascontiguousarray")(numpy_array);
  }
  numpy_array_ = numpy_array;

  if (dtype == TRITONSERVER_TYPE_BYTES) {
    py::module triton_pb_utils =
        py::module::import("triton_python_backend_utils");
    numpy_array_serialized_ =
        triton_pb_utils.attr("serialize_byte_tensor")(numpy_array);
    memory_ptr_ = numpy_array_serialized_.request().ptr;
    byte_size_ = numpy_array_serialized_.nbytes();

  } else {
    memory_ptr_ = numpy_array_.request().ptr;
    byte_size_ = numpy_array_.nbytes();
  }
  memory_type_ = TRITONSERVER_MEMORY_CPU;
  dtype_ = dtype;

  // Initialize tensor dimension
  size_t dims_count = numpy_array_.ndim();

  const ssize_t* numpy_shape = numpy_array_.shape();
  for (size_t i = 0; i < dims_count; i++) {
    dims_.push_back(numpy_shape[i]);
  }
  memory_type_id_ = 0;
  dl_managed_tensor_ = nullptr;
}
#endif  // TRITON_PB_STUB

PbTensor::PbTensor(
    const std::string& name, const std::vector<int64_t>& dims,
    TRITONSERVER_DataType dtype, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, void* memory_ptr, uint64_t byte_size,
    DLManagedTensor* dl_managed_tensor)
{
  if (name == "") {
    throw PythonBackendException("Tensor name cannot be an empty string.");
  }

  name_ = name;
  memory_ptr_ = memory_ptr;
  memory_type_ = memory_type;
  memory_type_id_ = memory_type_id;
  dtype_ = dtype;
  dims_ = dims;

#ifdef TRITON_PB_STUB
  if (memory_type_ == TRITONSERVER_MEMORY_CPU ||
      memory_type_ == TRITONSERVER_MEMORY_CPU_PINNED) {
    if (dtype == TRITONSERVER_TYPE_BF16) {
      // No native numpy representation for BF16. DLPack should be used instead.
      numpy_array_ = py::none();
    } else if (dtype != TRITONSERVER_TYPE_BYTES) {
      py::object numpy_array =
          py::array(triton_to_pybind_dtype(dtype_), dims_, (void*)memory_ptr_);
      numpy_array_ = numpy_array.attr("view")(triton_to_numpy_type(dtype_));
    } else {
      py::object numpy_array = py::array(
          triton_to_pybind_dtype(TRITONSERVER_TYPE_UINT8), {byte_size},
          (void*)memory_ptr_);
      py::module triton_pb_utils =
          py::module::import("triton_python_backend_utils");
      numpy_array_ =
          triton_pb_utils.attr("deserialize_bytes_tensor")(numpy_array)
              .attr("reshape")(dims);
    }
  } else {
    numpy_array_ = py::none();
  }
#endif

  byte_size_ = byte_size;
  dl_managed_tensor_ = dl_managed_tensor;
}

bool
PbTensor::IsCPU() const
{
  if (memory_type_ == TRITONSERVER_MEMORY_CPU ||
      memory_type_ == TRITONSERVER_MEMORY_CPU_PINNED) {
    return true;
  } else {
    return false;
  }
}

TRITONSERVER_MemoryType
PbTensor::MemoryType() const
{
  return memory_type_;
}

int64_t
PbTensor::MemoryTypeId() const
{
  return memory_type_id_;
}

uint64_t
PbTensor::ByteSize() const
{
  return byte_size_;
}

const std::vector<int64_t>&
PbTensor::Dims() const
{
  return dims_;
}

void
PbTensor::SetMemory(std::unique_ptr<PbMemory>&& memory)
{
  pb_memory_ = std::move(memory);
  memory_type_ = pb_memory_->MemoryType();
  memory_type_id_ = pb_memory_->MemoryTypeId();
  byte_size_ = pb_memory_->ByteSize();
  memory_ptr_ = pb_memory_->DataPtr();
}

#ifdef TRITON_PB_STUB
void
delete_unused_dltensor(PyObject* dlp)
{
  if (PyCapsule_IsValid(dlp, "dltensor")) {
    DLManagedTensor* dl_managed_tensor =
        static_cast<DLManagedTensor*>(PyCapsule_GetPointer(dlp, "dltensor"));
    dl_managed_tensor->deleter(dl_managed_tensor);
  }
}

std::shared_ptr<PbTensor>
PbTensor::FromNumpy(const std::string& name, py::array& numpy_array)
{
  return std::make_shared<PbTensor>(name, numpy_array);
}

DLDeviceType
PbTensor::DeviceType()
{
  DLDeviceType device_type{};

  switch (memory_type_) {
    case TRITONSERVER_MEMORY_GPU:
      device_type = DLDeviceType::kDLCUDA;
      break;
    case TRITONSERVER_MEMORY_CPU:
      device_type = DLDeviceType::kDLCPU;
      break;
    case TRITONSERVER_MEMORY_CPU_PINNED:
      device_type = DLDeviceType::kDLCUDAHost;
      break;
  }

  return device_type;
}

py::capsule
PbTensor::DLPack(const py::object& stream)
{
  // Here external tensor requests PbTensor's `__dlpack__` method to provide
  // a PyCapsule. By the design of PbTensor, in a GPU case no pending work
  // is scheduled to work with PbTensor's data and we can simply pass
  // the capsule without a synchronization.
  return this->ToDLPack();
}

py::capsule
PbTensor::ToDLPack()
{
  if (dtype_ == TRITONSERVER_TYPE_BYTES) {
    throw PythonBackendException(
        "DLPack does not have support for string tensors.");
  }

  DLManagedTensor* dlpack_tensor = new DLManagedTensor;
  dlpack_tensor->dl_tensor.ndim = dims_.size();
  dlpack_tensor->dl_tensor.byte_offset = 0;
  dlpack_tensor->dl_tensor.data = memory_ptr_;
  dlpack_tensor->dl_tensor.shape = &dims_[0];
  dlpack_tensor->dl_tensor.strides = nullptr;
  dlpack_tensor->manager_ctx = this;
  dlpack_tensor->deleter = [](DLManagedTensor* m) {
    // We need to acquire GIL since the framework that deleted the dlpack tensor
    // may not have acquired GIL when calling this function.
    py::gil_scoped_acquire gil;
    if (m->manager_ctx == nullptr) {
      return;
    }

    PbTensor* tensor = reinterpret_cast<PbTensor*>(m->manager_ctx);
    py::handle tensor_handle = py::cast(tensor);
    tensor_handle.dec_ref();
    free(m);
  };

  PbTensor* tensor = reinterpret_cast<PbTensor*>(this);
  py::handle tensor_handle = py::cast(tensor);

  // Increase the reference count by one to make sure that the DLPack
  // representation doesn't become invalid when the tensor object goes out of
  // scope.
  tensor_handle.inc_ref();

  dlpack_tensor->dl_tensor.device.device_id = memory_type_id_;
  dlpack_tensor->dl_tensor.device.device_type = this->DeviceType();
  dlpack_tensor->dl_tensor.dtype = triton_to_dlpack_type(dtype_);

  return py::capsule(
      static_cast<void*>(dlpack_tensor), "dltensor", &delete_unused_dltensor);
}

std::pair<int32_t, int64_t>
PbTensor::DLPackDevice()
{
  return std::pair<int32_t, int64_t>(this->DeviceType(), memory_type_id_);
}

#endif  // TRITON_PB_STUB

void
PbTensor::DeleteDLPack()
{
  if (dl_managed_tensor_ != nullptr) {
    dl_managed_tensor_->deleter(dl_managed_tensor_);
    dl_managed_tensor_ = nullptr;
  }
}

std::unique_ptr<PbMemory>&
PbTensor::Memory()
{
  return pb_memory_;
}

#ifdef TRITON_PB_STUB
std::shared_ptr<PbTensor>
PbTensor::FromDLPack(const std::string& name, const py::object& tensor)
{
  if (name == "") {
    throw PythonBackendException("Tensor name cannot be an empty string.");
  }
  if (py::isinstance<py::capsule>(tensor)) {
    return FromDLPackCapsule(name, tensor);
  }

  if (!py::hasattr(tensor, "__dlpack__") ||
      !py::hasattr(tensor, "__dlpack_device__")) {
    throw PythonBackendException(
        "Provided tensor is not supported. Tensor must be a DLPack capsule \
        or have `__dlpack__` and `__dlpack_device__` attributes");
  }

  auto capsule_device_info =
      tensor.attr("__dlpack_device__")().cast<std::pair<int32_t, int64_t>>();
  if (capsule_device_info.first == DLDeviceType::kDLCUDA) {
#ifdef TRITON_ENABLE_GPU
    int current_device;
    cudaError_t err = cudaGetDevice(&current_device);
    std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
    if (err != cudaSuccess) {
      throw PythonBackendException("Failed to get current CUDA device id.");
    }
    ScopedSetDevice scoped_set_device(capsule_device_info.second);

    bool overridden = (current_device != capsule_device_info.second);
    cudaStream_t proxy_stream = stub->GetProxyStream(current_device);

    // Array API requirements for the stream argument:
    // stream = 1 the legacy default stream (in this case should
    // synchronize on CUDA stream 0)
    // For CPU, `stream=None` is the only accepted argument
    // according to array API. For GPU, when `stream=None`  producer
    // must assume the legacy default stream. Reference:
    // https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html
    auto ptr_to_tensor = FromDLPackCapsule(
        name, tensor.attr("__dlpack__")(
                  py::arg("stream") =
                      py::int_(reinterpret_cast<int64_t>(proxy_stream))));

    // In case there is a pending job on the data, where this capsule
    // is pointing to, we need to wait for it to finish before returning
    // capsule.
    // We synchronize on the proxy stream explicitly since that what we
    // pass to external tensor's `__dlpack__` method.
    err = cudaStreamSynchronize(proxy_stream);
    if (err != cudaSuccess) {
      throw PythonBackendException(
          "Failed to synchronize CUDA device with id " +
          std::to_string(
              overridden ? capsule_device_info.second : current_device));
    }

    return ptr_to_tensor;
#else
    throw PythonBackendException(
        "DLPack capsule passed pointer to memory allocated on GPU device, \
          when GPU is not available");
#endif
  } else if (
      capsule_device_info.first != DLDeviceType::kDLCPU &&
      capsule_device_info.first != DLDeviceType::kDLCUDAHost) {
    throw PythonBackendException(
        "DLDevice type " + std::to_string(capsule_device_info.first) +
        " is not support by Python backend.");
  }

  // If data is located on a CPU, `stream=None` is the only accepted argument
  // according to array API.
  // Reference:
  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html
  return FromDLPackCapsule(
      name, tensor.attr("__dlpack__")(py::arg("stream") = py::none()));
}

std::shared_ptr<PbTensor>
PbTensor::FromDLPackCapsule(
    const std::string& name, const py::capsule& dlpack_tensor)
{
  DLManagedTensor* dl_managed_tensor =
      static_cast<DLManagedTensor*>(dlpack_tensor.get_pointer());

  void* memory_ptr = dl_managed_tensor->dl_tensor.data;
  memory_ptr = reinterpret_cast<char*>(memory_ptr) +
               dl_managed_tensor->dl_tensor.byte_offset;

  int64_t* strides = dl_managed_tensor->dl_tensor.strides;

  int ndim = dl_managed_tensor->dl_tensor.ndim;
  std::vector<int64_t> dims(
      dl_managed_tensor->dl_tensor.shape,
      dl_managed_tensor->dl_tensor.shape + ndim);

  // Check if the input is contiguous and in C order
  if (strides != nullptr) {
    int64_t calculated_stride{1};
    bool is_contiguous_c_order = true;
    for (size_t i = 1; i < dims.size(); i++) {
      if (dims[ndim - i] != 1) {
        if (strides[ndim - i] != calculated_stride) {
          is_contiguous_c_order = false;
          break;
        }

        calculated_stride *= dims[ndim - i];
      }
    }

    if (!is_contiguous_c_order) {
      throw PythonBackendException(
          "DLPack tensor is not contiguous. Only contiguous DLPack "
          "tensors that are stored in C-Order are supported.");
    }
  }

  TRITONSERVER_MemoryType memory_type;
  int64_t memory_type_id;

  switch (dl_managed_tensor->dl_tensor.device.device_type) {
    case DLDeviceType::kDLCUDA:
      memory_type = TRITONSERVER_MEMORY_GPU;
      memory_type_id = dl_managed_tensor->dl_tensor.device.device_id;
      break;
    case DLDeviceType::kDLCPU:
      memory_type = TRITONSERVER_MEMORY_CPU;
      memory_type_id = 0;
      break;
    case DLDeviceType::kDLCUDAHost:
      memory_type = TRITONSERVER_MEMORY_CPU;
      memory_type_id = 0;
      break;
    default:
      throw PythonBackendException(
          "DLDevice type " +
          std::to_string(dl_managed_tensor->dl_tensor.device.device_type) +
          " is not support by Python backend.");
      break;
  }

  TRITONSERVER_DataType dtype =
      dlpack_to_triton_type(dl_managed_tensor->dl_tensor.dtype);

  // Calculate tensor size.
  uint64_t byte_size = 1;
  for (auto& dim : dims) {
    byte_size *= dim;
  }
  byte_size *= (dl_managed_tensor->dl_tensor.dtype.bits + 7) / 8;

  PyCapsule_SetName(dlpack_tensor.ptr(), "used_dlpack");
  return std::make_unique<PbTensor>(
      name, dims, dtype, memory_type, memory_type_id, memory_ptr, byte_size,
      dl_managed_tensor);
}
#endif  // TRITON_PB_STUB

PbTensor::~PbTensor() noexcept(false)
{
  pb_memory_.reset();
  DeleteDLPack();

#ifdef TRITON_PB_STUB
  {
    py::gil_scoped_acquire acquire;
    py::array numpy_array_local(std::move(numpy_array_));
    py::array numpy_array_serialized_local(std::move(numpy_array_serialized_));
  }
#endif
}

const std::string&
PbTensor::Name() const
{
  return name_;
}

#ifdef TRITON_PB_STUB
const py::array*
PbTensor::AsNumpy() const
{
  if (!IsCPU()) {
    throw PythonBackendException(
        "Tensor is stored in GPU and cannot be converted to NumPy.");
  }

  if (dtype_ == TRITONSERVER_TYPE_BF16) {
    throw PythonBackendException(
        "Tensor dtype is BF16 and cannot be converted to NumPy. Use "
        "to_dlpack() and from_dlpack() instead.");
  }

  return &numpy_array_;
}
#endif  // TRITON_PB_STUB

void
PbTensor::SaveToSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool, bool copy_gpu)
{
  if (!tensor_shm_.data_) {
    uint64_t byte_size;
    if (!pb_memory_) {
      byte_size = sizeof(TensorShm) + sizeof(int64_t) * dims_.size() +
                  PbString::ShmStructSize(name_) +
                  PbMemory::ShmStructSize(memory_type_, byte_size_);

    } else {
      byte_size = sizeof(TensorShm) + sizeof(int64_t) * dims_.size() +
                  PbString::ShmStructSize(name_);
    }
    tensor_shm_ = shm_pool->Construct<char>(byte_size);

    tensor_shm_ptr_ = reinterpret_cast<TensorShm*>(tensor_shm_.data_.get());
    tensor_shm_ptr_->dtype = dtype_;
    tensor_shm_ptr_->dims_count = dims_.size();
    shm_handle_ = tensor_shm_.handle_;

    dims_shm_ptr_ = reinterpret_cast<int64_t*>(
        reinterpret_cast<char*>(tensor_shm_ptr_) + sizeof(TensorShm));

    // Write the dimensions data to shared memory.
    for (size_t i = 0; i < dims_.size(); i++) {
      dims_shm_ptr_[i] = dims_[i];
    }

    std::size_t name_offset =
        sizeof(TensorShm) + sizeof(int64_t) * dims_.size();
    name_shm_ = PbString::Create(
        name_, reinterpret_cast<char*>(tensor_shm_ptr_) + name_offset,
        shm_handle_ + name_offset);
    std::size_t pb_memory_offset = name_offset + PbString::ShmStructSize(name_);

    if (!pb_memory_) {
      pb_memory_ = PbMemory::Create(
          shm_pool, memory_type_, memory_type_id_, byte_size_,
          reinterpret_cast<char*>(memory_ptr_),
          reinterpret_cast<char*>(tensor_shm_ptr_) + pb_memory_offset,
          shm_handle_ + pb_memory_offset, copy_gpu);
      tensor_shm_ptr_->memory = 0;
    } else {
      tensor_shm_ptr_->memory = pb_memory_->ShmHandle();
    }

    memory_ptr_ = pb_memory_->DataPtr();
  }
}

std::unique_ptr<PbTensor>
PbTensor::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t tensor_handle, bool open_cuda_handle)
{
  AllocatedSharedMemory<char> tensor_shm = shm_pool->Load<char>(tensor_handle);
  TensorShm* tensor_shm_ptr =
      reinterpret_cast<TensorShm*>(tensor_shm.data_.get());
  size_t name_offset =
      sizeof(TensorShm) + sizeof(int64_t) * tensor_shm_ptr->dims_count;
  std::unique_ptr<PbString> name_shm = PbString::LoadFromSharedMemory(
      tensor_handle + name_offset, tensor_shm.data_.get() + name_offset);

  std::unique_ptr<PbMemory> pb_memory;
  if (tensor_shm_ptr->memory == 0) {
    std::size_t pb_memory_offset = name_offset + name_shm->Size();
    pb_memory = PbMemory::LoadFromSharedMemory(
        shm_pool, pb_memory_offset, tensor_shm.data_.get() + pb_memory_offset,
        open_cuda_handle);
  } else {
    pb_memory = PbMemory::LoadFromSharedMemory(
        shm_pool, tensor_shm_ptr->memory, open_cuda_handle);
  }

  return std::unique_ptr<PbTensor>(
      new PbTensor(tensor_shm, name_shm, pb_memory));
}

TRITONSERVER_DataType
PbTensor::TritonDtype() const
{
  return dtype_;
}

void*
PbTensor::DataPtr()
{
  return memory_ptr_;
}

bi::managed_external_buffer::handle_t
PbTensor::ShmHandle()
{
  return shm_handle_;
}

PbTensor::PbTensor(
    AllocatedSharedMemory<char>& tensor_shm,
    std::unique_ptr<PbString>& name_shm, std::unique_ptr<PbMemory>& pb_memory)
    : tensor_shm_(std::move(tensor_shm)), name_shm_(std::move(name_shm)),
      pb_memory_(std::move(pb_memory))
{
  tensor_shm_ptr_ = reinterpret_cast<TensorShm*>(tensor_shm_.data_.get());
  dims_shm_ptr_ = reinterpret_cast<int64_t*>(
      reinterpret_cast<char*>(tensor_shm_ptr_) + sizeof(TensorShm));

  name_ = name_shm_->String();
  dims_ = std::vector<int64_t>(
      dims_shm_ptr_, dims_shm_ptr_ + tensor_shm_ptr_->dims_count);
  dtype_ = tensor_shm_ptr_->dtype;
  dl_managed_tensor_ = nullptr;
  byte_size_ = pb_memory_->ByteSize();
  memory_ptr_ = pb_memory_->DataPtr();
  memory_type_ = pb_memory_->MemoryType();
  memory_type_id_ = pb_memory_->MemoryTypeId();
  shm_handle_ = tensor_shm_.handle_;

#ifdef TRITON_PB_STUB
  if (memory_type_ == TRITONSERVER_MEMORY_CPU ||
      memory_type_ == TRITONSERVER_MEMORY_CPU_PINNED) {
    if (dtype_ == TRITONSERVER_TYPE_BF16) {
      // No native numpy representation for BF16. DLPack should be used instead.
      numpy_array_ = py::none();
    } else if (dtype_ != TRITONSERVER_TYPE_BYTES) {
      py::object numpy_array =
          py::array(triton_to_pybind_dtype(dtype_), dims_, (void*)memory_ptr_);
      numpy_array_ = numpy_array.attr("view")(triton_to_numpy_type(dtype_));
    } else {
      py::object numpy_array = py::array(
          triton_to_pybind_dtype(TRITONSERVER_TYPE_UINT8), {byte_size_},
          (void*)memory_ptr_);
      py::module triton_pb_utils =
          py::module::import("triton_python_backend_utils");
      numpy_array_ =
          triton_pb_utils.attr("deserialize_bytes_tensor")(numpy_array)
              .attr("reshape")(dims_);
    }
  } else {
    numpy_array_ = py::none();
  }
#endif
}
}}}  // namespace triton::backend::python
