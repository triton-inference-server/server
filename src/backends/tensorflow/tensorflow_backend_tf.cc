// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "trtis/tensorflow_backend_tf.h"

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu/gpu_mem_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"

TRTISTF_Error* TRTISTF_ErrorNew(const std::string& str);
TRTISTF_Shape* TRTISTF_ShapeNew(size_t rank, int64_t* dims);
void TRTISTF_ShapeDelete(TRTISTF_Shape* shape);
TRTISTF_IOList* TRTISTF_IOListNew(
    const char* name, const char* inmodel_name, TRTISTF_IOList* next);
void TRTISTF_IOListDelete(TRTISTF_IOList* list);

// If TensorFlow status is non-OK, return the equivalent TRTISTF_Error
#define RETURN_IF_TF_ERROR(TFS)                          \
  do {                                                   \
    const tensorflow::Status& status__ = (TFS);          \
    if (status__.code() != 0) {                          \
      return TRTISTF_ErrorNew(status__.error_message()); \
    }                                                    \
  } while (false)

namespace {

static TRTISTF_DataType
ConvertDataType(tensorflow::DataType dtype)
{
  switch (dtype) {
    case tensorflow::DT_INVALID:
      return TRTISTF_DataType::TRTISTF_TYPE_INVALID;
    case tensorflow::DT_BOOL:
      return TRTISTF_DataType::TRTISTF_TYPE_BOOL;
    case tensorflow::DT_UINT8:
      return TRTISTF_DataType::TRTISTF_TYPE_UINT8;
    case tensorflow::DT_UINT16:
      return TRTISTF_DataType::TRTISTF_TYPE_UINT16;
    case tensorflow::DT_UINT32:
      return TRTISTF_DataType::TRTISTF_TYPE_UINT32;
    case tensorflow::DT_UINT64:
      return TRTISTF_DataType::TRTISTF_TYPE_UINT64;
    case tensorflow::DT_INT8:
      return TRTISTF_DataType::TRTISTF_TYPE_INT8;
    case tensorflow::DT_INT16:
      return TRTISTF_DataType::TRTISTF_TYPE_INT16;
    case tensorflow::DT_INT32:
      return TRTISTF_DataType::TRTISTF_TYPE_INT32;
    case tensorflow::DT_INT64:
      return TRTISTF_DataType::TRTISTF_TYPE_INT64;
    case tensorflow::DT_HALF:
      return TRTISTF_DataType::TRTISTF_TYPE_FP16;
    case tensorflow::DT_FLOAT:
      return TRTISTF_DataType::TRTISTF_TYPE_FP32;
    case tensorflow::DT_DOUBLE:
      return TRTISTF_DataType::TRTISTF_TYPE_FP64;
    case tensorflow::DT_STRING:
      return TRTISTF_DataType::TRTISTF_TYPE_STRING;
    default:
      break;
  }

  return TRTISTF_DataType::TRTISTF_TYPE_INVALID;
}

tensorflow::DataType
ConvertDataType(TRTISTF_DataType dtype)
{
  switch (dtype) {
    case TRTISTF_DataType::TRTISTF_TYPE_INVALID:
      return tensorflow::DT_INVALID;
    case TRTISTF_DataType::TRTISTF_TYPE_BOOL:
      return tensorflow::DT_BOOL;
    case TRTISTF_DataType::TRTISTF_TYPE_UINT8:
      return tensorflow::DT_UINT8;
    case TRTISTF_DataType::TRTISTF_TYPE_UINT16:
      return tensorflow::DT_UINT16;
    case TRTISTF_DataType::TRTISTF_TYPE_UINT32:
      return tensorflow::DT_UINT32;
    case TRTISTF_DataType::TRTISTF_TYPE_UINT64:
      return tensorflow::DT_UINT64;
    case TRTISTF_DataType::TRTISTF_TYPE_INT8:
      return tensorflow::DT_INT8;
    case TRTISTF_DataType::TRTISTF_TYPE_INT16:
      return tensorflow::DT_INT16;
    case TRTISTF_DataType::TRTISTF_TYPE_INT32:
      return tensorflow::DT_INT32;
    case TRTISTF_DataType::TRTISTF_TYPE_INT64:
      return tensorflow::DT_INT64;
    case TRTISTF_DataType::TRTISTF_TYPE_FP16:
      return tensorflow::DT_HALF;
    case TRTISTF_DataType::TRTISTF_TYPE_FP32:
      return tensorflow::DT_FLOAT;
    case TRTISTF_DataType::TRTISTF_TYPE_FP64:
      return tensorflow::DT_DOUBLE;
    case TRTISTF_DataType::TRTISTF_TYPE_STRING:
      return tensorflow::DT_STRING;
    default:
      break;
  }

  return tensorflow::DT_INVALID;
}

void
ConvertShape(TRTISTF_Shape* shape, tensorflow::TensorShape* tfshape)
{
  for (size_t itr = 0; itr < shape->rank_; itr++) {
    const int64_t dim = shape->dims_[itr];
    tfshape->AddDim(dim);
  }
}

TRTISTF_Shape*
ConvertShape(const tensorflow::TensorShape& tfshape)
{
  TRTISTF_Shape* shape = new TRTISTF_Shape;
  shape->rank_ = tfshape.dims();
  shape->dims_ = nullptr;

  if (shape->rank_ > 0) {
    shape->dims_ = new int64_t[shape->rank_];
    for (int i = 0; i < tfshape.dims(); ++i) {
      shape->dims_[i] = tfshape.dim_size(i);
    }
  }
  return shape;
}

std::string
PrecisionModeToString(const TRTISTF_TFTRTPrecisionMode m)
{
  switch (m) {
    case TRTISTF_MODE_INT8:
      return "INT8";
    case TRTISTF_MODE_FP16:
      return "FP16";
    default:
      return "FP32";
  }
}

// Helper function to help setting Callable related parts properly,
// adapoted from
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/graph_execution_state.cc#L382
bool
IsGPUFeedAndFetchSupported(TRTISTF_DataType dtype)
{
  switch (ConvertDataType(dtype)) {
    case tensorflow::DT_BFLOAT16:
    case tensorflow::DT_BOOL:
    case tensorflow::DT_COMPLEX128:
    case tensorflow::DT_COMPLEX64:
    case tensorflow::DT_DOUBLE:
    case tensorflow::DT_FLOAT:
    case tensorflow::DT_HALF:
    case tensorflow::DT_INT16:
    case tensorflow::DT_INT64:
    case tensorflow::DT_INT8:
    case tensorflow::DT_UINT16:
    case tensorflow::DT_UINT8:
      return true;
    default:
      return false;
  }
}

void
NewSessionOptions(
    const bool has_graph_level, const int graph_level,
    const bool allow_gpu_memory_growth,
    const float per_process_gpu_memory_fraction,
    const bool allow_soft_placement,
    const std::map<int, std::vector<float>>& memory_limit_mb,
    const TRTISTF_TFTRTConfig* tftrt_config,
    tensorflow::SessionOptions* session_options)
{
  session_options->config.mutable_gpu_options()->set_allow_growth(
      allow_gpu_memory_growth);
  session_options->config.mutable_gpu_options()
      ->set_per_process_gpu_memory_fraction(per_process_gpu_memory_fraction);
  session_options->config.set_allow_soft_placement(allow_soft_placement);

  // Create virtual devices
  if (!memory_limit_mb.empty()) {
    (*(session_options->config.mutable_device_count()))["GPU"] =
        memory_limit_mb.size();
    std::string visible_device_list = "";
    for (const auto& v : memory_limit_mb) {
      auto virtual_devices = session_options->config.mutable_gpu_options()
                                 ->mutable_experimental()
                                 ->add_virtual_devices();
      visible_device_list += ("," + std::to_string(v.first));
      if (!v.second.empty()) {
        for (float mb : v.second) {
          virtual_devices->add_memory_limit_mb(mb);
        }
      }
    }
    session_options->config.mutable_gpu_options()->set_visible_device_list(
        visible_device_list.substr(1));
  }

  // Enable/disable XLA based on the model config optimization
  // setting.
  tensorflow::OptimizerOptions::GlobalJitLevel xla =
      tensorflow::OptimizerOptions::DEFAULT;
  if (has_graph_level) {
    if (graph_level == -1) {
      xla = tensorflow::OptimizerOptions::OFF;
    } else if (graph_level == 1) {
      xla = tensorflow::OptimizerOptions::ON_1;
    } else if (graph_level > 1) {
      xla = tensorflow::OptimizerOptions::ON_2;
    }
  }

  session_options->config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(xla);

  // TF-TRT optimization. Parameters that are not specified in 'tftrt_config'
  // are specified based on:
  // https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/tensorrt/test/test_tftrt.py#L238
  if (tftrt_config != nullptr) {
    auto opt_config = session_options->config.mutable_graph_options()
                          ->mutable_rewrite_options();
    opt_config->set_meta_optimizer_iterations(tensorflow::RewriterConfig::ONE);
    opt_config->add_optimizers("constfold");
    opt_config->add_optimizers("layout");
    auto trt_optimizer = opt_config->add_custom_optimizers();
    trt_optimizer->set_name("TensorRTOptimizer");

    auto trt_parameter_map = trt_optimizer->mutable_parameter_map();
    (*trt_parameter_map)["is_dynamic_op"].set_b(tftrt_config->is_dynamic_op_);
    (*trt_parameter_map)["minimum_segment_size"].set_i(
        tftrt_config->minimum_segment_size_);
    (*trt_parameter_map)["precision_mode"].set_s(
        PrecisionModeToString(tftrt_config->precision_mode_));
    (*trt_parameter_map)["max_batch_size"].set_i(tftrt_config->max_batch_size_);
    (*trt_parameter_map)["max_workspace_size_bytes"].set_i(
        tftrt_config->max_workspace_size_bytes_);
    (*trt_parameter_map)["max_cached_engines"].set_i(
        tftrt_config->max_cached_engines_);
  }
}

// Get the device name in model session given a non-negative device_id.
TRTISTF_Error*
GetTFGPUDeviceName(
    std::string* device_name, tensorflow::Session* session, const int device_id)
{
  if (device_id >= 0) {
    std::vector<tensorflow::DeviceAttributes> devices;
    RETURN_IF_TF_ERROR(session->ListDevices(&devices));
    for (const auto& d : devices) {
      if (d.device_type() == "GPU" || d.device_type() == "gpu") {
        // Session seems to be aware of all devices on the system,
        // thus need to filter out the correct full name for the device
        tensorflow::DeviceNameUtils::ParsedName parsed;
        if (tensorflow::DeviceNameUtils::ParseFullName(d.name(), &parsed)) {
          if (parsed.id == device_id) {
            *device_name = d.name();
            break;
          }
        }
      }
    }
  }
  return nullptr;
}

//
// TensorImpl
//
class TensorImpl {
 public:
  TensorImpl(
      const char* name, TRTISTF_DataType dtype, TRTISTF_Shape* shape,
      const tensorflow::TensorShape& tfshape, const int tf_gpu_id);
  TensorImpl(tensorflow::Tensor&& tftensor);
  ~TensorImpl();

  const std::string& Name() const { return name_; }
  TRTISTF_DataType DataType() const { return dtype_; }
  TRTISTF_Shape* Shape() const { return shape_; }

  tensorflow::Tensor& TFTensor() { return tftensor_; }

  char* Base() const { return nonstring_base_; }
  size_t ByteSize() const { return nonstring_byte_size_; }
  bool IsGPUTensor() const { return gpu_tensor_; }

  const std::string& String(size_t idx) const;
  void SetString(size_t idx, const std::string& str);

 private:
  void Init();

  const std::string name_;
  const TRTISTF_DataType dtype_;
  TRTISTF_Shape* shape_;

  tensorflow::Tensor tftensor_;
  char* nonstring_base_;
  size_t nonstring_byte_size_;
  bool gpu_tensor_;
};


TensorImpl::TensorImpl(
    const char* name, TRTISTF_DataType dtype, TRTISTF_Shape* shape,
    const tensorflow::TensorShape& tfshape, const int tf_gpu_id)
    : name_(name), dtype_(dtype), shape_(shape)
{
  // Only request for GPU allocator for supported data type
  auto a = ((tf_gpu_id >= 0) && IsGPUFeedAndFetchSupported(dtype))
               ? tensorflow::GPUProcessState::singleton()->GetGPUAllocator(
                     tensorflow::GPUOptions(), tensorflow::TfGpuId(tf_gpu_id),
                     1 << 28 /* total_memory_size */)
               : nullptr;
  if (a == nullptr) {
    tftensor_ = tensorflow::Tensor(ConvertDataType(dtype), tfshape);
  } else {
    tftensor_ = tensorflow::Tensor(a, ConvertDataType(dtype), tfshape);
  }
  Init();
}

TensorImpl::TensorImpl(tensorflow::Tensor&& tftensor)
    : name_(), dtype_(ConvertDataType(tftensor.dtype())),
      shape_(ConvertShape(tftensor.shape())), tftensor_(std::move(tftensor))
{
  Init();
}

TensorImpl::~TensorImpl()
{
  TRTISTF_ShapeDelete(shape_);
}

void
TensorImpl::Init()
{
  nonstring_base_ = nullptr;
  nonstring_byte_size_ = 0;
  gpu_tensor_ = false;

  // Implement differently for string and non-string
  if (tftensor_.dtype() != tensorflow::DT_STRING) {
    auto flat = tftensor_.bit_casted_shaped<char, 1>(
        {tftensor_.NumElements() *
         tensorflow::DataTypeSize(tftensor_.dtype())});
    nonstring_base_ = static_cast<char*>(flat.data());
    nonstring_byte_size_ = flat.size();

    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, nonstring_base_);
    gpu_tensor_ =
        ((err == cudaSuccess) &&
         (attributes.memoryType == cudaMemoryTypeDevice));
  }
}

const std::string&
TensorImpl::String(size_t idx) const
{
  auto flat = tftensor_.flat<std::string>();
  return flat(idx);
}

void
TensorImpl::SetString(size_t idx, const std::string& str)
{
  auto flat = tftensor_.flat<std::string>();
  flat(idx) = str;
}

//
// ModelImpl
//
class ModelImpl {
 public:
  ModelImpl(
      const std::string& model_name,
      std::unique_ptr<tensorflow::SavedModelBundle> bundle,
      TRTISTF_IOList* inputs, TRTISTF_IOList* outputs,
      const std::string& device_name);
  ModelImpl(
      const std::string& model_name, tensorflow::Session* session,
      TRTISTF_IOList* inputs, TRTISTF_IOList* outputs,
      const std::string& device_name);
  ~ModelImpl();

  TRTISTF_IOList* Inputs() const { return inputs_; }
  TRTISTF_IOList* Outputs() const { return outputs_; }
  const std::string& DeviceName() const { return device_name_; }

  TRTISTF_Error* MakeCallable(const tensorflow::CallableOptions& opts);

  TRTISTF_Error* Run(
      TRTISTF_TensorList* input_tensors,
      const std::vector<std::string>& output_names,
      TRTISTF_TensorList** output_tensors);

 private:
  const std::string model_name_;
  std::unique_ptr<tensorflow::SavedModelBundle> bundle_;
  tensorflow::Session* session_;
  TRTISTF_IOList* inputs_;
  TRTISTF_IOList* outputs_;

  // Variables for callable
  bool has_callable_;
  std::string device_name_;
  tensorflow::Session::CallableHandle callable_;
  // RunCallable will return all outputs specified in callable option in order,
  // using map to quickly locate the requested output for each request.
  std::map<std::string, size_t> output_index_map_;
};

ModelImpl::ModelImpl(
    const std::string& model_name,
    std::unique_ptr<tensorflow::SavedModelBundle> bundle,
    TRTISTF_IOList* inputs, TRTISTF_IOList* outputs,
    const std::string& device_name)
    : model_name_(model_name), bundle_(std::move(bundle)), inputs_(inputs),
      outputs_(outputs), has_callable_(false), device_name_(device_name)
{
  session_ = bundle_->session.release();
}

ModelImpl::ModelImpl(
    const std::string& model_name, tensorflow::Session* session,
    TRTISTF_IOList* inputs, TRTISTF_IOList* outputs,
    const std::string& device_name)
    : model_name_(model_name), session_(session), inputs_(inputs),
      outputs_(outputs), has_callable_(false), device_name_(device_name)
{
}

ModelImpl::~ModelImpl()
{
  if (session_ != nullptr) {
    if (has_callable_) {
      session_->ReleaseCallable(callable_).IgnoreError();
    }
    session_->Close().IgnoreError();
    delete session_;
    session_ = nullptr;
  }

  TRTISTF_IOListDelete(inputs_);
  TRTISTF_IOListDelete(outputs_);
}

TRTISTF_Error*
ModelImpl::MakeCallable(const tensorflow::CallableOptions& opts)
{
  if (has_callable_) {
    session_->ReleaseCallable(callable_).IgnoreError();
    has_callable_ = false;
  }

  RETURN_IF_TF_ERROR(session_->MakeCallable(opts, &callable_));
  for (int idx = 0; idx < opts.fetch_size(); idx++) {
    output_index_map_[opts.fetch(idx)] = idx;
  }
  has_callable_ = true;
  return nullptr;
}

TRTISTF_Error*
ModelImpl::Run(
    TRTISTF_TensorList* input_tensors,
    const std::vector<std::string>& output_names,
    TRTISTF_TensorList** output_tensors)
{
  // I/O needs to be prepared differently for callable
  if (has_callable_) {
    std::vector<tensorflow::Tensor> tfinputs;

    for (TRTISTF_TensorList* itr = input_tensors; itr != nullptr;
         itr = itr->next_) {
      if (itr->tensor_ != nullptr) {
        TensorImpl* tensor = reinterpret_cast<TensorImpl*>(itr->tensor_);
        tfinputs.emplace_back(std::move(tensor->TFTensor()));
      }
    }
    TRTISTF_TensorListDelete(input_tensors);

    tensorflow::RunMetadata meta_data;
    std::vector<tensorflow::Tensor> tfoutputs;
    RETURN_IF_TF_ERROR(
        session_->RunCallable(callable_, tfinputs, &tfoutputs, &meta_data));

    *output_tensors = nullptr;
    for (auto ri = output_names.rbegin(); ri != output_names.rend(); ++ri) {
      const auto oidx = output_index_map_[*ri];
      TRTISTF_Tensor* tensor = reinterpret_cast<TRTISTF_Tensor*>(
          new TensorImpl(std::move(tfoutputs[oidx])));
      *output_tensors = TRTISTF_TensorListNew(tensor, *output_tensors);
    }

    // Callable documentation suggested caller to use Device::Sync() to ensure
    // that GPU outputs are ready. But it appears that in the default setting,
    // TensorFlow will issue the Sync() call at the end of execution, so we
    // won't do it twice. Ref:
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/executor.cc#L2578
  } else {
    std::vector<std::pair<std::string, tensorflow::Tensor>> tfinputs;

    for (TRTISTF_TensorList* itr = input_tensors; itr != nullptr;
         itr = itr->next_) {
      if (itr->tensor_ != nullptr) {
        TensorImpl* tensor = reinterpret_cast<TensorImpl*>(itr->tensor_);
        tfinputs.emplace_back(
            std::make_pair(tensor->Name(), std::move(tensor->TFTensor())));
      }
    }
    TRTISTF_TensorListDelete(input_tensors);

    std::vector<tensorflow::Tensor> tfoutputs;
    RETURN_IF_TF_ERROR(session_->Run(tfinputs, output_names, {}, &tfoutputs));

    *output_tensors = nullptr;
    for (std::vector<tensorflow::Tensor>::reverse_iterator ri =
             tfoutputs.rbegin();
         ri != tfoutputs.rend(); ++ri) {
      TRTISTF_Tensor* tensor =
          reinterpret_cast<TRTISTF_Tensor*>(new TensorImpl(std::move(*ri)));
      *output_tensors = TRTISTF_TensorListNew(tensor, *output_tensors);
    }
  }

  return nullptr;
}

}  // namespace

//
// TRTISTF_Error
//

TRTISTF_Error*
TRTISTF_ErrorNew(const std::string& str)
{
  TRTISTF_Error* error = new TRTISTF_Error;
  error->msg_ = new char[str.size() + 1];
  strcpy(error->msg_, str.c_str());
  return error;
}

void
TRTISTF_ErrorDelete(TRTISTF_Error* error)
{
  if (error == nullptr) {
    return;
  }

  delete[] error->msg_;
  delete error;
}

//
// TRTISTF_Shape
//
TRTISTF_Shape*
TRTISTF_ShapeNew(size_t rank, int64_t* dims)
{
  TRTISTF_Shape* shape = new TRTISTF_Shape;
  shape->rank_ = rank;
  shape->dims_ = nullptr;
  if (rank > 0) {
    shape->dims_ = new int64_t[rank];
    memcpy(shape->dims_, dims, rank * sizeof(int64_t));
  }

  return shape;
}

void
TRTISTF_ShapeDelete(TRTISTF_Shape* shape)
{
  if (shape != nullptr) {
    delete[] shape->dims_;
    delete shape;
  }
}

//
// TRTISTF_IOList
//
TRTISTF_IOList*
TRTISTF_IOListNew(
    const char* name, const char* inmodel_name, TRTISTF_IOList* next)
{
  TRTISTF_IO* io = new TRTISTF_IO;

  io->name_ = nullptr;
  if (name != nullptr) {
    io->name_ = new char[strlen(name) + 1];
    strcpy(io->name_, name);
  }

  io->inmodel_name_ = nullptr;
  if (inmodel_name != nullptr) {
    io->inmodel_name_ = new char[strlen(inmodel_name) + 1];
    strcpy(io->inmodel_name_, inmodel_name);
  }

  io->data_type_ = TRTISTF_DataType::TRTISTF_TYPE_INVALID;
  io->shape_ = nullptr;

  TRTISTF_IOList* iol = new TRTISTF_IOList;
  iol->io_ = io;
  iol->next_ = next;

  return iol;
}

void
TRTISTF_IOListDelete(TRTISTF_IOList* list)
{
  while (list != nullptr) {
    if (list->io_ != nullptr) {
      delete[] list->io_->name_;
      delete[] list->io_->inmodel_name_;
      TRTISTF_ShapeDelete(list->io_->shape_);
      delete list->io_;
    }

    TRTISTF_IOList* next = list->next_;
    delete list;
    list = next;
  }
}

//
// TRTISTF_TensorList
//
TRTISTF_TensorList*
TRTISTF_TensorListNew(TRTISTF_Tensor* tensor, TRTISTF_TensorList* next)
{
  TRTISTF_TensorList* tl = new TRTISTF_TensorList;
  tl->tensor_ = tensor;
  tl->next_ = next;
  return tl;
}

void
TRTISTF_TensorListDelete(TRTISTF_TensorList* list)
{
  while (list != nullptr) {
    if (list->tensor_ != nullptr) {
      TensorImpl* tensor = reinterpret_cast<TensorImpl*>(list->tensor_);
      delete tensor;
      list->tensor_ = nullptr;
    }

    TRTISTF_TensorList* next = list->next_;
    list->next_ = nullptr;
    delete list;
    list = next;
  }
}

//
// TRTISTF_Tensor
//
TRTISTF_Tensor*
TRTISTF_TensorNew(
    const char* name, TRTISTF_DataType dtype, size_t shape_rank,
    int64_t* shape_dims, const int tf_gpu_id)
{
  TRTISTF_Shape* shape = TRTISTF_ShapeNew(shape_rank, shape_dims);
  tensorflow::TensorShape tfshape;
  ConvertShape(shape, &tfshape);

  TensorImpl* tensor = new TensorImpl(name, dtype, shape, tfshape, tf_gpu_id);
  // If data type is non-string, make sure TensorImpl contains valid TF tensor
  if (dtype != TRTISTF_DataType::TRTISTF_TYPE_STRING) {
    // tensor's byte size is set to value required and it is independent to
    // the data pointer. So make sure data is not nullptr if byte size > 0
    if ((tensor->ByteSize() != 0) && (tensor->Base() == nullptr)) {
      delete tensor;
      return nullptr;
    }
  }
  return reinterpret_cast<TRTISTF_Tensor*>(tensor);
}

TRTISTF_DataType
TRTISTF_TensorDataType(TRTISTF_Tensor* tensor)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  return t->DataType();
}

int64_t
TRTISTF_TensorDataTypeByteSize(TRTISTF_Tensor* tensor)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  return tensorflow::DataTypeSize(t->TFTensor().dtype());
}

TRTISTF_Shape*
TRTISTF_TensorShape(TRTISTF_Tensor* tensor)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  return t->Shape();
}

char*
TRTISTF_TensorData(TRTISTF_Tensor* tensor)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  return t->Base();
}

bool
TRTISTF_TensorIsGPUTensor(TRTISTF_Tensor* tensor)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  return t->IsGPUTensor();
}

size_t
TRTISTF_TensorDataByteSize(TRTISTF_Tensor* tensor)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  return t->ByteSize();
}

const char*
TRTISTF_TensorString(TRTISTF_Tensor* tensor, size_t idx, size_t* length)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  const std::string& str = t->String(idx);
  *length = str.length();
  return str.c_str();
}

void
TRTISTF_TensorSetString(
    TRTISTF_Tensor* tensor, size_t idx, const char* cstr, size_t length)
{
  TensorImpl* t = reinterpret_cast<TensorImpl*>(tensor);
  std::string str;
  if (cstr != nullptr) {
    str = std::string(cstr, length);
  }

  t->SetString(idx, str);
}

//
// TRTISTF_Model
//
TRTISTF_Error*
TRTISTF_ModelCreateFromGraphDef(
    TRTISTF_Model** trtistf_model, const char* model_name,
    const char* model_path, const int device_id, const bool has_graph_level,
    const int graph_level, const bool allow_gpu_memory_growth,
    const float per_process_gpu_memory_fraction,
    const bool allow_soft_placement,
    const std::map<int, std::vector<float>>& memory_limit_mb,
    const TRTISTF_TFTRTConfig* tftrt_config)
{
  tensorflow::SessionOptions session_options;
  NewSessionOptions(
      has_graph_level, graph_level, allow_gpu_memory_growth,
      per_process_gpu_memory_fraction, allow_soft_placement, memory_limit_mb,
      tftrt_config, &session_options);

  tensorflow::Session* session;
  RETURN_IF_TF_ERROR(tensorflow::NewSession(session_options, &session));

  tensorflow::GraphDef graph_def;
  RETURN_IF_TF_ERROR(tensorflow::ReadBinaryProto(
      tensorflow::Env::Default(), model_path, &graph_def));
  if (graph_def.node_size() == 0) {
    return TRTISTF_ErrorNew(
        "model " + std::string(model_name) + " has an empty network");
  }

  if (device_id != TRTISTF_MODEL_DEVICE) {
    // Clear the device field from the graphdef so that the default device
    // setting below will control which GPU the graph will run on
    for (tensorflow::NodeDef& node : *graph_def.mutable_node()) {
      if (!tensorflow::grappler::NodeIsOnCpu(&node)) {
        node.clear_device();
      }
    }
    // Set the default device to control the CPU/GPU that the graph runs
    // on.
    if (device_id == TRTISTF_NO_GPU_DEVICE) {
      tensorflow::graph::SetDefaultDevice("/cpu:0", &graph_def);
    } else {
      tensorflow::graph::SetDefaultDevice(
          "/gpu:" + std::to_string(device_id), &graph_def);
    }
  }

  RETURN_IF_TF_ERROR(session->Create(graph_def));

  // Go through all graph nodes and collect the possible inputs and
  // outputs. We use this to verify the requested inputs and outputs
  // when initializing. Unfortunately graphdef isn't explicit in
  // indicating inputs and outputs so we assume any Placeholder can be
  // an input and any node can be an output.
  TRTISTF_IOList* potential_inputs = nullptr;
  TRTISTF_IOList* potential_outputs = nullptr;
  for (const auto& node : graph_def.node()) {
    if (node.op() == "Placeholder") {
      potential_inputs =
          TRTISTF_IOListNew(node.name().c_str(), nullptr, potential_inputs);
    } else {
      potential_outputs =
          TRTISTF_IOListNew(node.name().c_str(), nullptr, potential_outputs);
    }
  }

  std::string device_name;
  if ((device_id != TRTISTF_MODEL_DEVICE) &&
      (device_id != TRTISTF_NO_GPU_DEVICE)) {
    TRTISTF_Error* err = GetTFGPUDeviceName(&device_name, session, device_id);
    if (err != nullptr) {
      return err;
    }
  }
  ModelImpl* model = new ModelImpl(
      model_name, session, potential_inputs, potential_outputs, device_name);
  *trtistf_model = reinterpret_cast<TRTISTF_Model*>(model);

  return nullptr;
}

TRTISTF_Error*
TRTISTF_ModelCreateFromSavedModel(
    TRTISTF_Model** trtistf_model, const char* model_name,
    const char* model_path, const int device_id, const bool has_graph_level,
    const int graph_level, const bool allow_gpu_memory_growth,
    const float per_process_gpu_memory_fraction,
    const bool allow_soft_placement,
    const std::map<int, std::vector<float>>& memory_limit_mb,
    const TRTISTF_TFTRTConfig* tftrt_config)
{
  tensorflow::SessionOptions session_options;
  NewSessionOptions(
      has_graph_level, graph_level, allow_gpu_memory_growth,
      per_process_gpu_memory_fraction, allow_soft_placement, memory_limit_mb,
      tftrt_config, &session_options);


  if (device_id != TRTISTF_MODEL_DEVICE) {
    // Set the default device to control the CPU/GPU that the graph runs
    // on.
    //
    // The GraphDef where we need to use this workaround is only
    // available in tensorflow/cc/saved_model/loader.cc so we use
    // allocator_type in pass in the gpu_device we want and then
    // loader.cc (our modified version) will use that to
    // SetDefaultDevice appropriately.
    if (device_id == TRTISTF_NO_GPU_DEVICE) {
      session_options.config.mutable_gpu_options()->set_allocator_type(
          "/cpu:0");
    } else {
      session_options.config.mutable_gpu_options()->set_allocator_type(
          "/gpu:" + std::to_string(device_id));
    }
  } else {
    // Make sure the allocator_type field is empty so that loader.cc doesn't set
    // device for the placement and let Tensorflow handle the model placement.
    session_options.config.mutable_gpu_options()->clear_allocator_type();
  }

  std::unique_ptr<tensorflow::SavedModelBundle> bundle(
      new tensorflow::SavedModelBundle);

  std::unordered_set<std::string> saved_model_tags;
  saved_model_tags.insert(tensorflow::kSavedModelTagServe);

  tensorflow::RunOptions run_options;
  RETURN_IF_TF_ERROR(tensorflow::LoadSavedModel(
      session_options, run_options, model_path, saved_model_tags,
      bundle.get()));

  // Verify that the bundle has the "serve" tag
  bool found_serve_tag = false;
  for (const auto& tag : bundle->meta_graph_def.meta_info_def().tags()) {
    if (tag == tensorflow::kSavedModelTagServe) {
      found_serve_tag = true;
      break;
    }
  }
  if (!found_serve_tag) {
    return TRTISTF_ErrorNew(
        "unable to load model '" + std::string(model_name) + "', expected '" +
        tensorflow::kSavedModelTagServe + "' tag");
  }

  // Verify that a "serving_default" signature exists, that is what
  // will be used to verify the inputs and outputs.
  static const std::string DEFAULT_SERVING_SIGNATURE_DEF_KEY("serving_default");
  static const std::string INIT_OP_SIGNATURE_DEF_KEY("__saved_model_init_op");
  static const std::string TRAIN_OP_SIGNATURE_DEF_KEY("__saved_model_train_op");
  auto sig_itr = bundle->meta_graph_def.signature_def().find(
      DEFAULT_SERVING_SIGNATURE_DEF_KEY);
  if (sig_itr == bundle->meta_graph_def.signature_def().end()) {
    // If default serving signature_def key is not found, maybe it is named
    // something else, use one that is neither init_op key nor train_op key
    for (sig_itr = bundle->meta_graph_def.signature_def().begin();
         sig_itr != bundle->meta_graph_def.signature_def().end(); sig_itr++) {
      if ((sig_itr->first != INIT_OP_SIGNATURE_DEF_KEY) &&
          (sig_itr->first != TRAIN_OP_SIGNATURE_DEF_KEY)) {
        LOG(WARNING) << "unable to find default serving signature '"
                     << DEFAULT_SERVING_SIGNATURE_DEF_KEY
                     << "', using signature '" << sig_itr->first << "'";
        break;
      }
    }
    if (sig_itr == bundle->meta_graph_def.signature_def().end()) {
      return TRTISTF_ErrorNew(
          "unable to load model '" + std::string(model_name) + "', expected '" +
          DEFAULT_SERVING_SIGNATURE_DEF_KEY + "' signature");
    }
  }

  const tensorflow::SignatureDef& def = sig_itr->second;

  // Collect the inputs...
  TRTISTF_IOList* inputs = nullptr;
  for (const auto& sin : def.inputs()) {
    inputs =
        TRTISTF_IOListNew(sin.first.c_str(), sin.second.name().c_str(), inputs);
    TRTISTF_IO* io = inputs->io_;

    const TRTISTF_DataType dt = ConvertDataType(sin.second.dtype());
    if (dt == TRTISTF_DataType::TRTISTF_TYPE_INVALID) {
      return TRTISTF_ErrorNew(
          "unable to process input '" + std::string(io->name_) + "' for '" +
          std::string(model_name) + "', unsupported datatype '" +
          tensorflow::DataType_Name(sin.second.dtype()) + "'");
    }

    io->data_type_ = dt;

    const tensorflow::TensorShapeProto& shape = sin.second.tensor_shape();
    int64_t shape_dims[shape.dim().size()];
    for (int i = 0; i < shape.dim().size(); ++i) {
      shape_dims[i] = shape.dim(i).size();
    }

    io->shape_ = TRTISTF_ShapeNew(shape.dim().size(), shape_dims);
  }

  // Collect the outputs...
  TRTISTF_IOList* outputs = nullptr;
  for (const auto& sout : def.outputs()) {
    outputs = TRTISTF_IOListNew(
        sout.first.c_str(), sout.second.name().c_str(), outputs);
    TRTISTF_IO* io = outputs->io_;

    const TRTISTF_DataType dt = ConvertDataType(sout.second.dtype());
    if (dt == TRTISTF_DataType::TRTISTF_TYPE_INVALID) {
      return TRTISTF_ErrorNew(
          "unable to process output '" + std::string(io->name_) + "' for '" +
          std::string(model_name) + "', unsupported datatype '" +
          tensorflow::DataType_Name(sout.second.dtype()) + "'");
    }

    io->data_type_ = dt;

    const tensorflow::TensorShapeProto& shape = sout.second.tensor_shape();
    int64_t shape_dims[shape.dim().size()];
    for (int i = 0; i < shape.dim().size(); ++i) {
      shape_dims[i] = shape.dim(i).size();
    }

    io->shape_ = TRTISTF_ShapeNew(shape.dim().size(), shape_dims);
  }

  std::string device_name;
  if ((device_id != TRTISTF_MODEL_DEVICE) &&
      (device_id != TRTISTF_NO_GPU_DEVICE)) {
    TRTISTF_Error* err =
        GetTFGPUDeviceName(&device_name, bundle->session.get(), device_id);
    if (err != nullptr) {
      return err;
    }
  }
  ModelImpl* model = new ModelImpl(
      model_name, std::move(bundle), inputs, outputs, device_name);
  *trtistf_model = reinterpret_cast<TRTISTF_Model*>(model);

  return nullptr;
}

void
TRTISTF_ModelDelete(TRTISTF_Model* model)
{
  if (model != nullptr) {
    ModelImpl* mi = reinterpret_cast<ModelImpl*>(model);
    delete mi;
  }
}

TRTISTF_IOList*
TRTISTF_ModelInputs(TRTISTF_Model* model)
{
  ModelImpl* m = reinterpret_cast<ModelImpl*>(model);
  return m->Inputs();
}

TRTISTF_IOList*
TRTISTF_ModelOutputs(TRTISTF_Model* model)
{
  ModelImpl* m = reinterpret_cast<ModelImpl*>(model);
  return m->Outputs();
}

TRTISTF_Error*
TRTISTF_ModelMakeCallable(
    TRTISTF_Model* model, const char** input_names,
    const TRTISTF_DataType* input_types, const size_t num_inputs,
    const char** output_names, const TRTISTF_DataType* output_types,
    const size_t num_outputs)
{
  ModelImpl* m = reinterpret_cast<ModelImpl*>(model);

  const auto& device_name = m->DeviceName();
  if (device_name.empty()) {
    return TRTISTF_ErrorNew(
        "model session does not have an assigned GPU device");
  }

  tensorflow::CallableOptions opts;
  for (size_t i = 0; i < num_inputs; ++i) {
    const std::string input_name = input_names[i];
    opts.add_feed(input_name);
    if (IsGPUFeedAndFetchSupported(input_types[i])) {
      opts.mutable_feed_devices()->insert({input_name, device_name});
    }
  }

  for (size_t i = 0; i < num_outputs; ++i) {
    const std::string output_name = output_names[i];
    opts.add_fetch(output_name);
    if (IsGPUFeedAndFetchSupported(output_types[i])) {
      opts.mutable_fetch_devices()->insert({output_name, device_name});
    }
  }

  // CallableOptions.fetch_skip_sync = false is not yet implemented, we
  // will have to synchronize after the callable is run.
  opts.set_fetch_skip_sync(true);

  return m->MakeCallable(opts);
}

TRTISTF_Error*
TRTISTF_ModelRun(
    TRTISTF_Model* model, TRTISTF_TensorList* input_tensors, size_t num_outputs,
    const char** output_names, TRTISTF_TensorList** output_tensors)
{
  ModelImpl* m = reinterpret_cast<ModelImpl*>(model);

  std::vector<std::string> output_tensor_names;
  for (size_t i = 0; i < num_outputs; ++i) {
    output_tensor_names.emplace_back(output_names[i]);
  }

  return m->Run(input_tensors, output_tensor_names, output_tensors);
}
