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

#include "libtorch/core/torch_backend_c2.h"

#include <google/protobuf/io/coded_stream.h>
#include <stdint.h>
#include "libtorch/core/context_gpu.h"

#include <ATen/DLConvertor.h>
#include <ATen/Functions.h>

namespace nvidia { namespace inferenceserver {

class LibTorchWorkspaceImpl : public LibTorchWorkspace {
 public:
  static Error Create(
      LibTorchWorkspaceImpl** ltws, const std::string& model_name,
      const int max_batch_size, const std::vector<std::string>& input_names,
      const std::vector<std::string>& output_names,
      const std::string torch_model_path);
  LibTorchWorkspaceImpl() = default;
  ~LibTorchWorkspaceImpl() = default;

  const std::set<std::string>& PotentialInputNames() const override
  {
    return potential_input_names_;
  }
  const std::set<std::string>& PotentialOutputNames() const override
  {
    return potential_output_names_;
  }

  Error SetInputTensor(
      const std::string& name, const std::vector<int64_t>& shape,
      const DataType dtype, const char* content, size_t byte_size) override;
  Error GetOutputTensor(
      const std::string& name, const LibTorchWorkspace::DataType dtype,
      const char** content, size_t* byte_size,
      std::vector<int64_t>* content_shape) override;
  Error Run() override;

 private:

  // The name of the model in the model store. This is not necessarily
  // the name in the Torch ScriptModule.
  std::string model_name_;

  // Maximum batch size to allow. NO_BATCHING indicates that
  // batching is not supported.
  int max_batch_size_;

  // The name of the model in the Torch Pt. This does not
  // necessarily match the model-store name of the model.
  // std::string torch_model_name_;

  // Names of all possible inputs and outputs for the model. These are
  // names reported by the model .pt itself as external inputs and
  // outputs.
  std::set<std::string> potential_input_names_;
  std::set<std::string> potential_output_names_;
};

namespace {

const std::string
DimsDebugString(const at::IntArrayRef& dims)
{
  bool first = true;
  std::string str;
  str.append("[");
  for (size_t i = 0; i < dims.size(); ++i) {
    if (!first) {
      str.append(",");
    }
    str.append(std::to_string(dims[i]));
    first = false;
  }
  str.append("]");
  return str;
}

bool
ReadBinaryProto(
    const std::vector<char>& blob, google::protobuf::MessageLite* msg)
{
  google::protobuf::io::CodedInputStream coded_stream(
      reinterpret_cast<const uint8_t*>(&blob[0]), blob.size());
  coded_stream.SetTotalBytesLimit(INT_MAX, INT_MAX);
  return msg->ParseFromCodedStream(&coded_stream);
}

std::pair<bool, const DLDataType>
ConvertTorchToDLDataType(const at::ScalarType& type)
{
  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = type.elementSizeInBytes() * 8;
  switch (type) {
    case at::ScalarType::Byte:
      dtype.code = DLDataTypeCode::kDLUInt;
      break;
    case at::ScalarType::Char:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case at::ScalarType::Double:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case at::ScalarType::Float:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case at::ScalarType::Int:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case at::ScalarType::Long:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case at::ScalarType::Short:
      dtype.code = DLDataTypeCode::kDLInt;
      break;
    case at::ScalarType::Half:
      dtype.code = DLDataTypeCode::kDLFloat;
      break;
    case at::ScalarType::Bool:
      dtype.code = DLDataTypeCode::kDLUInt
    default:
        return std::make_pair(false, dtype);
  }

  return std::make_pair(true, dtype);
}

std::pair<bool, const torch::ScalarType>
ConvertDataTypeToTorchType(const DataType& dtype)
{
  torch::ScalarType type;
  switch (data_type) {
    case TYPE_UINT8:
      type = torch::kByte;
    case TYPE_INT8:
      type = torch::kChar;
    case TYPE_INT16:
      type = torch::kShort;
    case TYPE_INT32:
      type = torch::kInt;
    case TYPE_INT64:
      type = torch::kLong;
    case TYPE_FP16:
      type = torch::kHalf;
    case TYPE_FP32:
      type = torch::kFloat;
    case TYPE_FP64:
      type = torch::kDouble;
    case TYPE_UINT16:
    case TYPE_UINT32:
    case TYPE_UINT64:
    case TYPE_STRING:
    default:
        return std::make_pair(false, type);
  }

  return std::make_pair(true, type);
}

const std::string
DataTypeName(const DLDataType datatype)
{
  switch (datatype.code) {
    case DLDataTypeCode::Invalid;
      return "INVALID";
    case DLDataTypeCode::kDLUInt;
      return "UINT";
    case DLDataTypeCode::kDLInt;
      return "INT";
    case DLDataTypeCode::kDLFloat;
      return "FLOAT";
  }

  return "<unknown>";
}

}  // namespace


LibTorchWorkspace::Error
LibTorchWorkspaceCreate(
    LibTorchWorkspace** ltws, const std::string& model_name,
    const int max_batch_size, const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names, const int gpu_device,
    const std::string torch_model_path)
{
  // Set the device for this model. It seems necessary to set the
  // device not only on the network but also on each individual
  // operator.
  // Declare global torch::Device
  if (gpu_device == LibTorchWorkspace::NO_GPU_DEVICE) {
    device = torch::Device(torch::kCPU);
  } else {
    device = torch::Device(torch::kCUDA, device_option.device_id)
  }
  try {
    std::shared_ptr<torch::jit::script::Module> torch_model = torch::jit::load(torch_model_path, device);
  }
  catch{
    return LibTorchWorkspace::Error("failed to load LibTorch model");
  }

  LibTorchWorkspaceImpl* ltwsimpl;
  LibTorchWorkspace::Error err = LibTorchWorkspaceImpl::Create(
      &ltwsimpl, model_name, max_batch_size, input_names, output_names,
      torch_model, device);
  *ltws = ltwsimpl;
  return err;
}

LibTorchWorkspace::Error
LibTorchWorkspaceImpl::Create(
    LibTorchWorkspaceImpl** ltws, const std::string& model_name,
    const int max_batch_size, const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    std::shared_ptr<torch::jit::script::Module>& torch_model,
    torch::Device device)
{
  *ltws = new LibTorchWorkspaceImpl();
  (*ltws)->model_name_ = model_name;
  (*ltws)->max_batch_size_ = max_batch_size;
  (*ltws)->torch_model_ = torch_model;
  (*ltws)->device_ = device;

  return Error();
}

LibTorchWorkspace::Error
LibTorchWorkspaceImpl::SetInputTensor(
    const std::string& name, const std::vector<int64_t>& shape,
    const DataType dtype, const char* content,
    size_t byte_size)
{
  const auto pr = ConvertDataTypeToTorchType(dtype);
  if (!pr.first) {
    return Error(
        "Failed to convert DataType '" + DataTypeName(dtype) +
        "' to Torch datatype");
  }

  torch::Tensor input_tensor = torch::from_blob(content, shape, pr.second.code, device_);

  if ((input_tensor.numel() * pr.second.bits / 8) != byte_size) {
    return Error(
        "unexpected size " + std::to_string(byte_size) +
        " for inference input '" + name + "', expecting " +
        std::to_string(input_tensor.nbytes()));
  }
  inputs_.push_back(input_tensor);
  return Error();
}

LibTorchWorkspace::Error
LibTorchWorkspaceImpl::GetOutputTensor(
    const std::string& name, const DataType dtype,
    const char** content, size_t* byte_size,
    std::vector<int64_t>* content_shape)
{
  // Initialize char* content[output_flat.nbytes()];
  torch::DeviceType output_device = torch::kCPU;
  try{
    outputs_ = outputs_.to(output_device)
    torch::Tensor output_flat = outputs_.flatten();
    std::vector<float> outputs_vector;
    for(int i=0;i<output_flat.sizes()[0];i++){
      outputs_vector.push_back(output_flat[i].item().to<float>());
    }
    // Copy output into buffer
    memcpy(*content, static_cast<const char*>&outputs_vector[0], output_flat.nbytes());
    //  Set content shape
    auto shape = outputs_.sizes();
    for (auto itr = shape.begin(); itr != shape.end(); itr++){
      content_shape.push_back(*itr);
    }
  }
  catch {
    return Error("failed to get LibTorch output");
  }

  return Error();
}

LibTorchWorkspace::Error
LibTorchWorkspaceImpl::Run()
{
  try {
      outputs_ = torch_model_->forward(inputs_).toTensor(); // toTuple() for two outputs
  }
  catch (exception& ex) {
    return Error("failed to run model '" + model_name_ + "': " + ex.what());
  }

  return Error();
}

}}  // namespace nvidia::inferenceserver
