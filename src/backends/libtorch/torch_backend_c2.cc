// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

namespace nvidia { namespace inferenceserver {

class LibTorchWorkspaceImpl : public LibTorchWorkspace {
 public:
  static Error Create(
      LibTorchWorkspaceImpl** c2ws, const std::string& model_name,
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
  // The Torch workspace.
  // std::unique_ptr<caffe2::Workspace> ws_;

  // The name of the model in the model store. This is not necessarily
  // the name in the Torch ScriptModule.
  std::string model_name_;

  // Maximum batch size to allow. NO_BATCHING indicates that
  // batching is not supported.
  int max_batch_size_;

  // The name of the model in the Torch NetDef. This does not
  // necessarily match the model-store name of the model.
  // std::string torch_model_name_;

  // Names of all possible inputs and outputs for the model. These are
  // names reported by the model netdef itself as external inputs and
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

std::pair<bool, const DataType>
ConvertDatatype(const at::Type& type)
{
  DataType dtype;
  dtype.lanes = 1;
  dtype.bits = type.elementSizeInBytes() * 8;
  switch (type.scalarType()) {
    case at::ScalarType::Byte:
      dtype.code = LibTorchWorkspace::DataTypeCode::kDLUInt;
      break;
    case at::ScalarType::Char:
      dtype.code = LibTorchWorkspace::DataTypeCode::kDLInt;
      break;
    case at::ScalarType::Double:
      dtype.code = LibTorchWorkspace::DataTypeCode::kDLFloat;
      break;
    case at::ScalarType::Float:
      dtype.code = LibTorchWorkspace::DataTypeCode::kDLFloat;
      break;
    case at::ScalarType::Int:
      dtype.code = LibTorchWorkspace::DataTypeCode::kDLInt;
      break;
    case at::ScalarType::Long:
      dtype.code = LibTorchWorkspace::DataTypeCode::kDLInt;
      break;
    case at::ScalarType::Short:
      dtype.code = LibTorchWorkspace::DataTypeCode::kDLInt;
      break;
    case at::ScalarType::Half:
      dtype.code = LibTorchWorkspace::DataTypeCode::kDLFloat;
      break;
    default:
      return std::make_pair(false, dtype);
  }

  return std::make_pair(true, dtype);
}

const std::string
DataTypeName(const LibTorchWorkspace::DataType datatype)
{
  switch (datatype.code) {
    case LibTorchWorkspace::DataTypeCode::Invalid;
      return "INVALID";
    case LibTorchWorkspace::DataTypeCode::kDLUInt;
      return "UINT";
    case LibTorchWorkspace::DataTypeCode::kDLInt;
      return "INT";
    case LibTorchWorkspace::DataTypeCode::kDLFloat;
      return "FLOAT";
  }

  return "<unknown>";
}

caffe2::OperatorDef*
AddOp(
    caffe2::NetDef* net, const std::string& name,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs)
{
  auto op = net->add_op();
  op->set_type(name);
  for (auto input : inputs) {
    op->add_input(input);
  }
  for (auto output : outputs) {
    op->add_output(output);
  }

  return op;
}

caffe2::OperatorDef*
AddCopyFromCpuInput(
    caffe2::NetDef* net, const std::string& input, const std::string& output)
{
  return AddOp(net, "CopyFromCPUInput", {input}, {output});
}

caffe2::OperatorDef*
AddEnsureCpuOutput(
    caffe2::NetDef* net, const std::string& input, const std::string& output)
{
  return AddOp(net, "EnsureCPUOutput", {input}, {output});
}

}  // namespace


LibTorchWorkspace::Error
LibTorchWorkspaceCreate(
    LibTorchWorkspace** c2ws, const std::string& model_name,
    const int max_batch_size, const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names, const int gpu_device,
    const std::string torch_model_path)
{
  new caffe2::CUDAContext(0);

  try {
    torch_model_ = torch::jit::load(torch_model_path);
  }
  catch{
  // (!ReadBinaryProto(model_blob, &torch_model_)) {
    return LibTorchWorkspace::Error("failed to load LibTorch model");
  }

  // Set the device for this model. It seems necessary to set the
  // device not only on the network but also on each individual
  // operator.
  DeviceContext device_option;
  if (gpu_device == LibTorchWorkspace::NO_GPU_DEVICE) {
    device_option.device_type = static_cast<int>(LibTorchWorkspace::DeviceType::kDLCPU);
  } else {
    device_option.device_type = static_cast<int>(LibTorchWorkspace::DeviceType::kDLGPU);
    device_option.device_id = gpu_device;
    torch_model_->to(torch::Device(torch::kCUDA, device_option.device_id));
  }

  // For each input that feeds an operator that is executed on a GPU,
  // add an operation that copies that input to the GPU. For each
  // output that is produced on the GPU add an operation that copies
  // that output to the CPU.
  // TODO caffe to LibTorch
  std::unordered_map<std::string, std::string> io_name_map;
  caffe2::NetDef new_input_ops;

  // We don't want to revisit newly added operations, so get the
  // current size before starting the iteration.
  const int op_cnt = netdef_model.op().size();
  for (int opidx = 0; opidx < op_cnt; ++opidx) {
    caffe2::OperatorDef* opdef = netdef_model.mutable_op(opidx);
    if (opdef->device_option().device_type() !=
        static_cast<int>(caffe2::CUDA)) {
      continue;
    }

    // Inputs...
    for (const auto& input_name : input_names) {
      auto itr = io_name_map.find(input_name);

      for (int iidx = 0; iidx < opdef->input().size(); ++iidx) {
        if (opdef->input(iidx) == input_name) {
          if (itr == io_name_map.end()) {
            const std::string gpu_name = input_name + "_in_nvis_";
            caffe2::OperatorDef* cpdef =
                AddCopyFromCpuInput(&new_input_ops, input_name, gpu_name);
            cpdef->mutable_device_option()->CopyFrom(device_option);
            auto pr = io_name_map.emplace(input_name, gpu_name);
            itr = pr.first;
          }

          opdef->set_input(iidx, itr->second);
        }
      }
    }

    // Outputs...
    for (const auto& output_name : output_names) {
      auto itr = io_name_map.find(output_name);

      for (int oidx = 0; oidx < opdef->output().size(); ++oidx) {
        if (opdef->output(oidx) == output_name) {
          if (itr == io_name_map.end()) {
            const std::string gpu_name = output_name + "_out_nvis_";
            caffe2::OperatorDef* cpdef =
                AddEnsureCpuOutput(&netdef_model, gpu_name, output_name);
            cpdef->mutable_device_option()->CopyFrom(device_option);
            auto pr = io_name_map.emplace(output_name, gpu_name);
            itr = pr.first;
          }

          opdef->set_output(oidx, itr->second);
        }
      }
    }
  }

  // If we added new ops for the inputs they need to be added to the
  // beginning of the ops. NetDef seems to require the tensor be
  // defined before references... not sure how it handles
  // cycles. Protobuf doesn't have any way to do this gracefully...
  // TODO caffe to LibTorch
  if (new_input_ops.op().size() > 0) {
    new_input_ops.mutable_op()->MergeFrom(netdef_model.op());
    netdef_model.mutable_op()->CopyFrom(new_input_ops.op());
  }

  LibTorchWorkspaceImpl* c2wsimpl;
  LibTorchWorkspace::Error err = LibTorchWorkspaceImpl::Create(
      &c2wsimpl, model_name, max_batch_size, input_names, output_names,
      libtorch_model);
  *c2ws = c2wsimpl;
  return err;
}

LibTorchWorkspace::Error
LibTorchWorkspaceImpl::Create(
    LibTorchWorkspaceImpl** c2ws, const std::string& model_name,
    const int max_batch_size, const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string torch_model_path)
{
  *c2ws = new LibTorchWorkspaceImpl();
  (*c2ws)->model_name_ = model_name;
  (*c2ws)->max_batch_size_ = max_batch_size;
  (*c2ws)->ws_.reset(new caffe2::Workspace("/tmp"));
  if ((*c2ws)->ws_ == nullptr) {
    delete *c2ws;
    *c2ws = nullptr;
    return Error(
        "Failed to create Torch workspace for model '" + model_name + "'");
  }

  // if (!(*c2ws)->ws_->RunNetOnce(netdef_init)) {
  //   delete *c2ws;
  //   *c2ws = nullptr;
  //   return Error(
  //       "Failed to run Torch init workspace for model '" + model_name + "'");
  // }

  // Create the blobs for each input
  for (const auto& input_name : input_names) {
    caffe2::Blob* input = nullptr;
    try {
      input = (*c2ws)->ws_->CreateBlob(input_name);
    }
    catch (caffe2::EnforceNotMet ex) {
      delete *c2ws;
      *c2ws = nullptr;
      return Error(
          "Failed to create Torch blob for input '" + input_name +
          "' for model '" + model_name + "': " + ex.msg());
    }
    if (input == nullptr) {
      delete *c2ws;
      *c2ws = nullptr;
      return Error(
          "Failed to create Torch blob for input '" + input_name +
          "' for model '" + model_name + "'");
    }
  }

  // Collect allowed inputs and outputs...
  for (const auto& i : netdef_model.external_input()) {
    (*c2ws)->potential_input_names_.insert(i);
  }
  for (const auto& o : netdef_model.external_output()) {
    (*c2ws)->potential_output_names_.insert(o);
  }

  if ((*c2ws)->ws_->CreateNet(netdef_model) == nullptr) {
    delete *c2ws;
    *c2ws = nullptr;
    return Error(
        "Failed to create Torch model for model '" + model_name + "'");
  }

  return Error();
}

LibTorchWorkspace::Error
LibTorchWorkspaceImpl::SetInputTensor(
    const std::string& name, const std::vector<int64_t>& shape,
    const LibTorchWorkspace::DataType dtype, const char* content,
    size_t byte_size)
{
  // Find the input tensor in the model and set it to use 'content'
  // in-place.
  caffe2::Blob* blob = nullptr;
  try {
    blob = ws_->GetBlob(name);
  }
  catch (caffe2::EnforceNotMet ex) {
    return Error(
        "failed to get LibTorch blob for input '" + name + "': " + ex.msg());
  }
  if (blob == nullptr) {
    return Error("failed to get LibTorch blob for input '" + name + "'");
  }

  caffe2::Tensor* input = nullptr;
  try {
    input = BlobGetMutableTensor(blob, caffe2::CPU);
  }
  catch (caffe2::EnforceNotMet ex) {
    return Error(
        "failed to get LibTorch tensor for input '" + name + "': " + ex.msg());
  }
  if (input == nullptr) {
    return Error("failed to get LibTorch tensor for input '" + name + "'");
  }

  input->Resize(shape);

  const auto pr = ConvertDatatype(dtype);
  if (!pr.first) {
    return Error(
        "Failed to convert datatype '" + DataTypeName(dtype) +
        "' to Torch LibTorch datatype");
  }

  input->ShareExternalPointer(const_cast<char*>(content), pr.second, byte_size);

  if ((input->size() * input->itemsize()) != byte_size) {
    return Error(
        "unexpected size " + std::to_string(byte_size) +
        " for inference input '" + name + "', expecting " +
        std::to_string(input->size() * input->itemsize()));
  }

  return Error();
}

LibTorchWorkspace::Error
LibTorchWorkspaceImpl::GetOutputTensor(
    const std::string& name, const LibTorchWorkspace::DataType dtype,
    const char** content, size_t* byte_size,
    std::vector<int64_t>* content_shape)
{
  // Find the output tensor in the model...
  caffe2::Blob* blob = nullptr;
  try {
    blob = ws_->GetBlob(name);
  }
  catch (caffe2::EnforceNotMet ex) {
    return Error(
        "failed to get LibTorch blob for output '" + name + "': " + ex.msg());
  }
  if (blob == nullptr) {
    return Error("failed to get LibTorch blob for output '" + name + "'");
  }

  caffe2::Tensor* output = nullptr;
  try {
    output = BlobGetMutableTensor(blob, caffe2::CPU);
  }
  catch (caffe2::EnforceNotMet ex) {
    return Error(
        "failed to get LibTorch tensor for output '" + name + "': " + ex.msg());
  }
  if (output == nullptr) {
    return Error("failed to get LibTorch tensor for output '" + name + "'");
  }

  const auto pr = ConvertDatatype(dtype);
  if (!pr.first) {
    return Error(
        "Failed to convert datatype '" + DataTypeName(dtype) +
        "' to Torch LibTorch datatype");
  }

  if (pr.second != output->meta()) {
    return Error(
        "unexpected datatype " + std::string(output->meta().name()) +
        " for inference output '" + name + "', expecting " +
        std::string(pr.second.name()));
  }

  content_shape->clear();
  for (auto d : output->sizes()) {
    content_shape->push_back(d);
  }

  *byte_size = output->nbytes();
  *content = static_cast<const char*>(output->raw_data());

  return Error();
}

LibTorchWorkspace::Error
LibTorchWorkspaceImpl::Run()
{
  try {
    if (!ws_->RunNet(netdef_model_name_)) {
      return Error("failed to run model '" + model_name_ + "'");
    }
  }
  catch (caffe2::EnforceNotMet ex) {
    return Error("failed to run model '" + model_name_ + "': " + ex.msg());
  }

  return Error();
}

}}  // namespace nvidia::inferenceserver
