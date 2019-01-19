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

#include "caffe2/core/netdef_bundle_c2.h"

#include <google/protobuf/io/coded_stream.h>
#include <stdint.h>
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/init.h"
#include "caffe2/core/types.h"
#include "caffe2/core/workspace.h"

namespace nvidia { namespace inferenceserver {

class Caffe2WorkspaceImpl : public Caffe2Workspace {
 public:
  static Error Create(
      Caffe2WorkspaceImpl** c2ws, const std::string& model_name,
      const int max_batch_size, const std::vector<std::string>& input_names,
      const std::vector<std::string>& output_names,
      const caffe2::NetDef& netdef_init, const caffe2::NetDef& netdef_model);
  Caffe2WorkspaceImpl() = default;
  ~Caffe2WorkspaceImpl() = default;

  const std::set<std::string>& PotentialInputNames() const override
  {
    return potential_input_names_;
  }
  const std::set<std::string>& PotentialOutputNames() const override
  {
    return potential_output_names_;
  }

  const std::unordered_map<std::string, size_t>& Outputs() const override
  {
    return outputs_;
  }

  Error AddOutputTensor(
      const std::string& name, const DataType datatype,
      const std::vector<int>& dims) override;
  Error SetInputTensor(
      const std::string& name, const std::vector<int64_t>& shape,
      const DataType dtype, const char* content, size_t byte_size) override;
  Error GetOutputTensor(
      const std::string& name, size_t batch_size, const char** content,
      size_t byte_size) override;
  Error Run() override;

 private:
  using IOTensorMap =
      std::unordered_map<std::string, std::unique_ptr<caffe2::Tensor>>;

  // The Caffe2 workspace.
  std::unique_ptr<caffe2::Workspace> ws_;

  // The name of the model in the model store. This is not necessarily
  // the name in the Caffe2 NetDef protobuf.
  std::string model_name_;

  // Maximum batch size to allow. NO_BATCHING indicates that
  // batching is not supported.
  int max_batch_size_;

  // The name of the model in the Caffe2 NetDef. This does not
  // necessarily match the model-store name of the model.
  std::string netdef_model_name_;

  // Names of all possible inputs and outputs for the model. These are
  // names reported by the model netdef itself as external inputs and
  // outputs.
  std::set<std::string> potential_input_names_;
  std::set<std::string> potential_output_names_;

  // The outputs of the model specified by the model configuration and
  // the size of each.
  std::unordered_map<std::string, size_t> outputs_;

  // Map from output name to caffe2 tensor holding shape and type for
  // that output. We can use TensorCPU for this even when using the
  // GPU since we are only interested in the shape and type of these
  // tensors.
  IOTensorMap output_tensor_map_;
};

namespace {

const std::string
DimsDebugString(const at::IntList& dims)
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

std::pair<bool, const caffe2::TypeMeta>
ConvertDatatype(Caffe2Workspace::DataType dtype)
{
  caffe2::TensorProto::DataType ctype;

  switch (dtype) {
    case Caffe2Workspace::DataType::TYPE_BOOL:
      ctype = caffe2::TensorProto_DataType_BOOL;
      break;
    case Caffe2Workspace::DataType::TYPE_UINT8:
      ctype = caffe2::TensorProto_DataType_UINT8;
      break;
    case Caffe2Workspace::DataType::TYPE_UINT16:
      ctype = caffe2::TensorProto_DataType_UINT16;
      break;
    case Caffe2Workspace::DataType::TYPE_INT8:
      ctype = caffe2::TensorProto_DataType_INT8;
      break;
    case Caffe2Workspace::DataType::TYPE_INT16:
      ctype = caffe2::TensorProto_DataType_INT16;
      break;
    case Caffe2Workspace::DataType::TYPE_INT32:
      ctype = caffe2::TensorProto_DataType_INT32;
      break;
    case Caffe2Workspace::DataType::TYPE_INT64:
      ctype = caffe2::TensorProto_DataType_INT64;
      break;
    case Caffe2Workspace::DataType::TYPE_FP16:
      ctype = caffe2::TensorProto_DataType_FLOAT16;
      break;
    case Caffe2Workspace::DataType::TYPE_FP32:
      ctype = caffe2::TensorProto_DataType_FLOAT;
      break;
    case Caffe2Workspace::DataType::TYPE_FP64:
      ctype = caffe2::TensorProto_DataType_DOUBLE;
      break;
    case Caffe2Workspace::DataType::TYPE_STRING:
      ctype = caffe2::TensorProto_DataType_STRING;
      break;
    default:
      return std::make_pair(false, caffe2::TypeMeta());
  }

  return std::make_pair(true, caffe2::DataTypeToTypeMeta(ctype));
}

const std::string
DataTypeName(const Caffe2Workspace::DataType datatype)
{
  switch (datatype) {
    case Caffe2Workspace::DataType::TYPE_INVALID:
      return "INVALID";
    case Caffe2Workspace::DataType::TYPE_BOOL:
      return "BOOL";
    case Caffe2Workspace::DataType::TYPE_UINT8:
      return "UINT8";
    case Caffe2Workspace::DataType::TYPE_UINT16:
      return "UINT16";
    case Caffe2Workspace::DataType::TYPE_UINT32:
      return "UINT32";
    case Caffe2Workspace::DataType::TYPE_UINT64:
      return "UINT64";
    case Caffe2Workspace::DataType::TYPE_INT8:
      return "INT8";
    case Caffe2Workspace::DataType::TYPE_INT16:
      return "INT16";
    case Caffe2Workspace::DataType::TYPE_INT32:
      return "INT32";
    case Caffe2Workspace::DataType::TYPE_INT64:
      return "INT64";
    case Caffe2Workspace::DataType::TYPE_FP16:
      return "FP16";
    case Caffe2Workspace::DataType::TYPE_FP32:
      return "FP32";
    case Caffe2Workspace::DataType::TYPE_FP64:
      return "FP64";
    case Caffe2Workspace::DataType::TYPE_STRING:
      return "STRING";
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


Caffe2Workspace::Error
Caffe2WorkspaceCreate(
    Caffe2Workspace** c2ws, const std::string& model_name,
    const int max_batch_size, const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names, const int gpu_device,
    const std::vector<char>& init_blob, const std::vector<char>& model_blob)
{
  caffe2::GlobalInit();

  // We must construct a caffe2::CUDAContext to get the side-effect of
  // initializing Caffe2 to use CUDA. It is ok to call this multiple
  // times.
  new caffe2::CUDAContext(0);

  caffe2::NetDef netdef_init, netdef_model;
  if (!ReadBinaryProto(init_blob, &netdef_init) ||
      !ReadBinaryProto(model_blob, &netdef_model)) {
    return Caffe2Workspace::Error("failed to parse NetDef model");
  }

  // Set the device for this model. It seems necessary to set the
  // device not only on the network but also on each individual
  // operator.
  caffe2::DeviceOption device_option;
  if (gpu_device == Caffe2Workspace::NO_GPU_DEVICE) {
    device_option.set_device_type(static_cast<int>(caffe2::CPU));
  } else {
    device_option.set_device_type(static_cast<int>(caffe2::CUDA));
    device_option.set_device_id(gpu_device);
  }

  netdef_init.mutable_device_option()->CopyFrom(device_option);
  netdef_model.mutable_device_option()->CopyFrom(device_option);

  for (int i = 0; i < netdef_model.op().size(); ++i) {
    netdef_model.mutable_op(i)->mutable_device_option()->CopyFrom(
        device_option);
  }

  // For each input that feeds an operator that is executed on a GPU,
  // add an operation that copies that input to the GPU. For each
  // output that is produced on the GPU add an operation that copies
  // that output to the CPU.
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
  if (new_input_ops.op().size() > 0) {
    new_input_ops.mutable_op()->MergeFrom(netdef_model.op());
    netdef_model.mutable_op()->CopyFrom(new_input_ops.op());
  }

  Caffe2WorkspaceImpl* c2wsimpl;
  Caffe2Workspace::Error err = Caffe2WorkspaceImpl::Create(
      &c2wsimpl, model_name, max_batch_size, input_names, output_names,
      netdef_init, netdef_model);
  *c2ws = c2wsimpl;
  return err;
}

Caffe2Workspace::Error
Caffe2WorkspaceImpl::Create(
    Caffe2WorkspaceImpl** c2ws, const std::string& model_name,
    const int max_batch_size, const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const caffe2::NetDef& netdef_init, const caffe2::NetDef& netdef_model)
{
  *c2ws = new Caffe2WorkspaceImpl();
  (*c2ws)->model_name_ = model_name;
  (*c2ws)->max_batch_size_ = max_batch_size;
  (*c2ws)->netdef_model_name_ = netdef_model.name();
  (*c2ws)->ws_.reset(new caffe2::Workspace("/tmp"));
  if ((*c2ws)->ws_ == nullptr) {
    delete *c2ws;
    *c2ws = nullptr;
    return Error(
        "Failed to create Caffe2 workspace for model '" + model_name + "'");
  }

  if (!(*c2ws)->ws_->RunNetOnce(netdef_init)) {
    delete *c2ws;
    *c2ws = nullptr;
    return Error(
        "Failed to run Caffe2 init workspace for model '" + model_name + "'");
  }

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
          "Failed to create Caffe2 blob for input '" + input_name +
          "' for model '" + model_name + "': " + ex.msg());
    }
    if (input == nullptr) {
      delete *c2ws;
      *c2ws = nullptr;
      return Error(
          "Failed to create Caffe2 blob for input '" + input_name +
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
        "Failed to create Caffe2 model for model '" + model_name + "'");
  }

  return Error();
}

Caffe2Workspace::Error
Caffe2WorkspaceImpl::AddOutputTensor(
    const std::string& name, const DataType datatype,
    const std::vector<int>& dims)
{
  // Create a Tensor to hold the shape and datatype.
  const auto pr = ConvertDatatype(datatype);
  if (!pr.first) {
    return Error(
        "Failed to convert datatype '" + DataTypeName(datatype) +
        "' to Caffe2 NetDef datatype");
  }

  // Tensor::ShareExternalPointer allows us to explicitly set the
  // tensor's type.
  std::unique_ptr<caffe2::Tensor> tensor(new caffe2::Tensor(dims, caffe2::CPU));
  tensor->ShareExternalPointer(nullptr, pr.second);

  outputs_.insert(std::make_pair(name, tensor->size() * tensor->itemsize()));

  output_tensor_map_.emplace(
      std::piecewise_construct, std::make_tuple(name),
      std::make_tuple(std::move(tensor)));

  return Error();
}

Caffe2Workspace::Error
Caffe2WorkspaceImpl::SetInputTensor(
    const std::string& name, const std::vector<int64_t>& shape,
    const Caffe2Workspace::DataType dtype, const char* content,
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
        "failed to get NetDef blob for input '" + name + "': " + ex.msg());
  }
  if (blob == nullptr) {
    return Error("failed to get NetDef blob for input '" + name + "'");
  }

  caffe2::Tensor* input = nullptr;
  try {
    input = BlobGetMutableTensor(blob, caffe2::CPU);
  }
  catch (caffe2::EnforceNotMet ex) {
    return Error(
        "failed to get NetDef tensor for input '" + name + "': " + ex.msg());
  }
  if (input == nullptr) {
    return Error("failed to get NetDef tensor for input '" + name + "'");
  }

  input->Resize(shape);

  const auto pr = ConvertDatatype(dtype);
  if (!pr.first) {
    return Error(
        "Failed to convert datatype '" + DataTypeName(dtype) +
        "' to Caffe2 NetDef datatype");
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

Caffe2Workspace::Error
Caffe2WorkspaceImpl::GetOutputTensor(
    const std::string& name, size_t batch_size, const char** content,
    size_t byte_size)
{
  const auto itr = output_tensor_map_.find(name);
  if (itr == output_tensor_map_.end()) {
    return Error("unexpected inference output '" + name + "'");
  }

  // Find the output tensor in the model...
  caffe2::Blob* blob = nullptr;
  try {
    blob = ws_->GetBlob(name);
  }
  catch (caffe2::EnforceNotMet ex) {
    return Error(
        "failed to get NetDef blob for output '" + name + "': " + ex.msg());
  }
  if (blob == nullptr) {
    return Error("failed to get NetDef blob for output '" + name + "'");
  }

  caffe2::Tensor* output = nullptr;
  try {
    output = BlobGetMutableTensor(blob, caffe2::CPU);
  }
  catch (caffe2::EnforceNotMet ex) {
    return Error(
        "failed to get NetDef tensor for output '" + name + "': " + ex.msg());
  }
  if (output == nullptr) {
    return Error("failed to get NetDef tensor for output '" + name + "'");
  }

  if (itr->second->meta() != output->meta()) {
    return Error(
        "unexpected datatype " + std::string(output->meta().name()) +
        " for inference output '" + name + "', expecting " +
        std::string(itr->second->meta().name()));
  }

  // If model supports batching then prepend the batch dimension onto
  // the output shape.
  std::vector<long int> expected_dims;
  if (max_batch_size_ != NO_BATCHING) {
    expected_dims.push_back(batch_size);
  }
  for (const auto d : itr->second->dims()) {
    expected_dims.push_back(d);
  }

  if (expected_dims != output->dims()) {
    return Error(
        "unexpected shape " + DimsDebugString(output->dims()) +
        " for inference output '" + name + "', expecting " +
        DimsDebugString(expected_dims));
  }

  if (byte_size != output->nbytes()) {
    return Error(
        "unexpected size " + std::to_string(output->nbytes()) +
        " for inference output '" + name + "', expecting " +
        std::to_string(byte_size));
  }

  *content = static_cast<const char*>(output->raw_data());

  return Error();
}

Caffe2Workspace::Error
Caffe2WorkspaceImpl::Run()
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
