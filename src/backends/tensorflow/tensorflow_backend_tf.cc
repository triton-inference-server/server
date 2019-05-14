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

#include "tensorflow/tensorflow_backend_tf.h"

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

// If TensorFlow status is non-OK, return the equivalent TFWorkspaceError
#define RETURN_IF_TF_ERROR(TFS)                            \
  do {                                                     \
    const tensorflow::Status& status__ = (TFS);            \
    if (status__.code() != 0) {                            \
      return TFWorkspace::Error(status__.error_message()); \
    }                                                      \
  } while (false)

namespace nvidia { namespace inferenceserver {

namespace {

TFWorkspace::DataType
ConvertDataType(tensorflow::DataType dtype)
{
  switch (dtype) {
    case tensorflow::DT_INVALID:
      return TFWorkspace::DataType::TYPE_INVALID;
    case tensorflow::DT_BOOL:
      return TFWorkspace::DataType::TYPE_BOOL;
    case tensorflow::DT_UINT8:
      return TFWorkspace::DataType::TYPE_UINT8;
    case tensorflow::DT_UINT16:
      return TFWorkspace::DataType::TYPE_UINT16;
    case tensorflow::DT_UINT32:
      return TFWorkspace::DataType::TYPE_UINT32;
    case tensorflow::DT_UINT64:
      return TFWorkspace::DataType::TYPE_UINT64;
    case tensorflow::DT_INT8:
      return TFWorkspace::DataType::TYPE_INT8;
    case tensorflow::DT_INT16:
      return TFWorkspace::DataType::TYPE_INT16;
    case tensorflow::DT_INT32:
      return TFWorkspace::DataType::TYPE_INT32;
    case tensorflow::DT_INT64:
      return TFWorkspace::DataType::TYPE_INT64;
    case tensorflow::DT_HALF:
      return TFWorkspace::DataType::TYPE_FP16;
    case tensorflow::DT_FLOAT:
      return TFWorkspace::DataType::TYPE_FP32;
    case tensorflow::DT_DOUBLE:
      return TFWorkspace::DataType::TYPE_FP64;
    case tensorflow::DT_STRING:
      return TFWorkspace::DataType::TYPE_STRING;
    default:
      break;
  }

  return TFWorkspace::DataType::TYPE_INVALID;
}

tensorflow::DataType
ConvertDataType(TFWorkspace::DataType dtype)
{
  switch (dtype) {
    case TFWorkspace::DataType::TYPE_INVALID:
      return tensorflow::DT_INVALID;
    case TFWorkspace::DataType::TYPE_BOOL:
      return tensorflow::DT_BOOL;
    case TFWorkspace::DataType::TYPE_UINT8:
      return tensorflow::DT_UINT8;
    case TFWorkspace::DataType::TYPE_UINT16:
      return tensorflow::DT_UINT16;
    case TFWorkspace::DataType::TYPE_UINT32:
      return tensorflow::DT_UINT32;
    case TFWorkspace::DataType::TYPE_UINT64:
      return tensorflow::DT_UINT64;
    case TFWorkspace::DataType::TYPE_INT8:
      return tensorflow::DT_INT8;
    case TFWorkspace::DataType::TYPE_INT16:
      return tensorflow::DT_INT16;
    case TFWorkspace::DataType::TYPE_INT32:
      return tensorflow::DT_INT32;
    case TFWorkspace::DataType::TYPE_INT64:
      return tensorflow::DT_INT64;
    case TFWorkspace::DataType::TYPE_FP16:
      return tensorflow::DT_HALF;
    case TFWorkspace::DataType::TYPE_FP32:
      return tensorflow::DT_FLOAT;
    case TFWorkspace::DataType::TYPE_FP64:
      return tensorflow::DT_DOUBLE;
    case TFWorkspace::DataType::TYPE_STRING:
      return tensorflow::DT_STRING;
    default:
      break;
  }

  return tensorflow::DT_INVALID;
}

void
NewSessionOptions(
    const bool has_graph_level, const int graph_level,
    const bool allow_gpu_memory_growth,
    const float per_process_gpu_memory_fraction,
    const bool allow_soft_placement,
    tensorflow::SessionOptions* session_options)
{
  session_options->config.mutable_gpu_options()->set_allow_growth(
      allow_gpu_memory_growth);
  session_options->config.mutable_gpu_options()
      ->set_per_process_gpu_memory_fraction(per_process_gpu_memory_fraction);
  session_options->config.set_allow_soft_placement(allow_soft_placement);

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
}

}  // namespace

//
// TensorImpl
//
class TensorImpl : public TFWorkspace::Tensor {
 public:
  TensorImpl(
      const tensorflow::DataType dtype, const tensorflow::TensorShape& shape);
  TensorImpl(tensorflow::Tensor&& tftensor);

  TFWorkspace::DataType DataType() const override;
  int64_t DataTypeByteSize() const override;
  void Shape(std::vector<int64_t>* shape) const override;

  tensorflow::Tensor& TFTensor() { return tftensor_; }

  char* Base() const override { return nonstring_base_; }
  size_t ByteSize() const override { return nonstring_byte_size_; }

  const std::string& String(size_t idx) const override;
  void SetString(size_t idx, const std::string& str) override;

 private:
  void Init();

  tensorflow::Tensor tftensor_;
  char* nonstring_base_;
  size_t nonstring_byte_size_;
};

TensorImpl::TensorImpl(
    const tensorflow::DataType dtype, const tensorflow::TensorShape& shape)
    : tftensor_(dtype, shape)
{
  Init();
}

TensorImpl::TensorImpl(tensorflow::Tensor&& tftensor)
    : tftensor_(std::move(tftensor))
{
  Init();
}

void
TensorImpl::Init()
{
  nonstring_base_ = nullptr;
  nonstring_byte_size_ = 0;

  // Implement differently for string and non-string
  if (tftensor_.dtype() != tensorflow::DT_STRING) {
    auto flat = tftensor_.bit_casted_shaped<char, 1>(
        {tftensor_.NumElements() * DataTypeByteSize()});
    nonstring_base_ = static_cast<char*>(flat.data());
    nonstring_byte_size_ = flat.size();
  }
}

TFWorkspace::DataType
TensorImpl::DataType() const
{
  return ConvertDataType(tftensor_.dtype());
}

int64_t
TensorImpl::DataTypeByteSize() const
{
  return tensorflow::DataTypeSize(tftensor_.dtype());
}

void
TensorImpl::Shape(std::vector<int64_t>* shape) const
{
  for (int i = 0; i < tftensor_.shape().dims(); ++i) {
    int64_t dim = tftensor_.shape().dim_size(i);
    shape->push_back(dim);
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

TFWorkspace::Error
TFWorkspace::Tensor::Create(
    const TFWorkspace::DataType data_type, const std::vector<int64_t>& shape,
    std::unique_ptr<TFWorkspace::Tensor>* tensor)
{
  tensorflow::TensorShape tfshape;
  for (auto dim : shape) {
    tfshape.AddDim(dim);
  }

  tensor->reset(new TensorImpl(ConvertDataType(data_type), tfshape));
  return TFWorkspace::Error();
}

//
// TFWorkspaceImpl
//
class TFWorkspaceImpl : public TFWorkspace {
 public:
  TFWorkspaceImpl(
      const std::string& model_name,
      std::unique_ptr<tensorflow::SavedModelBundle> bundle,
      const IOList& inputs, const IOList& outputs);
  TFWorkspaceImpl(
      const std::string& model_name, tensorflow::Session* session,
      const IOList& inputs, const IOList& outputs);
  ~TFWorkspaceImpl();

  const IOList& Inputs() const override { return inputs_; }
  const IOList& Outputs() const override { return outputs_; }

  TFWorkspace::Error Run(
      TFWorkspace::TensorVec* input_tensors,
      const std::vector<std::string>& output_names,
      std::vector<std::unique_ptr<TFWorkspace::Tensor>>* outputs) override;

 private:
  std::string model_name_;
  std::unique_ptr<tensorflow::SavedModelBundle> bundle_;
  tensorflow::Session* session_;
  IOList inputs_;
  IOList outputs_;
};

TFWorkspaceImpl::TFWorkspaceImpl(
    const std::string& model_name,
    std::unique_ptr<tensorflow::SavedModelBundle> bundle, const IOList& inputs,
    const IOList& outputs)
    : model_name_(model_name), bundle_(std::move(bundle)), inputs_(inputs),
      outputs_(outputs)
{
  session_ = bundle_->session.release();
}

TFWorkspaceImpl::TFWorkspaceImpl(
    const std::string& model_name, tensorflow::Session* session,
    const IOList& inputs, const IOList& outputs)
    : session_(session), inputs_(inputs), outputs_(outputs)
{
}

TFWorkspaceImpl::~TFWorkspaceImpl()
{
  if (session_ != nullptr) {
    session_->Close().IgnoreError();
    delete session_;
    session_ = nullptr;
  }
}

TFWorkspace::Error
TFWorkspaceImpl::Run(
    TFWorkspace::TensorVec* input_tensors,
    const std::vector<std::string>& output_names,
    std::vector<std::unique_ptr<TFWorkspace::Tensor>>* outputs)
{
  std::vector<std::pair<std::string, tensorflow::Tensor>> tfinputs;
  for (const auto& pr : *input_tensors) {
    TensorImpl* tensor = static_cast<TensorImpl*>(pr.second.get());
    tfinputs.emplace_back(
        std::make_pair(pr.first, std::move(tensor->TFTensor())));
  }

  input_tensors->clear();

  std::vector<tensorflow::Tensor> tfoutputs;
  RETURN_IF_TF_ERROR(session_->Run(tfinputs, output_names, {}, &tfoutputs));

  for (auto& tftensor : tfoutputs) {
    outputs->emplace_back(new TensorImpl(std::move(tftensor)));
  }

  return TFWorkspace::Error();
}

TFWorkspace::Error
TFWorkspaceCreateFromGraphDef(
    TFWorkspace** tfws, const std::string& model_name,
    const std::string& model_path, const int gpu_device,
    const bool has_graph_level, const int graph_level,
    const bool allow_gpu_memory_growth,
    const float per_process_gpu_memory_fraction,
    const bool allow_soft_placement)
{
  tensorflow::SessionOptions session_options;
  NewSessionOptions(
      has_graph_level, graph_level, allow_gpu_memory_growth,
      per_process_gpu_memory_fraction, allow_soft_placement, &session_options);

  tensorflow::Session* session;
  RETURN_IF_TF_ERROR(tensorflow::NewSession(session_options, &session));

  tensorflow::GraphDef graph_def;
  RETURN_IF_TF_ERROR(tensorflow::ReadBinaryProto(
      tensorflow::Env::Default(), model_path, &graph_def));
  if (graph_def.node_size() == 0) {
    return TFWorkspace::Error("model " + model_name + " has an empty network");
  }

  // Set the default device to control the CPU/GPU that the graph runs
  // on. This isn't foolproof since individual operations in the graph
  // could specify a specific run location. But given that
  // visible_device_list doesn't work it seems like the only option we
  // have. [DLIS-43]
  if (gpu_device == TFWorkspace::NO_GPU_DEVICE) {
    tensorflow::graph::SetDefaultDevice("/cpu:0", &graph_def);
  } else {
    tensorflow::graph::SetDefaultDevice(
        "/gpu:" + std::to_string(gpu_device), &graph_def);
  }

  RETURN_IF_TF_ERROR(session->Create(graph_def));

  // Go through all graph nodes and collect the possible inputs and
  // outputs. We use this to verify the requested inputs and outputs
  // when initializing. Unfortunately graphdef isn't explicit in
  // indicating inputs and outputs so we assume any Placeholder can be
  // an input and any node can be an output.
  TFWorkspace::IOList potential_inputs, potential_outputs;
  for (const auto& node : graph_def.node()) {
    if (node.op() == "Placeholder") {
      potential_inputs.emplace_back();
      potential_inputs.back().name_ = node.name();
      potential_inputs.back().inmodel_name_ = node.name();
    } else {
      potential_outputs.emplace_back();
      potential_outputs.back().name_ = node.name();
      potential_outputs.back().inmodel_name_ = node.name();
    }
  }

  *tfws = new TFWorkspaceImpl(
      model_name, session, potential_inputs, potential_outputs);
  return TFWorkspace::Error();
}

TFWorkspace::Error
TFWorkspaceCreateFromSavedModel(
    TFWorkspace** tfws, const std::string& model_name,
    const std::string& model_path, const int gpu_device,
    const bool has_graph_level, const int graph_level,
    const bool allow_gpu_memory_growth,
    const float per_process_gpu_memory_fraction,
    const bool allow_soft_placement)
{
  tensorflow::SessionOptions session_options;
  NewSessionOptions(
      has_graph_level, graph_level, allow_gpu_memory_growth,
      per_process_gpu_memory_fraction, allow_soft_placement, &session_options);

  // Set the default device to control the CPU/GPU that the graph runs
  // on. This isn't foolproof since individual operations in the graph
  // could specify a specific run location. But given that
  // visible_device_list doesn't work it seems like the only option we
  // have. [DLIS-43]
  //
  // The GraphDef where we need to use this workaround is only
  // available in tensorflow/cc/saved_model/loader.cc so we use
  // visible_device_list in pass in the gpu_device we want and then
  // loader.cc (our modified version) will use that to
  // SetDefaultDevice appropriately.
  if (gpu_device == TFWorkspace::NO_GPU_DEVICE) {
    session_options.config.mutable_gpu_options()->set_visible_device_list(
        "/cpu:0");
  } else {
    session_options.config.mutable_gpu_options()->set_visible_device_list(
        "/gpu:" + std::to_string(gpu_device));
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
    return TFWorkspace::Error(
        "unable to load model '" + model_name + "', expected '" +
        tensorflow::kSavedModelTagServe + "' tag");
  }

  // Verify that a "serving_default" signature exists, that is what
  // will be used to verify the inputs and outputs.
  static const std::string DEFAULT_SERVING_SIGNATURE_DEF_KEY("serving_default");
  const auto& sig_itr = bundle->meta_graph_def.signature_def().find(
      DEFAULT_SERVING_SIGNATURE_DEF_KEY);
  if (sig_itr == bundle->meta_graph_def.signature_def().end()) {
    return TFWorkspace::Error(
        "unable to load model '" + model_name + "', expected '" +
        DEFAULT_SERVING_SIGNATURE_DEF_KEY + "' signature");
  }

  const tensorflow::SignatureDef& def = sig_itr->second;

  // Collect the inputs...
  TFWorkspace::IOList inputs;
  for (const auto& sin : def.inputs()) {
    inputs.emplace_back();
    auto& io = inputs.back();

    io.name_ = sin.first;

    const TFWorkspace::DataType dt = ConvertDataType(sin.second.dtype());
    if (dt == TFWorkspace::DataType::TYPE_INVALID) {
      return TFWorkspace::Error(
          "unable to process input '" + io.name_ + "' for '" + model_name +
          "', unsupported data-type '" +
          tensorflow::DataType_Name(sin.second.dtype()) + "'");
    }

    io.data_type_ = dt;

    const tensorflow::TensorShapeProto& shape = sin.second.tensor_shape();
    for (int i = 0; i < shape.dim().size(); ++i) {
      io.shape_.push_back(shape.dim(i).size());
    }
  }

  // Collect the outputs...
  TFWorkspace::IOList outputs;
  for (const auto& sin : def.outputs()) {
    outputs.emplace_back();
    auto& io = outputs.back();

    io.name_ = sin.first;
    io.inmodel_name_ = sin.second.name();

    const TFWorkspace::DataType dt = ConvertDataType(sin.second.dtype());
    if (dt == TFWorkspace::DataType::TYPE_INVALID) {
      return TFWorkspace::Error(
          "unable to process output '" + io.name_ + "' for '" + model_name +
          "', unsupported data-type '" +
          tensorflow::DataType_Name(sin.second.dtype()) + "'");
    }

    io.data_type_ = dt;

    const tensorflow::TensorShapeProto& shape = sin.second.tensor_shape();
    for (int i = 0; i < shape.dim().size(); ++i) {
      io.shape_.push_back(shape.dim(i).size());
    }
  }

  *tfws = new TFWorkspaceImpl(model_name, std::move(bundle), inputs, outputs);
  return TFWorkspace::Error();
}

}}  // namespace nvidia::inferenceserver
