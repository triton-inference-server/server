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

#include "src/servables/tensorflow/base_bundle.h"

#include <set>
#include "cuda/include/cuda_runtime_api.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.pb.h"
#include "src/core/provider.h"
#include "src/core/server_status.h"
#include "src/core/utils.h"
#include "src/servables/tensorflow/tf_utils.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/lib/io/path.h"

namespace nvidia { namespace inferenceserver {

BaseBundle::Context::Context(
    const std::string& name, const int gpu_device, const int max_batch_size)
    : name_(name), gpu_device_(gpu_device), max_batch_size_(max_batch_size),
      session_(nullptr)
{
}

BaseBundle::Context::Context(Context&& o)
    : name_(std::move(o.name_)), gpu_device_(o.gpu_device_),
      max_batch_size_(o.max_batch_size_),
      input_name_map_(std::move(o.input_name_map_)),
      output_name_map_(std::move(o.output_name_map_)), session_(o.session_)
{
  o.gpu_device_ = NO_GPU_DEVICE;
  o.max_batch_size_ = NO_BATCHING;
  o.session_ = nullptr;
}

BaseBundle::Context::~Context()
{
  LOG_VERBOSE(1) << "~BaseBundle::Context ";

  if (session_ != nullptr) {
    session_->Close().IgnoreError();
    session_ = nullptr;
  }
}

tensorflow::Status
BaseBundle::Init(const tensorflow::StringPiece& path, const ModelConfig& config)
{
  TF_RETURN_IF_ERROR(SetModelConfig(path, config));
  return tensorflow::Status::OK();
}

tensorflow::Status
BaseBundle::CreateExecutionContexts(
    const tensorflow::ConfigProto& session_config,
    const std::unordered_map<std::string, std::string>& paths)
{
  if (LOG_VERBOSE_IS_ON(1)) {
    LOG_INFO << "Creating execution contexts for:";
    for (const auto p : paths) {
      LOG_INFO << "  " << p.first << ": " << p.second;
    }
  }

  uint32_t total_context_cnt = 0;

  for (const auto& group : Config().instance_group()) {
    for (int c = 0; c < group.count(); c++) {
      if (group.kind() == ModelInstanceGroup::KIND_CPU) {
        const std::string instance_name =
            group.name() + "_" + std::to_string(c) + "_cpu";
        TF_RETURN_IF_ERROR(CreateExecutionContext(
            instance_name, Context::NO_GPU_DEVICE, session_config, paths));
        total_context_cnt++;
      } else {
        for (int gpu_device : group.gpus()) {
          const std::string instance_name = group.name() + "_" +
                                            std::to_string(c) + "_gpu" +
                                            std::to_string(gpu_device);
          TF_RETURN_IF_ERROR(CreateExecutionContext(
              instance_name, gpu_device, session_config, paths));
          total_context_cnt++;
        }
      }
    }
  }

  // Create a scheduler with one thread for each context available for
  // this model. Each runner is exclusively tied to the context.
  TF_RETURN_IF_ERROR(SetConfiguredScheduler(
      total_context_cnt,
      [this](
          uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
          std::function<void(tensorflow::Status)> func) {
        Run(runner_idx, payloads, func);
      }));

  LOG_VERBOSE(1) << "bundle for " << Name() << std::endl << *this;

  return tensorflow::Status::OK();
}

tensorflow::Status
BaseBundle::CreateExecutionContext(
    const std::string& instance_name, const int gpu_device,
    const tensorflow::ConfigProto& session_config,
    const std::unordered_map<std::string, std::string>& paths)
{
  // For a GPU context, determine the model file to use for device
  // compute capability. CPU always uses the default model file.
  std::string cc_model_filename;
  if (gpu_device == Context::NO_GPU_DEVICE) {
    cc_model_filename = Config().default_model_filename();

    LOG_INFO << "Creating instance " << instance_name << " on CPU using "
             << cc_model_filename;
  } else {
    cudaDeviceProp cuprops;
    cudaError_t cuerr = cudaGetDeviceProperties(&cuprops, gpu_device);
    if (cuerr != cudaSuccess) {
      return tensorflow::errors::Internal(
          "unable to get CUDA device properties for ", Name(), ": ",
          cudaGetErrorString(cuerr));
    }

    const std::string cc =
        std::to_string(cuprops.major) + "." + std::to_string(cuprops.minor);
    const auto& cc_itr = Config().cc_model_filenames().find(cc);
    cc_model_filename = (cc_itr == Config().cc_model_filenames().end())
                            ? Config().default_model_filename()
                            : cc_itr->second;

    LOG_INFO << "Creating instance " << instance_name << " on GPU "
             << gpu_device << " (" << cc << ") using " << cc_model_filename;
  }

  const auto& gdp_itr = paths.find(cc_model_filename);
  if (gdp_itr == paths.end()) {
    return tensorflow::errors::Internal(
        "unable to find model '", cc_model_filename, "' for ", Name());
  }

  // Max batch size. A value of 0 in the config becomes NO_BATCHING.
  const int mbs = (Config().max_batch_size() <= 0) ? Context::NO_BATCHING
                                                   : Config().max_batch_size();

  contexts_.emplace_back(instance_name, gpu_device, mbs);
  Context& context = contexts_.back();

  // Session GPU option visible_device_list does not work (see
  // https://github.com/tensorflow/tensorflow/issues/8136 and many
  // related issues), so we can't use it here to set the GPU (see
  // CreateSession implementations for SetDefaultDevice). [DLIS-43]
  tensorflow::SessionOptions options;
  options.config = session_config;

  // Enable/disable XLA based on the model config optimization
  // setting.
  tensorflow::OptimizerOptions::GlobalJitLevel xla =
      tensorflow::OptimizerOptions::DEFAULT;
  if (Config().optimization().has_graph()) {
    if (Config().optimization().graph().level() == -1) {
      xla = tensorflow::OptimizerOptions::OFF;
    } else if (Config().optimization().graph().level() == 1) {
      xla = tensorflow::OptimizerOptions::ON_1;
    } else if (Config().optimization().graph().level() > 1) {
      xla = tensorflow::OptimizerOptions::ON_2;
    }
  }

  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_global_jit_level(xla);

  TF_RETURN_IF_ERROR(CreateSession(
      options, gpu_device, gdp_itr->second, &context.session_,
      &context.input_name_map_, &context.output_name_map_));

  return tensorflow::Status::OK();
}

void
BaseBundle::Run(
    uint32_t runner_idx, std::vector<Scheduler::Payload>* payloads,
    std::function<void(tensorflow::Status)> OnCompleteQueuedPayloads)
{
  // Each runner executes using the corresponding context...
  if (runner_idx >= contexts_.size()) {
    OnCompleteQueuedPayloads(tensorflow::errors::Internal(
        "unexpected runner index", runner_idx, ", max allowed ",
        contexts_.size()));
    return;
  }

  std::vector<ModelInferStats::ScopedTimer> compute_timers;
  for (auto& payload : *payloads) {
    // Stop queue timer when the payload is scheduled to run
    payload.queue_timer_.reset();

    compute_timers.emplace_back();
    payload.stats_->StartComputeTimer(&compute_timers.back());
    payload.stats_->SetGPUDevice(contexts_[runner_idx].gpu_device_);
  }

  OnCompleteQueuedPayloads(contexts_[runner_idx].Run(this, payloads));
}

namespace {

void
SetFixedSizedInputTensor(
    tensorflow::Tensor& tensor, const std::string& input_name,
    const size_t batch1_byte_size, std::vector<Scheduler::Payload>* payloads)
{
  auto flat = tensor.bit_casted_shaped<char, 1>(
      {tensor.NumElements() * tensorflow::DataTypeSize(tensor.dtype())});
  size_t tensor_copy_offset = 0;

  // Visit the payloads in order and copy the input values into the
  // input tensor. Skip payloads that had errors since they are not
  // included in the dynamic batch.
  for (auto& payload : *payloads) {
    if (!payload.status_.ok()) {
      continue;
    }

    const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();
    const size_t expected_byte_size =
        request_header.batch_size() * batch1_byte_size;

    int input_idx = 0;
    for (const auto& input : request_header.input()) {
      if (input.name() == input_name) {
        size_t copied_byte_size = 0;
        while (payload.compute_status_.ok()) {
          const void* content;
          size_t content_byte_size = expected_byte_size - copied_byte_size;
          payload.compute_status_ =
              payload.request_provider_->GetNextInputContent(
                  input_idx, &content, &content_byte_size, false);
          if (!payload.compute_status_.ok()) {
            break;
          }

          // No more input content available then done with copying...
          if (content == nullptr) {
            break;
          }

          if ((tensor_copy_offset + copied_byte_size + content_byte_size) >
              ((size_t)flat.size())) {
            payload.compute_status_ = tensorflow::errors::InvalidArgument(
                "unexpected size ",
                tensor_copy_offset + copied_byte_size + content_byte_size,
                " for inference input '", input_name, "', expecting ",
                flat.size());
            break;
          }

          memcpy(
              static_cast<char*>(flat.data()) + tensor_copy_offset +
                  copied_byte_size,
              content, content_byte_size);
          copied_byte_size += content_byte_size;
        }

        if (payload.compute_status_.ok() &&
            (copied_byte_size != expected_byte_size)) {
          payload.compute_status_ = tensorflow::errors::Internal(
              "expected ", expected_byte_size,
              " bytes of data for inference input '", input_name, "', got ",
              copied_byte_size);
        }

        break;
      }

      input_idx++;
    }

    tensor_copy_offset += expected_byte_size;
  }
}

void
FillStringTensor(tensorflow::Tensor& tensor, const size_t idx, const size_t cnt)
{
  auto flat = tensor.flat<std::string>();
  std::string empty;

  for (size_t c = 0; c < cnt; ++c) {
    flat(idx + c) = empty;
  }
}

void
SetStringInputTensor(
    tensorflow::Tensor& tensor, const std::string& input_name,
    const size_t batch1_element_cnt, std::vector<Scheduler::Payload>* payloads)
{
  auto flat = tensor.flat<std::string>();
  size_t tensor_element_idx = 0;

  // Visit the payloads in order and copy the input values into the
  // input tensor. Skip payloads that had errors since they are not
  // included in the dynamic batch.
  for (auto& payload : *payloads) {
    if (!payload.status_.ok()) {
      continue;
    }

    const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();
    const size_t expected_element_cnt =
        request_header.batch_size() * batch1_element_cnt;
    size_t element_idx = 0;

    int input_idx = 0;
    for (const auto& input : request_header.input()) {
      if (input.name() == input_name) {
        const void* vcontent;
        size_t content_byte_size = expected_element_cnt * sizeof(uint32_t);
        payload.compute_status_ =
            payload.request_provider_->GetNextInputContent(
                input_idx, &vcontent, &content_byte_size, true);
        if (!payload.compute_status_.ok()) {
          FillStringTensor(
              tensor, tensor_element_idx + element_idx,
              expected_element_cnt - element_idx);
          break;
        }

        const char* content = reinterpret_cast<const char*>(vcontent);

        // No more input content available then done with copying...
        if (content == nullptr) {
          break;
        }

        // Parse content and assign them to the 'tensor'. Each string
        // in 'content' is a 4-byte length followed by the string
        // itself with no null-terminator.
        while (content_byte_size >= sizeof(uint32_t)) {
          if (element_idx >= expected_element_cnt) {
            payload.compute_status_ = tensorflow::errors::InvalidArgument(
                "unexpected number of string elements ", element_idx + 1,
                " for inference input '", input_name, "', expecting ",
                expected_element_cnt);
            FillStringTensor(
                tensor, tensor_element_idx + element_idx,
                expected_element_cnt - element_idx);
            break;
          }

          const uint32_t len = *(reinterpret_cast<const uint32_t*>(content));
          content += sizeof(uint32_t);
          content_byte_size -= sizeof(uint32_t);

          if (content_byte_size < len) {
            payload.compute_status_ = tensorflow::errors::InvalidArgument(
                "incomplete string data for inference input '", input_name,
                "', expecting string of length ", len, " but only ",
                content_byte_size, " bytes available");
            FillStringTensor(
                tensor, tensor_element_idx + element_idx,
                expected_element_cnt - element_idx);
            break;
          }

          std::string str(content, len);
          content += len;
          content_byte_size -= len;

          flat(tensor_element_idx + element_idx) = str;
          element_idx++;
        }

        break;
      }

      input_idx++;
    }

    if (payload.compute_status_.ok() && (element_idx != expected_element_cnt)) {
      payload.compute_status_ = tensorflow::errors::Internal(
          "expected ", expected_element_cnt, " strings for inference input '",
          input_name, "', got ", element_idx);
      FillStringTensor(
          tensor, tensor_element_idx + element_idx,
          expected_element_cnt - element_idx);
    }

    tensor_element_idx += expected_element_cnt;
  }
}

void
ReadFixedSizedOutputTensor(
    tensorflow::Tensor& tensor, const std::string& output_name,
    const std::vector<int64_t>& shape, const size_t batch1_byte_size,
    std::vector<Scheduler::Payload>* payloads)
{
  const auto& flat = tensor.bit_casted_shaped<char, 1>(
      {tensor.NumElements() * tensorflow::DataTypeSize(tensor.dtype())});
  size_t tensor_copy_offset = 0;

  for (auto& payload : *payloads) {
    if (!payload.status_.ok()) {
      continue;
    }

    // If 'payload' requested this output then copy it from the
    // GPU. If it did not request this output then just skip it in
    // the output buffer.
    const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();
    const size_t expected_byte_size =
        request_header.batch_size() * batch1_byte_size;

    if (payload.response_provider_->RequiresOutput(output_name)) {
      void* content = nullptr;
      tensorflow::Status status = payload.response_provider_->GetOutputBuffer(
          output_name, &content, expected_byte_size, shape);
      if (!status.ok()) {
        payload.compute_status_ = status;
      } else {
        if ((tensor_copy_offset + expected_byte_size) > ((size_t)flat.size())) {
          payload.compute_status_ = tensorflow::errors::InvalidArgument(
              "unexpected size ", tensor_copy_offset + expected_byte_size,
              " for inference output '", output_name, "', expecting ",
              flat.size());
        } else {
          memcpy(
              content, static_cast<char*>(flat.data()) + tensor_copy_offset,
              expected_byte_size);
        }
      }
    }

    tensor_copy_offset += expected_byte_size;
  }
}

void
ReadStringOutputTensor(
    tensorflow::Tensor& tensor, const std::string& output_name,
    const std::vector<int64_t>& shape, const size_t batch1_element_cnt,
    std::vector<Scheduler::Payload>* payloads)
{
  auto flat = tensor.flat<std::string>();
  size_t tensor_element_idx = 0;

  for (auto& payload : *payloads) {
    if (!payload.status_.ok()) {
      continue;
    }

    // If 'payload' requested this output then copy it from the
    // GPU. If it did not request this output then just skip it in
    // the output tensor.
    const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();
    const size_t expected_element_cnt =
        request_header.batch_size() * batch1_element_cnt;

    if (payload.response_provider_->RequiresOutput(output_name)) {
      // Serialize the output tensor strings. Each string is
      // serialized as a 4-byte length followed by the string itself
      // with no null-terminator.
      std::string serialized;
      for (size_t e = 0; e < expected_element_cnt; ++e) {
        std::string& str = flat(tensor_element_idx + e);
        const uint32_t len = str.size();
        serialized.append(
            reinterpret_cast<const char*>(&len), sizeof(uint32_t));
        serialized.append(str);
      }

      void* content;
      tensorflow::Status status = payload.response_provider_->GetOutputBuffer(
          output_name, &content, serialized.size(), shape);
      if (status.ok()) {
        memcpy(
            content, reinterpret_cast<const void*>(serialized.c_str()),
            serialized.size());
      } else {
        payload.compute_status_ = status;
      }
    }

    tensor_element_idx += expected_element_cnt;
  }
}

}  // namespace


tensorflow::Status
BaseBundle::Context::Run(
    const BaseBundle* base, std::vector<Scheduler::Payload>* payloads)
{
  LOG_VERBOSE(1) << "Running " << name_ << " with " << payloads->size()
                 << " request payloads";

  const InferRequestHeader* input_request_header = nullptr;

  // For each request in 'payloads' collect the total batch size for
  // this inference execution. The batch-size, number of inputs, and
  // size of each input has already been checked by each payloads
  // request provider so don't need to do that here.
  size_t total_batch_size = 0;
  for (auto& payload : *payloads) {
    if (payload.status_.ok()) {
      total_batch_size +=
          payload.request_provider_->RequestHeader().batch_size();

      // All payloads must have equally-sized input tensors so use any
      // payload as the representative for the input tensors.
      input_request_header = &(payload.request_provider_->RequestHeader());
    }
  }

  // If there are no valid payloads then no need to run the
  // inference. The payloads will have their error status set so can
  // just return.
  if (total_batch_size == 0) {
    return tensorflow::Status::OK();
  }

  // total_batch_size can be 1 for models that don't support batching
  // (i.e. max_batch_size_ == 0).
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size_)) {
    return tensorflow::errors::Internal(
        "dynamic batch size ", total_batch_size, " for '", name_,
        "', max allowed is ", max_batch_size_);
  }

  // Create a tensor for each input sized correctly for the total
  // payload batch size. Concatenate input values from each payload
  // into the corresponding tensor.
  using TensorVec = std::vector<std::pair<std::string, tensorflow::Tensor>>;
  TensorVec input_tensors;

  for (const auto& input : input_request_header->input()) {
    const std::string& name = input.name();

    const ModelInput* input_config;
    TF_RETURN_IF_ERROR(base->GetInput(input.name(), &input_config));

    const tensorflow::DataType dtype =
        ConvertDataType(input_config->data_type());

    // Get the shape of the input. The provider has already checked
    // that the request shape is valid so don't need to do it here.
    tensorflow::TensorShape shape;

    // If model supports batching then prepend the batch dimension
    // onto the input shape.
    if (max_batch_size_ != NO_BATCHING) {
      shape.AddDim(total_batch_size);
    }

    size_t batch1_element_cnt = 1;
    for (auto dim : input.dims()) {
      shape.AddDim(dim);
      batch1_element_cnt *= dim;
    }

    const std::string* input_tensor_name = &name;
    const auto& tn_itr = input_name_map_.find(name);
    if (tn_itr != input_name_map_.end()) {
      input_tensor_name = &tn_itr->second;
    }

    input_tensors.emplace_back(
        std::make_pair(*input_tensor_name, tensorflow::Tensor(dtype, shape)));
    tensorflow::Tensor& tensor = input_tensors.back().second;

    if (dtype != tensorflow::DT_STRING) {
      const size_t batch1_byte_size =
          batch1_element_cnt * tensorflow::DataTypeSize(dtype);
      SetFixedSizedInputTensor(tensor, name, batch1_byte_size, payloads);
    } else {
      SetStringInputTensor(tensor, name, batch1_element_cnt, payloads);
    }
  }

  // Collect the names of outputs requested by any request
  // payload. Skip payloads that have an error.
  std::set<std::string> required_outputs;
  for (auto& payload : *payloads) {
    if (!payload.status_.ok()) {
      continue;
    }

    const InferRequestHeader& request_header =
        payload.request_provider_->RequestHeader();
    for (const auto& output : request_header.output()) {
      required_outputs.insert(output.name());
    }
  }

  // Create the vector of required output names using the names
  // expected by the model.
  std::vector<std::string> model_output_names;
  std::vector<std::string> output_names;
  for (const auto& name : required_outputs) {
    model_output_names.push_back(name);
    const auto& tn_itr = output_name_map_.find(name);
    if (tn_itr == output_name_map_.end()) {
      output_names.push_back(name);
    } else {
      output_names.push_back(tn_itr->second);
    }
  }

  // Run. Session will update the 'outputs'.
  std::vector<tensorflow::Tensor> outputs;
  TF_RETURN_IF_ERROR(session_->Run(input_tensors, output_names, {}, &outputs));

  // Make sure each output is of the expected size and copy it into
  // the appropriate response providers.
  int output_idx = 0;
  for (const auto& name : model_output_names) {
    const ModelOutput* output_config;
    TF_RETURN_IF_ERROR(base->GetOutput(name, &output_config));

    // Get the shape of the output from the output tensor.
    std::vector<int64_t> shape;
    bool skip_element_cnt = (max_batch_size_ != NO_BATCHING);
    size_t batch1_element_cnt = 1;
    for (int i = 0; i < outputs[output_idx].shape().dims(); ++i) {
      int64_t dim = outputs[output_idx].shape().dim_size(i);
      shape.push_back(dim);

      if (!skip_element_cnt) {
        batch1_element_cnt *= dim;
      }
      skip_element_cnt = false;
    }

    tensorflow::DataType dtype = ConvertDataType(output_config->data_type());
    if (dtype != outputs[output_idx].dtype()) {
      return tensorflow::errors::InvalidArgument(
          "unexpected datatype ", outputs[output_idx].dtype(),
          " for inference output '", name, "', expecting ", dtype);
    }

    if (dtype != tensorflow::DT_STRING) {
      const size_t batch1_byte_size =
          batch1_element_cnt * tensorflow::DataTypeSize(dtype);
      ReadFixedSizedOutputTensor(
          outputs[output_idx], name, shape, batch1_byte_size, payloads);
    } else {
      ReadStringOutputTensor(
          outputs[output_idx], name, shape, batch1_element_cnt, payloads);
    }

    output_idx++;
  }

  return tensorflow::Status::OK();
}

std::ostream&
operator<<(std::ostream& out, const BaseBundle& pb)
{
  out << "name=" << pb.Name() << std::endl;
  out << "contexts:" << std::endl;
  for (const auto& context : pb.contexts_) {
    out << "  name=" << context.name_ << ", gpu="
        << ((context.gpu_device_ == BaseBundle::Context::NO_GPU_DEVICE)
                ? "<none>"
                : std::to_string(context.gpu_device_))
        << ", max_batch_size="
        << ((context.max_batch_size_ == BaseBundle::Context::NO_BATCHING)
                ? "<none>"
                : std::to_string(context.max_batch_size_))
        << std::endl;
  }

  return out;
}

}}  // namespace nvidia::inferenceserver
