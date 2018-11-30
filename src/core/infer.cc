// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include "src/core/infer.h"

#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <chrono>
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace nvidia { namespace inferenceserver {

GRPCInferRequestProvider::GRPCInferRequestProvider(
  const InferRequest& request, const int version)
    : InferRequestProvider(request.model_name(), version), request_(request)
{
  content_delivered_.resize(request_.raw_input_size(), false);
}

tensorflow::Status
GRPCInferRequestProvider::Create(
  const InferRequest& request,
  std::shared_ptr<GRPCInferRequestProvider>* infer_provider)
{
  // Make sure the request has a batch-size > 0. Even for models that
  // don't support batching the requested batch size must be 1.
  if (request.meta_data().batch_size() < 1) {
    return tensorflow::errors::InvalidArgument(
      "inference request batch-size must be >= 1 for models that ",
      "support batching, and must be 1 for models that don't ",
      "support batching");
  }

  const int version = (request.version() >= 0) ? request.version() : -1;
  infer_provider->reset(new GRPCInferRequestProvider(request, version));
  return tensorflow::Status::OK();
}

tensorflow::Status
GRPCInferRequestProvider::GetNextInputContent(
  int idx, const void** content, size_t* content_byte_size,
  bool force_contiguous)
{
  if ((idx < 0) || (idx >= request_.raw_input_size())) {
    return tensorflow::errors::Internal("unexpected input index ", idx);
  }

  if (content_delivered_[idx]) {
    *content = nullptr;
    *content_byte_size = 0;
  } else {
    const std::string& raw = request_.raw_input(idx);
    *content = raw.c_str();
    *content_byte_size = raw.size();
    content_delivered_[idx] = true;
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
HTTPInferRequestProvider::Create(
  evbuffer* input_buffer, const std::string& model_name,
  const std::string& model_version_str, const std::string& request_header_str,
  std::shared_ptr<HTTPInferRequestProvider>* infer_provider)
{
  int version = -1;
  if (!model_version_str.empty()) {
    version = std::atoi(model_version_str.c_str());
  }

  auto provider = new HTTPInferRequestProvider(model_name, version);
  infer_provider->reset(provider);

  if (!tensorflow::protobuf::TextFormat::ParseFromString(
        request_header_str, &(provider->request_header_))) {
    return tensorflow::errors::InvalidArgument(
      "unable to parse request for model '", model_name, "'");
  }

  // Make sure the request has a batch-size > 0. Even for models that
  // don't support batching the requested batch size must be 1.
  if (provider->request_header_.batch_size() < 1) {
    return tensorflow::errors::InvalidArgument(
      "inference request batch-size must be >= 1 for models that ",
      "support batching, and must be 1 for models that don't ",
      "support batching");
  }

  // Now need to create 'contents_'. Each input has one entry in
  // 'contents_' which gives a list of all the blocks of data for that
  // input. These blocks are not necessarily contiguous so we keep
  // track of each separately to avoid needing to copy everything into
  // one buffer.
  //
  // Get the addr and size of each chunk of input data from the
  // evbuffer.
  int n = evbuffer_peek(input_buffer, -1, NULL, NULL, 0);
  if (n > 0) {
    struct evbuffer_iovec* v = static_cast<struct evbuffer_iovec*>(
      alloca(sizeof(struct evbuffer_iovec) * n));
    if (evbuffer_peek(input_buffer, -1, NULL, v, n) != n) {
      return tensorflow::errors::Internal(
        "unexpected error getting input buffers ");
    }

    int v_idx = 0;

    // For each input get the blocks holding the data for that input
    for (const auto& input : provider->request_header_.input()) {
      provider->contents_idx_.push_back(0);
      provider->contents_.emplace_back();
      auto& blocks = provider->contents_.back();

      size_t total_byte_size =
        provider->request_header_.batch_size() * input.byte_size();
      while ((total_byte_size > 0) && (v_idx < n)) {
        blocks.emplace_back();
        Block& block = blocks.back();

        char* base = static_cast<char*>(v[v_idx].iov_base);
        block.first = base;
        if (v[v_idx].iov_len > total_byte_size) {
          block.second = total_byte_size;
          v[v_idx].iov_base = static_cast<void*>(base + total_byte_size);
          v[v_idx].iov_len -= total_byte_size;
          total_byte_size = 0;
        } else {
          block.second = v[v_idx].iov_len;
          total_byte_size -= v[v_idx].iov_len;
          v_idx++;
        }
      }
    }

    if (v_idx != n) {
      return tensorflow::errors::InvalidArgument(
        "unexpected additional input data for model '", provider->ModelName(),
        "'");
    }
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
HTTPInferRequestProvider::GetNextInputContent(
  int idx, const void** content, size_t* content_byte_size,
  bool force_contiguous)
{
  if ((idx < 0) || ((size_t)idx >= contents_.size())) {
    return tensorflow::errors::Internal("unexpected input index ", idx);
  }

  const size_t block_cnt = contents_[idx].size();
  const size_t block_idx = contents_idx_[idx];

  if (block_idx >= block_cnt) {
    *content = nullptr;
    *content_byte_size = 0;
  }
  // Return next block of data...
  else if (!force_contiguous || ((block_idx + 1) >= block_cnt)) {
    const auto& block = contents_[idx][block_idx];
    *content = block.first;
    *content_byte_size = block.second;
    contents_idx_[idx]++;
  }
  // If remaining data needs to be returned in one contiguous region
  // and there is more than one block remaining, then need to copy the
  // content into a single contiguous buffer.
  else {
    size_t total_size = 0;
    for (size_t i = block_idx; i < block_cnt; i++) {
      const auto& block = contents_[idx][i];
      total_size += block.second;
    }

    contiguous_buffers_.emplace_back();
    std::vector<char>& buf = contiguous_buffers_.back();
    buf.reserve(total_size);

    for (size_t i = block_idx; i < block_cnt; i++) {
      const auto& block = contents_[idx][i];
      buf.insert(buf.end(), block.first, block.first + block.second);
    }

    if (buf.size() != total_size) {
      return tensorflow::errors::Internal("contiguous input failed");
    }

    *content = &(buf[0]);
    *content_byte_size = total_size;
  }

  return tensorflow::Status::OK();
}


tensorflow::Status
GRPCInferResponseProvider::Create(
  const InferRequestHeader& request_header, InferResponse* response,
  std::shared_ptr<GRPCInferResponseProvider>* infer_provider)
{
  GRPCInferResponseProvider* provider =
    new GRPCInferResponseProvider(request_header, response);
  infer_provider->reset(provider);

  // Make space in the response for the output data. For outputs
  // returning raw tensor data we allocate space directly in the
  // response protobuf. For outputs returning classification we create
  // a buffer to hold the output that we can then post-process for
  // classifications.
  for (const auto& requested_output : request_header.output()) {
    std::string* output = provider->response_->add_raw_output();
    const size_t output_byte_size =
      request_header.batch_size() * requested_output.byte_size();

    if (requested_output.has_cls()) {
      provider->CreateOutputBuffer(output_byte_size);
    } else {
      output->resize(output_byte_size);
      provider->AddOutputBuffer(
        static_cast<void*>(&((*output)[0])), output->size());
    }
  }

  return tensorflow::Status::OK();
}

HTTPInferResponseProvider::HTTPInferResponseProvider(
  evbuffer* output_buffer, const InferRequestHeader& request_header)
    : InferResponseProvider(request_header), output_buffer_(output_buffer)
{
  // Get the total size needed for raw output tensors...
  total_raw_byte_size_ = 0;
  for (const auto& requested_output : request_header.output()) {
    if (!requested_output.has_cls()) {
      total_raw_byte_size_ +=
        request_header.batch_size() * requested_output.byte_size();
    }
  }
}

tensorflow::Status
HTTPInferResponseProvider::Create(
  evbuffer* output_buffer, const InferRequestHeader& request_header,
  std::shared_ptr<HTTPInferResponseProvider>* infer_provider)
{
  HTTPInferResponseProvider* provider =
    new HTTPInferResponseProvider(output_buffer, request_header);
  infer_provider->reset(provider);

  char* raw_output_base = nullptr;
  if (provider->total_raw_byte_size_ > 0) {
    // Reserve contiguous space in the output to hold all the raw output
    // tensor data that must be returned in the response.
    if (
      evbuffer_reserve_space(
        output_buffer, provider->total_raw_byte_size_, &provider->output_iovec_,
        1) != 1) {
      return tensorflow::errors::Internal(
        "failed to reserve ", provider->total_raw_byte_size_,
        " bytes in output tensor buffer");
    }

    if (provider->output_iovec_.iov_len < provider->total_raw_byte_size_) {
      return tensorflow::errors::Internal(
        "reserved ", provider->output_iovec_.iov_len,
        " bytes in output tensor buffer, need ",
        provider->total_raw_byte_size_);
    }

    provider->output_iovec_.iov_len = provider->total_raw_byte_size_;
    raw_output_base = static_cast<char*>(provider->output_iovec_.iov_base);
  }

  // For outputs returning raw tensor data we allocate space directly
  // from the space reserved in 'output_buffer'. For outputs returning
  // classification we create a buffer to hold the output that we can
  // then post-process for classifications.
  size_t raw_output_offset = 0;
  for (const auto& requested_output : request_header.output()) {
    const size_t output_byte_size =
      request_header.batch_size() * requested_output.byte_size();

    if (requested_output.has_cls()) {
      provider->CreateOutputBuffer(output_byte_size);
    } else {
      provider->AddOutputBuffer(
        static_cast<void*>(raw_output_base + raw_output_offset),
        output_byte_size);
      raw_output_offset += output_byte_size;
    }
  }

  if (raw_output_offset != provider->total_raw_byte_size_) {
    return tensorflow::errors::Internal(
      "failed to partition ", provider->total_raw_byte_size_,
      " bytes across output tensor buffer");
  }

  return tensorflow::Status::OK();
}

tensorflow::Status
HTTPInferResponseProvider::FinalizeResponse(const InferenceServable& is)
{
  if (total_raw_byte_size_ > 0) {
    if (evbuffer_commit_space(output_buffer_, &output_iovec_, 1) != 0) {
      return tensorflow::errors::Internal(
        "failed to commit output tensors to output buffer");
    }
  }

  return FinalizeResponseHeader(is);
}


namespace {

template <typename T>
void
AddClassResults(
  InferResponseHeader::Output* poutput, void* poutput_buffer,
  const size_t batch_size, const InferRequestHeader::Output& output,
  const LabelProvider& label_provider)
{
  T* probs = reinterpret_cast<T*>(poutput_buffer);
  const size_t entry_cnt = (output.byte_size() / sizeof(T));
  const size_t class_cnt = std::min((size_t)output.cls().count(), entry_cnt);
  std::vector<size_t> idx(entry_cnt);

  for (size_t i = 0; i < batch_size; ++i) {
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&probs](size_t i1, size_t i2) {
      return probs[i1] > probs[i2];
    });

    auto bcls = poutput->add_batch_classes();
    for (size_t k = 0; k < class_cnt; ++k) {
      auto cls = bcls->add_cls();
      cls->set_idx(idx[k]);
      cls->set_label(label_provider.GetLabel(output.name(), idx[k]));

      cls->set_value(static_cast<float>(probs[idx[k]]));
    }

    probs += entry_cnt;
  }
}

}  // namespace

void
InferResponseProvider::CreateOutputBuffer(size_t byte_size)
{
  char* buffer = new char[byte_size];
  created_buffers_.emplace_back(buffer);
  buffers_.emplace_back(buffer, byte_size);
}

void
InferResponseProvider::AddOutputBuffer(void* buffer, size_t byte_size)
{
  buffers_.emplace_back(buffer, byte_size);
}

tensorflow::Status
InferResponseProvider::GetOutputBuffer(
  int idx, void** buffer, size_t buffer_byte_size)
{
  if ((idx < 0) || (idx >= (int)buffers_.size())) {
    return tensorflow::errors::Internal("unexpected output index ", idx);
  }

  if (buffers_[idx].second != buffer_byte_size) {
    return tensorflow::errors::Internal(
      "unexpected output size ", buffers_[idx].second);
  }

  *buffer = buffers_[idx].first;
  return tensorflow::Status::OK();
}

tensorflow::Status
InferResponseProvider::FinalizeResponseHeader(const InferenceServable& is)
{
  InferResponseHeader* response_header = MutableResponseHeader();
  response_header->Clear();

  const LabelProvider& label_provider = is.GetLabelProvider();

  response_header->set_model_name(is.Name());
  response_header->set_model_version(is.Version());

  const size_t batch_size = request_header_.batch_size();
  response_header->set_batch_size(batch_size);

  int output_idx = 0;
  for (const auto& output : request_header_.output()) {
    auto poutput = response_header->add_output();
    poutput->set_name(output.name());

    if (!output.has_cls()) {
      poutput->mutable_raw()->set_byte_size(output.byte_size());
    } else {
      void* output_buffer;
      TF_RETURN_IF_ERROR(GetOutputBuffer(
        output_idx, &output_buffer, batch_size * output.byte_size()));

      DataType dtype;
      TF_RETURN_IF_ERROR(is.GetOutputDataType(output.name(), &dtype));

      switch (dtype) {
        case DataType::TYPE_UINT8:
          AddClassResults<uint8_t>(
            poutput, output_buffer, batch_size, output, label_provider);
          break;
        case DataType::TYPE_UINT16:
          AddClassResults<uint16_t>(
            poutput, output_buffer, batch_size, output, label_provider);
          break;
        case DataType::TYPE_UINT32:
          AddClassResults<uint32_t>(
            poutput, output_buffer, batch_size, output, label_provider);
          break;
        case DataType::TYPE_UINT64:
          AddClassResults<uint64_t>(
            poutput, output_buffer, batch_size, output, label_provider);
          break;

        case DataType::TYPE_INT8:
          AddClassResults<int8_t>(
            poutput, output_buffer, batch_size, output, label_provider);
          break;
        case DataType::TYPE_INT16:
          AddClassResults<int16_t>(
            poutput, output_buffer, batch_size, output, label_provider);
          break;
        case DataType::TYPE_INT32:
          AddClassResults<int32_t>(
            poutput, output_buffer, batch_size, output, label_provider);
          break;
        case DataType::TYPE_INT64:
          AddClassResults<int64_t>(
            poutput, output_buffer, batch_size, output, label_provider);
          break;

        case DataType::TYPE_FP32:
          AddClassResults<float>(
            poutput, output_buffer, batch_size, output, label_provider);
          break;
        case DataType::TYPE_FP64:
          AddClassResults<double>(
            poutput, output_buffer, batch_size, output, label_provider);
          break;

        default:
          return tensorflow::errors::InvalidArgument(
            "class result not available for output '", output.name(),
            "' due to unsupported type '", DataType_Name(dtype), "'");
      }
    }

    output_idx++;
  }

  return tensorflow::Status::OK();
}


InferenceServable::InferenceServable()
    : runner_cnt_(0), idle_runner_cnt_(0), max_preferred_batch_size_(0),
      pending_batch_delay_ns_(0), pending_batch_size_(0),
      pending_batch_queue_cnt_(0)
{
  runner_threads_exit_.store(false);
}

InferenceServable::~InferenceServable()
{
  // Signal the runner threads to exit and then wait for them...
  {
    std::unique_lock<std::mutex> lock(mu_);
    runner_threads_exit_.store(true);
    cv_.notify_all();
  }

  for (auto& runner : runner_threads_) {
    runner->join();
  }
}

void
InferenceServable::GetMetricLabels(
  std::map<std::string, std::string>* labels, const int gpu_device) const
{
  labels->insert(std::map<std::string, std::string>::value_type(
    std::string(kMetricsLabelModelName), Name()));
  labels->insert(std::map<std::string, std::string>::value_type(
    std::string(kMetricsLabelModelVersion), std::to_string(Version())));
  for (const auto& tag : Tags()) {
    labels->insert(std::map<std::string, std::string>::value_type(
      "_" + tag.first, tag.second));
  }

  // 'gpu_device' can be -1 to indicate that the GPU is not known. In
  // that case use a metric that doesn't have the gpu_uuid label.
  if (gpu_device >= 0) {
    std::string uuid;
    if (Metrics::UUIDForCudaDevice(gpu_device, &uuid)) {
      labels->insert(std::map<std::string, std::string>::value_type(
        std::string(kMetricsLabelGpuUuid), uuid));
    }
  }
}

prometheus::Counter&
InferenceServable::GetCounterMetric(
  std::map<int, prometheus::Counter*>& metrics,
  prometheus::Family<prometheus::Counter>& family, const int gpu_device) const
{
  const auto itr = metrics.find(gpu_device);
  if (itr != metrics.end()) {
    return *(itr->second);
  }

  std::map<std::string, std::string> labels;
  GetMetricLabels(&labels, gpu_device);

  prometheus::Counter& counter = family.Add(labels);
  metrics.insert(
    std::map<int, prometheus::Counter*>::value_type(gpu_device, &counter));
  return counter;
}

prometheus::Counter&
InferenceServable::MetricInferenceSuccess(int gpu_device) const
{
  return GetCounterMetric(
    metric_inf_success_, Metrics::FamilyInferenceSuccess(), gpu_device);
}

prometheus::Counter&
InferenceServable::MetricInferenceFailure(int gpu_device) const
{
  return GetCounterMetric(
    metric_inf_failure_, Metrics::FamilyInferenceFailure(), gpu_device);
}

prometheus::Counter&
InferenceServable::MetricInferenceCount(int gpu_device) const
{
  return GetCounterMetric(
    metric_inf_count_, Metrics::FamilyInferenceCount(), gpu_device);
}

prometheus::Counter&
InferenceServable::MetricInferenceExecutionCount(int gpu_device) const
{
  return GetCounterMetric(
    metric_inf_exec_count_, Metrics::FamilyInferenceExecutionCount(),
    gpu_device);
}

prometheus::Counter&
InferenceServable::MetricInferenceRequestDuration(int gpu_device) const
{
  return GetCounterMetric(
    metric_inf_request_duration_us_, Metrics::FamilyInferenceRequestDuration(),
    gpu_device);
}

prometheus::Counter&
InferenceServable::MetricInferenceComputeDuration(int gpu_device) const
{
  return GetCounterMetric(
    metric_inf_compute_duration_us_, Metrics::FamilyInferenceComputeDuration(),
    gpu_device);
}

prometheus::Counter&
InferenceServable::MetricInferenceQueueDuration(int gpu_device) const
{
  return GetCounterMetric(
    metric_inf_queue_duration_us_, Metrics::FamilyInferenceQueueDuration(),
    gpu_device);
}

prometheus::Histogram&
InferenceServable::MetricInferenceLoadRatio(int gpu_device) const
{
  const auto itr = metric_inf_load_ratio_.find(gpu_device);
  if (itr != metric_inf_load_ratio_.end()) {
    return *(itr->second);
  }

  std::map<std::string, std::string> labels;
  GetMetricLabels(&labels, gpu_device);

  prometheus::Histogram& hist = Metrics::FamilyInferenceLoadRatio().Add(
    labels, std::vector<double>{1.05, 1.10, 1.25, 1.5, 2.0, 10.0, 50.0});
  metric_inf_load_ratio_.insert(
    std::map<int, prometheus::Histogram*>::value_type(gpu_device, &hist));
  return hist;
}

tensorflow::Status
InferenceServable::SetModelConfig(
  const tensorflow::StringPiece& path, const ModelConfig& config)
{
  config_ = config;
  TF_RETURN_IF_ERROR(GetModelVersionFromPath(path, &version_));
  for (const auto& tag : config_.tags()) {
    tags_.insert(
      std::map<std::string, std::string>::value_type(tag.first, tag.second));
  }

  max_preferred_batch_size_ = 0;
  preferred_batch_sizes_.clear();
  for (const auto size : config.dynamic_batching().preferred_batch_size()) {
    max_preferred_batch_size_ =
      std::max(max_preferred_batch_size_, (size_t)size);
    preferred_batch_sizes_.insert(size);
  }

  pending_batch_delay_ns_ =
    (uint64_t)config.dynamic_batching().max_queue_delay_microseconds() * 1000;

  return tensorflow::Status::OK();
}

tensorflow::Status
InferenceServable::SetRunnerCount(uint32_t cnt)
{
  if (runner_cnt_ != 0) {
    return tensorflow::errors::Internal(
      "Attempt to change runner count from ", runner_cnt_, " to ", cnt,
      " not allowed");
  }

  runner_cnt_ = cnt;

  // Set default nice level unless overridden by model priority
  int nice = SCHEDULER_DEFAULT_NICE;
  if (config_.has_optimization()) {
    switch (config_.optimization().priority()) {
      case ModelOptimizationPolicy::PRIORITY_MAX:
        nice = 0;
        break;
      case ModelOptimizationPolicy::PRIORITY_MIN:
        nice = 19;
        break;
      default:
        nice = SCHEDULER_DEFAULT_NICE;
        break;
    }
  }

  // Create the runner threads for this servable.
  for (uint32_t c = 0; c < runner_cnt_; ++c) {
    runner_threads_.emplace_back(
      new std::thread([this, c, nice]() { RunnerThread(c, nice); }));
  }

  return tensorflow::Status::OK();
}

void
InferenceServable::AsyncRun(
  std::shared_ptr<ModelInferStats> stats,
  std::shared_ptr<InferRequestProvider> request_provider,
  std::shared_ptr<InferResponseProvider> response_provider,
  std::function<void(tensorflow::Status)> OnCompleteHandleInfer)
{
  auto run_timer = std::make_shared<ModelInferStats::ScopedTimer>();
  struct timespec queued_timestamp = stats->StartRunTimer(run_timer.get());
  bool wake_runner = false;
  {
    std::lock_guard<std::mutex> lock(mu_);
    queue_.emplace_back(
      queued_timestamp, stats, request_provider, response_provider,
      [OnCompleteHandleInfer, run_timer](tensorflow::Status status) mutable {
        run_timer.reset();
        OnCompleteHandleInfer(status);
      });

    // If there are any idle runners then wake one up to service this
    // request. We do the actual wake outside of the lock to avoid
    // having the woken thread immediately block on the lock
    wake_runner = (idle_runner_cnt_ > 0);
  }

  if (wake_runner) {
    cv_.notify_one();
  }
}

// Since callers are expecting synchronous behavior, this function
// must wait until the request is processed and the response is
// returned. This function can be simplified significantly once we
// have [DLIS-124].
void
InferenceServable::Run(
  std::shared_ptr<ModelInferStats> stats,
  std::shared_ptr<InferRequestProvider> request_provider,
  std::shared_ptr<InferResponseProvider> response_provider,
  std::function<void(tensorflow::Status)> OnCompleteHandleInfer)
{
  // Since this call is synchronous right now we can just use a scoped
  // timer to measure the entire run time.
  ModelInferStats::ScopedTimer run_timer;
  struct timespec queued_timestamp = stats->StartRunTimer(&run_timer);


  std::mutex lmu;
  std::condition_variable lcv;
  tensorflow::Status run_status;
  bool run_completed = false;
  bool wake_runner = false;

  // Add request to queue...
  {
    std::lock_guard<std::mutex> lock(mu_);
    queue_.emplace_back(
      queued_timestamp, stats, request_provider, response_provider,
      [&lmu, &lcv, &run_status, &run_completed](tensorflow::Status status) {
        // signal complete and propagate status
        {
          std::lock_guard<std::mutex> lk(lmu);
          run_status = status;
          run_completed = true;
        }
        lcv.notify_one();
      });

    // If there are any idle runners then wake one up to service this
    // request. We do the actual wake outside of the lock to avoid
    // having the woken thread immediately block on the lock
    wake_runner = (idle_runner_cnt_ > 0);
  }

  if (wake_runner) {
    cv_.notify_one();
  }

  // [DLIS-124] must wait for request to indicate complete...
  {
    std::chrono::seconds wait_timeout(1);
    std::unique_lock<std::mutex> lk(lmu);
    while (!run_completed) {
      lcv.wait_for(lk, wait_timeout);
    }
  }

  OnCompleteHandleInfer(run_status);
}

void
InferenceServable::RunnerThread(const uint32_t runner_id, const int nice)
{
  if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), nice) == 0) {
    LOG_INFO << "Starting runner thread " << runner_id << " at nice " << nice
             << "...";
  } else {
    LOG_ERROR << "Starting runner thread " << runner_id
              << " at default nice (requested nice " << nice << " failed)...";
  }

  // For testing, delay start of runner threads until the queue
  // contains the specified number of entries.
  const char* dstr = getenv("TRTSERVER_DELAY_SCHEDULER");
  size_t delay_cnt = 0;
  if (dstr != nullptr) {
    delay_cnt = atoi(dstr);
    LOG_INFO << "Delaying runner thread " << runner_id << " until " << delay_cnt
             << " queued payloads...";
  }

  const uint64_t default_wait_microseconds = 500 * 1000;
  const bool dynamic_batching_enabled = config_.has_dynamic_batching();

  while (!runner_threads_exit_.load()) {
    auto state = std::make_shared<RunnerThreadState>();
    bool wake_runner = false;
    uint64_t wait_microseconds = 0;

    // Hold the lock for as short a time as possible.
    {
      std::unique_lock<std::mutex> lock(mu_);
      if (delay_cnt > 0) {
        // Testing... wait until queue contains 'delay_cnt' items...
        wait_microseconds = 10 * 1000;
        if (queue_.size() >= delay_cnt) {
          delay_cnt = 0;
        }
      } else if (queue_.empty()) {
        wait_microseconds = default_wait_microseconds;
      } else if (dynamic_batching_enabled) {
        // Use dynamic batching to get request payload(s) to execute.
        wait_microseconds = GetDynamicBatch(config_.dynamic_batching());
        if (wait_microseconds == 0) {
          for (size_t idx = 0; idx < pending_batch_queue_cnt_; ++idx) {
            state->payloads.emplace_back(queue_.front());
            queue_.pop_front();
          }

          pending_batch_size_ = 0;
          pending_batch_queue_cnt_ = 0;

          // If there are still requests in the queue after removing
          // the pending batch and if there are any idle runners
          // then wake one up to service the requests remaining in
          // the queue. We need this special wake logic for the
          // dynamic batching case because we may delay handling
          // requests in the queue and so idle the runners that
          // would normally be handling those requests. We do the
          // actual wake outside of the lock to avoid having the
          // woken thread immediately block on the lock.
          wake_runner = !queue_.empty() && (idle_runner_cnt_ > 0);
        }
      } else {
        // No batching... execute next request payload
        state->payloads.emplace_back(queue_.front());
        queue_.pop_front();
      }

      // If no requests are to be handled, wait for notification or
      // for the specified timeout before checking the queue again.
      if (wait_microseconds > 0) {
        idle_runner_cnt_++;
        std::chrono::microseconds wait_timeout(wait_microseconds);
        cv_.wait_for(lock, wait_timeout);
        idle_runner_cnt_--;
      }
    }

    if (wake_runner) {
      cv_.notify_one();
    }

    if (!state->payloads.empty()) {
      auto OnCompleteQueuedPayloads = [state](tensorflow::Status status) {
        bool found_success = false;
        for (auto& payload : state->payloads) {
          tensorflow::Status final_status =
            status.ok() ? (payload.status_.ok() ? payload.compute_status_
                                                : payload.status_)
                        : status;

          // All the payloads executed together, so count 1 execution in
          // the first successful payload. Other payloads stay at 0
          // executions.
          if (!found_success && final_status.ok()) {
            payload.stats_->SetModelExecutionCount(1);
            found_success = true;
          }
          payload.complete_function_(final_status);
        }
      };
      Run(runner_id, &(state->payloads), OnCompleteQueuedPayloads);
    }

  }  // end runner loop

  LOG_INFO << "Stopping runner thread " << runner_id << "...";
}

uint64_t
InferenceServable::GetDynamicBatch(const ModelDynamicBatching& batching_config)
{
  // 'mu_' mutex must be held when this function is called. queue_
  // must not be empty.

  // Handle the cases where the pending batch or request must be
  // executed immediately.
  //
  //   1) if next request would make pending batch larger than the max
  //   preferred batch size then must execute the pending patch
  //   immediately
  //
  //   2) if no pending batch and next request on its own has batch
  //   size larger than the max preferred batch size then must execute
  //   immediately
  {
    const auto batch_size =
      queue_.front().request_provider_->RequestHeader().batch_size();
    if ((pending_batch_size_ + batch_size) >= max_preferred_batch_size_) {
      if (pending_batch_queue_cnt_ == 0) {
        pending_batch_size_ = batch_size;
        pending_batch_queue_cnt_ = 1;
      }
      return 0;
    }
  }

  // Examine the new requests. If adding these new requests to the
  // pending batch allows a preferred batch size then execute it
  // immediately. Stop examining requests if the maximum preferred
  // batch size would be exceeded.
  size_t best_preferred_batch_size = 0;
  size_t best_preferred_batch_cnt = 0;
  size_t search_batch_size = pending_batch_size_;
  size_t search_batch_cnt = pending_batch_queue_cnt_;
  for (auto idx = pending_batch_queue_cnt_; idx < queue_.size(); ++idx) {
    const auto batch_size =
      queue_[idx].request_provider_->RequestHeader().batch_size();

    if ((search_batch_size + batch_size) > max_preferred_batch_size_) {
      break;
    }

    search_batch_size += batch_size;
    search_batch_cnt++;

    if (
      preferred_batch_sizes_.find(search_batch_size) !=
      preferred_batch_sizes_.end()) {
      best_preferred_batch_size = search_batch_size;
      best_preferred_batch_cnt = search_batch_cnt;
    }
  }

  // If we found a preferred batch size then execute that.
  if (best_preferred_batch_size != 0) {
    pending_batch_size_ = best_preferred_batch_size;
    pending_batch_queue_cnt_ = best_preferred_batch_cnt;
    return 0;
  }

  pending_batch_size_ = search_batch_size;
  pending_batch_queue_cnt_ = search_batch_cnt;

  // Should always have at least one request in the pending batch at
  // this point.
  if (pending_batch_queue_cnt_ == 0) {
    LOG_ERROR << "unexpected pending batch size 0";
    return 0;
  }

  // If there is no batch queuing delay then just immediately
  // execute whatever is pending.
  if (pending_batch_delay_ns_ == 0) {
    return 0;
  }

  // Compare the age of the oldest pending request to the maximum
  // batch queuing delay and execute now if queuing delay is
  // exceeded. If queuing delay not exceeded create a timer to wakeup
  // a thread to check again at the maximum allowed delay.
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  struct timespec& queued = queue_.front().queued_timestamp_;
  uint64_t delay_ns = (now.tv_sec * NANOS_PER_SECOND + now.tv_nsec) -
                      (queued.tv_sec * NANOS_PER_SECOND + queued.tv_nsec);

  if (delay_ns >= pending_batch_delay_ns_) {
    return 0;
  }

  // Return non-zero wait microseconds to cause this runner to wait
  // until the queue delay has expired. Another thread may be awaken
  // due to incoming request to handle the pending batch before this
  // thread wakes and that is ok. But if no other request comes in
  // then this thread will wake and revist the pending batch (and at
  // that time will then see the delay has been exceeded and will send
  // the batch).
  return (pending_batch_delay_ns_ - delay_ns) / 1000;
}

}}  // namespace nvidia::inferenceserver
