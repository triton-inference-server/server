// Copyright 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tracer.h"

#include <stdlib.h>

#include "common.h"
#include "triton/common/logging.h"
#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU
#ifndef _WIN32
#include "opentelemetry/sdk/resource/semantic_conventions.h"
#include "opentelemetry/sdk/trace/batch_span_processor_factory.h"
namespace otel_common = opentelemetry::common;
#endif

namespace triton { namespace server {

TRITONSERVER_Error*
TraceManager::Create(
    TraceManager** manager, const TRITONSERVER_InferenceTraceLevel level,
    const uint32_t rate, const int32_t count, const uint32_t log_frequency,
    const std::string& filepath, const InferenceTraceMode mode,
    const triton::server::TraceConfigMap& config_map)
{
  // Always create TraceManager regardless of the global setting as they
  // can be updated at runtime even if tracing is not enable at start.
  // No trace should be sampled if the setting is not valid.
  *manager = new TraceManager(
      level, rate, count, log_frequency, filepath, mode, config_map);

  return nullptr;  // success
}

TraceManager::TraceManager(
    const TRITONSERVER_InferenceTraceLevel level, const uint32_t rate,
    const int32_t count, const uint32_t log_frequency,
    const std::string& filepath, const InferenceTraceMode mode,
    const TraceConfigMap& config_map)
{
  std::shared_ptr<TraceFile> file(new TraceFile(filepath));
  global_default_.reset(new TraceSetting(
      level, rate, count, log_frequency, file, mode, config_map,
      false /*level_specified*/, false /*rate_specified*/,
      false /*count_specified*/, false /*log_frequency_specified*/,
      false /*filepath_specified*/, false /*mode_specified*/,
      false /*config_map_specified*/));
  global_setting_.reset(new TraceSetting(
      level, rate, count, log_frequency, file, mode, config_map,
      false /*level_specified*/, false /*rate_specified*/,
      false /*count_specified*/, false /*log_frequency_specified*/,
      false /*filepath_specified*/, false /*mode_specified*/,
      false /*config_map_specified*/));
  trace_files_.emplace(filepath, file);

  InitTracer(config_map);
}

TRITONSERVER_Error*
TraceManager::UpdateTraceSetting(
    const std::string& model_name, const NewSetting& new_setting)
{
  std::lock_guard<std::mutex> w_lk(w_mu_);

  RETURN_IF_ERR(UpdateTraceSettingInternal(model_name, new_setting));
  // If updating global setting, must check and update the model settings
  // that are (partially) mirroring global setting.
  if (model_name.empty()) {
    // Default constructed setting means no active update,
    // only the unspecified fields will be checked and updated.
    NewSetting setting;
    // Make a copy of the set as UpdateTraceSettingInternal() may modify
    // 'fallback_used_models_'
    auto fallback_models = fallback_used_models_;
    for (const auto& name : fallback_models) {
      RETURN_IF_ERR(UpdateTraceSettingInternal(name, setting));
    }
  }
  return nullptr;
}

TRITONSERVER_Error*
TraceManager::UpdateTraceSettingInternal(
    const std::string& model_name, const NewSetting& new_setting)
{
  // First try to get the current setting and fallback setting,
  // current setting may be 'nullptr' if the setting is newly added
  const TraceSetting* current_setting = nullptr;
  const TraceSetting* fallback_setting = nullptr;
  if (!model_name.empty()) {
    auto it = model_settings_.find(model_name);
    if (it != model_settings_.end()) {
      current_setting = it->second.get();
    }
    fallback_setting = global_setting_.get();
  } else {
    current_setting = global_setting_.get();
    fallback_setting = global_default_.get();
  }

  // Prepare the updated setting, use two passes for simplicity:
  // 1. Set all fields based on 'fallback_setting'
  // 2. If there are specified fields based on current and new setting,
  //    use the specified value
  TRITONSERVER_InferenceTraceLevel level = fallback_setting->level_;
  uint32_t rate = fallback_setting->rate_;
  int32_t count = fallback_setting->count_;
  uint32_t log_frequency = fallback_setting->log_frequency_;
  std::string filepath = fallback_setting->file_->FileName();
  InferenceTraceMode mode = fallback_setting->mode_;
  TraceConfigMap config_map = fallback_setting->config_map_;

  // Whether the field value is specified:
  // if clear then it is not specified, otherwise,
  // it is specified if it is being updated, or it was previously specified
  const bool level_specified =
      (new_setting.clear_level_ ? false
                                : (((current_setting != nullptr) &&
                                    current_setting->level_specified_) ||
                                   (new_setting.level_ != nullptr)));
  const bool rate_specified =
      (new_setting.clear_rate_ ? false
                               : (((current_setting != nullptr) &&
                                   current_setting->rate_specified_) ||
                                  (new_setting.rate_ != nullptr)));
  const bool count_specified =
      (new_setting.clear_count_ ? false
                                : (((current_setting != nullptr) &&
                                    current_setting->count_specified_) ||
                                   (new_setting.count_ != nullptr)));
  const bool log_frequency_specified =
      (new_setting.clear_log_frequency_
           ? false
           : (((current_setting != nullptr) &&
               current_setting->log_frequency_specified_) ||
              (new_setting.log_frequency_ != nullptr)));
  const bool filepath_specified =
      (((current_setting != nullptr) && current_setting->filepath_specified_));

  if (level_specified) {
    level = (new_setting.level_ != nullptr) ? *new_setting.level_
                                            : current_setting->level_;
  }
  if (rate_specified) {
    rate = (new_setting.rate_ != nullptr) ? *new_setting.rate_
                                          : current_setting->rate_;
  }
  if (count_specified) {
    count = (new_setting.count_ != nullptr) ? *new_setting.count_
                                            : current_setting->count_;
  }
  if (log_frequency_specified) {
    log_frequency = (new_setting.log_frequency_ != nullptr)
                        ? *new_setting.log_frequency_
                        : current_setting->log_frequency_;
  }
  if (filepath_specified) {
    filepath = current_setting->file_->FileName();
  }

  // Some special case when updating model setting
  if (!model_name.empty()) {
    bool all_specified =
        (level_specified & rate_specified & count_specified &
         log_frequency_specified & filepath_specified);
    bool none_specified =
        !(level_specified | rate_specified | count_specified |
          log_frequency_specified | filepath_specified);
    if (all_specified) {
      fallback_used_models_.erase(model_name);
    } else if (none_specified) {
      // Simply let the model uses global setting
      std::lock_guard<std::mutex> r_lk(r_mu_);
      model_settings_.erase(model_name);
      return nullptr;
    } else {
      fallback_used_models_.emplace(model_name);
    }
  }

  // Create TraceSetting object with the updated setting
  std::shared_ptr<TraceFile> file;
  const auto it = trace_files_.find(filepath);
  if (it != trace_files_.end()) {
    file = it->second.lock();
    // The TraceFile object is no longer valid
    if (file == nullptr) {
      trace_files_.erase(it);
    }
  }
  if (file == nullptr) {
    file.reset(new TraceFile(filepath));
    trace_files_.emplace(filepath, file);
  }

  std::shared_ptr<TraceSetting> lts(new TraceSetting(
      level, rate, count, log_frequency, file, mode, config_map,
      level_specified, rate_specified, count_specified, log_frequency_specified,
      filepath_specified, false /*mode_specified*/,
      false /*config_map_specified*/));
  // The only invalid setting allowed is if it disables tracing
  if ((!lts->Valid()) && (level != TRITONSERVER_TRACE_LEVEL_DISABLED)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("Attempting to set invalid trace setting :") +
         lts->Reason())
            .c_str());
  }

  // Update / Init the setting in read lock to exclude reader access,
  // we replace the object instead of modifying the existing object in case
  // of there are ongoing traces. This makes sure those traces are referring
  // to the setting when the traces are sampled.
  {
    std::lock_guard<std::mutex> r_lk(r_mu_);
    if (model_name.empty()) {
      // global update
      global_setting_ = std::move(lts);
    } else {
      auto it = model_settings_.find(model_name);
      if (it != model_settings_.end()) {
        // Model update
        it->second = std::move(lts);
      } else {
        // Model init
        model_settings_.emplace(model_name, lts);
      }
    }
  }

  return nullptr;
}

void
TraceManager::GetTraceSetting(
    const std::string& model_name, TRITONSERVER_InferenceTraceLevel* level,
    uint32_t* rate, int32_t* count, uint32_t* log_frequency,
    std::string* filepath, InferenceTraceMode* trace_mode,
    TraceConfigMap* config_map)
{
  std::shared_ptr<TraceSetting> trace_setting;
  {
    std::lock_guard<std::mutex> r_lk(r_mu_);
    auto m_it = model_settings_.find(model_name);
    trace_setting =
        (m_it == model_settings_.end()) ? global_setting_ : m_it->second;
  }

  *level = trace_setting->level_;
  *rate = trace_setting->rate_;
  *count = trace_setting->count_;
  *log_frequency = trace_setting->log_frequency_;
  *filepath = trace_setting->file_->FileName();
  *trace_mode = trace_setting->mode_;
  *config_map = trace_setting->config_map_;
}

void
TraceManager::GetTraceSetting(
    const std::string& model_name, std::shared_ptr<TraceSetting>& trace_setting)
{
  std::lock_guard<std::mutex> r_lk(r_mu_);
  auto m_it = model_settings_.find(model_name);
  trace_setting =
      (m_it == model_settings_.end()) ? global_setting_ : m_it->second;
}

TraceManager::TraceStartOptions
TraceManager::GetTraceStartOptions(
    AbstractCarrier& carrier, const std::string& model_name)
{
  TraceManager::TraceStartOptions start_options;
  GetTraceSetting(model_name, start_options.trace_setting);
  if (!start_options.trace_setting->level_ ==
          TRITONSERVER_TRACE_LEVEL_DISABLED &&
      start_options.trace_setting->mode_ == TRACE_MODE_OPENTELEMETRY) {
#ifndef _WIN32
    auto prop =
        otel_cntxt::propagation::GlobalTextMapPropagator::GetGlobalPropagator();
    auto ctxt = otel_cntxt::Context();
    ctxt = prop->Extract(carrier, ctxt);
    otel_trace_api::SpanContext span_context =
        otel_trace_api::GetSpan(ctxt)->GetContext();
    if (span_context.IsValid()) {
      start_options.propagated_context = ctxt;
      start_options.force_sample = true;
    }
#else
    LOG_ERROR << "Unsupported trace mode: "
              << TraceManager::InferenceTraceModeString(
                     start_options.trace_setting->mode_);
#endif  // _WIN32
  }
  return start_options;
}


std::shared_ptr<TraceManager::Trace>
TraceManager::SampleTrace(const TraceStartOptions& start_options)
{
  std::shared_ptr<Trace> ts =
      start_options.trace_setting->SampleTrace(start_options.force_sample);
  if (ts != nullptr) {
    ts->setting_ = start_options.trace_setting;
    if (ts->setting_->mode_ == TRACE_MODE_OPENTELEMETRY) {
#ifndef _WIN32
      auto steady_timestamp_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count();
      if (ts->span_stacks_.find(ts->trace_id_) == ts->span_stacks_.end()) {
        std::unique_ptr<
            std::stack<opentelemetry::nostd::shared_ptr<otel_trace_api::Span>>>
            st(new std::stack<
                opentelemetry::nostd::shared_ptr<otel_trace_api::Span>>());
        ts->span_stacks_.emplace(ts->trace_id_, std::move(st));
      }
      auto active_span =
          otel_trace_api::GetSpan(start_options.propagated_context);
      if (active_span->GetContext().IsValid()) {
        ts->span_stacks_[ts->trace_id_]->emplace(active_span);
      }
      // Storing "InferRequest" span as a root span
      // to keep it alive for the duration of the request.
      ts->root_span_ =
          ts->StartSpan("InferRequest", steady_timestamp_ns, ts->trace_id_);
      ts->span_stacks_[ts->trace_id_]->emplace(ts->root_span_);
#else
      LOG_ERROR << "Unsupported trace mode: "
                << TraceManager::InferenceTraceModeString(ts->setting_->mode_);
#endif
    }
  }
  return ts;
}

TraceManager::Trace::~Trace()
{
  if (setting_->mode_ == TRACE_MODE_TRITON) {
    // Write trace now
    setting_->WriteTrace(streams_);
  } else if (setting_->mode_ == TRACE_MODE_OPENTELEMETRY) {
#ifndef _WIN32
    EndSpan(trace_id_);
#else
    LOG_ERROR << "Unsupported trace mode: "
              << TraceManager::InferenceTraceModeString(setting_->mode_);
#endif
  }
}

void
TraceManager::Trace::CaptureTimestamp(
    const std::string& name, uint64_t timestamp_ns)
{
  if (setting_->level_ & TRITONSERVER_TRACE_LEVEL_TIMESTAMPS) {
    if (setting_->mode_ == TRACE_MODE_TRITON) {
      std::lock_guard<std::mutex> lk(mtx_);
      std::stringstream* ss = nullptr;
      {
        if (streams_.find(trace_id_) == streams_.end()) {
          std::unique_ptr<std::stringstream> stream(new std::stringstream());
          ss = stream.get();
          streams_.emplace(trace_id_, std::move(stream));
        } else {
          ss = streams_[trace_id_].get();
          // If the string stream is not newly created, add "," as there is
          // already content in the string stream
          *ss << ",";
        }
      }
      *ss << "{\"id\":" << trace_id_ << ",\"timestamps\":["
          << "{\"name\":\"" << name << "\",\"ns\":" << timestamp_ns << "}]}";
    } else if (setting_->mode_ == TRACE_MODE_OPENTELEMETRY) {
#ifndef _WIN32
      root_span_->AddEvent(
          name, time_offset_ + std::chrono::nanoseconds{timestamp_ns});
#else
      LOG_ERROR << "Unsupported trace mode: "
                << TraceManager::InferenceTraceModeString(setting_->mode_);
#endif
    }
  }
}

std::string
TraceManager::Trace::RetrieveActivityName(
    TRITONSERVER_InferenceTrace* trace,
    TRITONSERVER_InferenceTraceActivity activity, uint64_t timestamp_ns)
{
  std::string activity_name =
      TRITONSERVER_InferenceTraceActivityString(activity);

  if (activity == TRITONSERVER_TRACE_CUSTOM_ACTIVITY) {
    const char* val = nullptr;
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceTraceContext(trace, &val),
        "Failed to retrieve trace context");
    std::string context_str = (val != nullptr) ? std::string(val) : "";
    triton::common::TritonJson::Value context;
    LOG_TRITONSERVER_ERROR(
        context.Parse(context_str), "Failed to parse trace context");
    std::string look_for_key = std::to_string(timestamp_ns);
    if (context.Find(look_for_key.c_str())) {
      context.MemberAsString(look_for_key.c_str(), &activity_name);
    }
  }

  return activity_name;
}

void
TraceManager::InitTracer(const triton::server::TraceConfigMap& config_map)
{
  switch (global_setting_->mode_) {
    case TRACE_MODE_OPENTELEMETRY: {
#ifndef _WIN32
      otlp::OtlpHttpExporterOptions exporter_options;
      otel_resource::ResourceAttributes attributes = {};
      otel_trace_sdk::BatchSpanProcessorOptions processor_options;

      ProcessOpenTelemetryParameters(
          config_map, exporter_options, attributes, processor_options);

      auto exporter = otlp::OtlpHttpExporterFactory::Create(exporter_options);
      auto processor = otel_trace_sdk::BatchSpanProcessorFactory::Create(
          std::move(exporter), processor_options);
      auto resource = otel_resource::Resource::Create(attributes);
      std::shared_ptr<otel_trace_api::TracerProvider> provider =
          otel_trace_sdk::TracerProviderFactory::Create(
              std::move(processor), resource);

      otel_trace_api::Provider::SetTracerProvider(provider);
      otel_cntxt::propagation::GlobalTextMapPropagator::SetGlobalPropagator(
          opentelemetry::nostd::shared_ptr<
              otel_cntxt::propagation::TextMapPropagator>(
              new otel_trace_api::propagation::HttpTraceContext()));
      break;
#else
      LOG_ERROR << "Unsupported trace mode: "
                << TraceManager::InferenceTraceModeString(
                       global_setting_->mode_);
      break;
#endif
    }
    default:
      return;
  }
}

void
TraceManager::CleanupTracer()
{
  switch (global_setting_->mode_) {
    case TRACE_MODE_OPENTELEMETRY: {
#ifndef _WIN32
      std::shared_ptr<otel_trace_api::TracerProvider> none;
      otel_trace_api::Provider::SetTracerProvider(none);
      break;
#else
      LOG_ERROR << "Unsupported trace mode: "
                << TraceManager::InferenceTraceModeString(
                       global_setting_->mode_);
      break;
#endif
    }
    default:
      return;
  }
}

#ifndef _WIN32
void
TraceManager::ProcessOpenTelemetryParameters(
    const triton::server::TraceConfigMap& config_map,
    otlp::OtlpHttpExporterOptions& exporter_options,
    otel_resource::ResourceAttributes& attributes,
    otel_trace_sdk::BatchSpanProcessorOptions& processor_options)
{
  attributes[otel_resource::SemanticConventions::kServiceName] =
      std::string("triton-inference-server");
  auto mode_key = std::to_string(TRACE_MODE_OPENTELEMETRY);
  auto otel_options_it = config_map.find(mode_key);
  if (otel_options_it == config_map.end()) {
    return;
  }
  for (const auto& [setting, value] : otel_options_it->second) {
    // FIXME add more configuration options of OTLP HTTP Exporter
    if (setting == "url") {
      exporter_options.url = std::get<std::string>(value);
    }
    if (setting == "resource") {
      auto user_setting = std::get<std::string>(value);
      auto pos = user_setting.find('=');
      auto key = user_setting.substr(0, pos);
      auto value = user_setting.substr(pos + 1);
      attributes[key] = value;
    }
    if (setting == "bsp_max_queue_size") {
      processor_options.max_queue_size = std::get<uint32_t>(value);
    }
    if (setting == "bsp_schedule_delay") {
      processor_options.schedule_delay_millis =
          std::chrono::milliseconds(std::get<uint32_t>(value));
    }
    if (setting == "bsp_max_export_batch_size") {
      processor_options.max_export_batch_size = std::get<uint32_t>(value);
    }
  }
}

void
TraceManager::Trace::StartSpan(
    TRITONSERVER_InferenceTrace* trace,
    TRITONSERVER_InferenceTraceActivity activity, uint64_t timestamp_ns,
    uint64_t trace_id, std::string display_name)
{
  uint64_t parent_id;
  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceTraceParentId(trace, &parent_id),
      "getting trace parent id");
  auto span_parent_id = parent_id;

  // Currently, only 2 types of sub-spans are supported:
  // request span and compute span. Compute span is a leaf span
  // and can not be a parent of any sub-span. If parent_id==0,
  // then current model is either a standalone model, or an ensemble model.
  // In both cases, the parent of the new request sub-span is the kRootSpan.
  // A request span with trace id = `trace_id` is a parent of a compute span,
  // started in the same trace.
  // If parent_id > 0, then this is a child trace, spawned from
  // the ensamble's main request. For this instance, the parent
  // span is the ensembles's request span.
  if ((parent_id == 0 && activity == TRITONSERVER_TRACE_REQUEST_START) ||
      (activity == TRITONSERVER_TRACE_COMPUTE_START) ||
      (activity == TRITONSERVER_TRACE_CUSTOM_ACTIVITY)) {
    span_parent_id = trace_id;
  }
  auto span = StartSpan(display_name, timestamp_ns, span_parent_id);

  if (activity == TRITONSERVER_TRACE_REQUEST_START) {
    int64_t model_version;
    const char* request_id;
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceTraceModelVersion(trace, &model_version),
        "getting model version");
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceTraceRequestId(trace, &request_id),
        "getting request id");
    span->SetAttribute("triton.model_name", display_name);
    span->SetAttribute("triton.model_version", model_version);
    span->SetAttribute("triton.trace_id", trace_id);
    span->SetAttribute("triton.trace_parent_id", parent_id);
    if (std::string(request_id) != "") {
      span->SetAttribute("triton.request_id", request_id);
    }
    triton::common::TritonJson::WriteBuffer buffer;
    PrepareTraceContext(span, &buffer);
    TRITONSERVER_InferenceTraceSetContext(trace, buffer.Contents().c_str());
  }
  span_stacks_[trace_id]->emplace(span);
}

opentelemetry::nostd::shared_ptr<otel_trace_api::Span>
TraceManager::Trace::StartSpan(
    std::string display_name, const uint64_t& raw_timestamp_ns,
    uint64_t trace_id)
{
  otel_trace_api::StartSpanOptions options;
  options.kind = otel_trace_api::SpanKind::kServer;
  options.start_system_time =
      time_offset_ + std::chrono::nanoseconds{raw_timestamp_ns};
  options.start_steady_time =
      otel_common::SteadyTimestamp{std::chrono::nanoseconds{raw_timestamp_ns}};

  // If the new span is a child span, we need to retrieve its parent and
  // provide it through StartSpanOptions to the child span
  if (span_stacks_.find(trace_id) != span_stacks_.end() &&
      !span_stacks_[trace_id]->empty()) {
    options.parent = span_stacks_[trace_id]->top()->GetContext();
  }
  auto provider = opentelemetry::trace::Provider::GetTracerProvider();
  return provider->GetTracer(kTritonTracer)->StartSpan(display_name, options);
}

void
TraceManager::Trace::EndSpan(uint64_t trace_id)
{
  auto timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();
  EndSpan(timestamp_ns, trace_id);
}


void
TraceManager::Trace::EndSpan(
    const uint64_t& raw_timestamp_ns, uint64_t trace_id)
{
  if (span_stacks_.find(trace_id) != span_stacks_.end() &&
      !span_stacks_[trace_id]->empty()) {
    otel_trace_api::EndSpanOptions end_options;
    end_options.end_steady_time = otel_common::SteadyTimestamp{
        std::chrono::nanoseconds{raw_timestamp_ns}};
    span_stacks_[trace_id]->top()->End(end_options);
    span_stacks_[trace_id]->pop();
  }
}

void
TraceManager::Trace::ReportToOpenTelemetry(
    TRITONSERVER_InferenceTrace* trace,
    TRITONSERVER_InferenceTraceActivity activity, uint64_t timestamp_ns)
{
  uint64_t id;
  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceTraceId(trace, &id), "getting trace id");
  if (span_stacks_.find(id) == span_stacks_.end()) {
    std::unique_ptr<
        std::stack<opentelemetry::nostd::shared_ptr<otel_trace_api::Span>>>
        st(new std::stack<
            opentelemetry::nostd::shared_ptr<otel_trace_api::Span>>());
    span_stacks_.emplace(id, std::move(st));
  }

  AddEvent(trace, activity, timestamp_ns, id);
}

void
TraceManager::Trace::AddEvent(
    TRITONSERVER_InferenceTrace* trace,
    TRITONSERVER_InferenceTraceActivity activity, uint64_t timestamp_ns,
    uint64_t trace_id)
{
  std::string activity_name =
      RetrieveActivityName(trace, activity, timestamp_ns);
  static std::string start = "_START";
  static std::string end = "_END";
  if (activity == TRITONSERVER_TRACE_REQUEST_START ||
      activity == TRITONSERVER_TRACE_COMPUTE_START ||
      (activity == TRITONSERVER_TRACE_CUSTOM_ACTIVITY &&
       activity_name.length() > start.length() &&
       std::equal(start.rbegin(), start.rend(), activity_name.rbegin()))) {
    std::string span_name = activity_name;

    if (activity == TRITONSERVER_TRACE_CUSTOM_ACTIVITY) {
      span_name =
          activity_name.substr(0, activity_name.length() - start.length());
    } else if (activity == TRITONSERVER_TRACE_REQUEST_START) {
      const char* model_name;
      LOG_TRITONSERVER_ERROR(
          TRITONSERVER_InferenceTraceModelName(trace, &model_name),
          "getting model name");
      span_name = model_name;
    } else if (activity == TRITONSERVER_TRACE_COMPUTE_START) {
      span_name = "compute";
    }

    StartSpan(trace, activity, timestamp_ns, trace_id, span_name);
  }

  AddEvent(activity_name, timestamp_ns, trace_id);

  if (activity == TRITONSERVER_TRACE_REQUEST_END ||
      activity == TRITONSERVER_TRACE_COMPUTE_END ||
      (activity == TRITONSERVER_TRACE_CUSTOM_ACTIVITY &&
       activity_name.length() > end.length() &&
       std::equal(end.rbegin(), end.rend(), activity_name.rbegin()))) {
    EndSpan(timestamp_ns, trace_id);
  }
}

void
TraceManager::Trace::AddEvent(
    const std::string& event, uint64_t timestamp, uint64_t trace_id)
{
  if (span_stacks_.find(trace_id) != span_stacks_.end() &&
      !span_stacks_[trace_id]->empty()) {
    span_stacks_[trace_id]->top()->AddEvent(
        event, time_offset_ + std::chrono::nanoseconds{timestamp});
  }
}

void
TraceManager::Trace::PrepareTraceContext(
    opentelemetry::nostd::shared_ptr<otel_trace_api::Span> span,
    triton::common::TritonJson::WriteBuffer* buffer)
{
  triton::common::TritonJson::Value json(
      triton::common::TritonJson::ValueType::OBJECT);
  char trace_id[32] = {0};
  char span_id[16] = {0};
  char trace_flags[2] = {0};
  span->GetContext().span_id().ToLowerBase16(span_id);
  span->GetContext().trace_id().ToLowerBase16(trace_id);
  span->GetContext().trace_flags().ToLowerBase16(trace_flags);
  std::string kTraceParent = std::string("traceparent");
  std::string kTraceState = std::string("tracestate");
  std::string traceparent = std::string("00-") + std::string(trace_id, 32) +
                            std::string("-") + std::string(span_id, 16) +
                            std::string("-") + std::string(trace_flags, 2);
  std::string tracestate = span->GetContext().trace_state()->ToHeader();
  json.SetStringObject(kTraceParent.c_str(), traceparent);
  if (!tracestate.empty()) {
    json.SetStringObject(kTraceState.c_str(), tracestate);
  }
  json.Write(buffer);
}
#endif

void
TraceManager::TraceRelease(TRITONSERVER_InferenceTrace* trace, void* userp)
{
  uint64_t id;
  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceTraceId(trace, &id), "getting trace id");

  auto ts = reinterpret_cast<std::shared_ptr<TraceManager::Trace>*>(userp);
  std::lock_guard<std::mutex> lk((*ts)->mtx_);
  (*ts)->spawned_traces_tracker_.erase(id);
  // The userp will be shared with the trace children, so only delete it
  // if no more TraceRelease calls are expected
  if ((*ts)->spawned_traces_tracker_.empty()) {
    delete ts;
  }
  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceTraceDelete(trace), "deleting trace");
}

const char*
TraceManager::InferenceTraceModeString(InferenceTraceMode mode)
{
  switch (mode) {
    case TRACE_MODE_TRITON:
      return "triton";
    case TRACE_MODE_OPENTELEMETRY:
      return "opentelemetry";
  }

  return "<unknown>";
}

void
TraceManager::TraceActivity(
    TRITONSERVER_InferenceTrace* trace,
    TRITONSERVER_InferenceTraceActivity activity, uint64_t timestamp_ns,
    void* userp)
{
  uint64_t id;
  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceTraceId(trace, &id), "getting trace id");

  // The function may be called with different traces but the same 'userp',
  // group the activity of the same trace together for more readable output.
  auto ts =
      reinterpret_cast<std::shared_ptr<TraceManager::Trace>*>(userp)->get();

  std::lock_guard<std::mutex> lk(ts->mtx_);
  if (ts->spawned_traces_tracker_.find(id) ==
      ts->spawned_traces_tracker_.end()) {
    ts->spawned_traces_tracker_.emplace(id);
  }

  if (ts->setting_->mode_ == TRACE_MODE_OPENTELEMETRY) {
#ifndef _WIN32
    ts->ReportToOpenTelemetry(trace, activity, timestamp_ns);
#else
    LOG_ERROR << "Unsupported trace mode: "
              << TraceManager::InferenceTraceModeString(ts->setting_->mode_);
#endif
    return;
  }
  std::stringstream* ss = nullptr;
  {
    if (ts->streams_.find(id) == ts->streams_.end()) {
      std::unique_ptr<std::stringstream> stream(new std::stringstream());
      ss = stream.get();
      ts->streams_.emplace(id, std::move(stream));
    } else {
      ss = ts->streams_[id].get();
      // If the string stream is not newly created, add "," as there is
      // already content in the string stream
      *ss << ",";
    }
  }
  // If 'activity' is TRITONSERVER_TRACE_REQUEST_START then collect
  // and serialize trace details.
  if (activity == TRITONSERVER_TRACE_REQUEST_START) {
    const char* model_name;
    int64_t model_version;
    uint64_t parent_id;
    const char* request_id;

    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceTraceModelName(trace, &model_name),
        "getting model name");
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceTraceModelVersion(trace, &model_version),
        "getting model version");
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceTraceParentId(trace, &parent_id),
        "getting trace parent id");
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceTraceRequestId(trace, &request_id),
        "getting request id");

    *ss << "{\"id\":" << id << ",\"model_name\":\"" << model_name
        << "\",\"model_version\":" << model_version;

    if (std::string(request_id) != "") {
      *ss << ",\"request_id\":\"" << request_id << "\"";
    }

    if (parent_id != 0) {
      *ss << ",\"parent_id\":" << parent_id;
    }
    *ss << "},";
  }

  *ss << "{\"id\":" << id << ",\"timestamps\":["
      << "{\"name\":\""
      << ts->RetrieveActivityName(trace, activity, timestamp_ns)
      << "\",\"ns\":" << timestamp_ns << "}]}";
}

void
TraceManager::TraceTensorActivity(
    TRITONSERVER_InferenceTrace* trace,
    TRITONSERVER_InferenceTraceActivity activity, const char* name,
    TRITONSERVER_DataType datatype, const void* base, size_t byte_size,
    const int64_t* shape, uint64_t dim_count,
    TRITONSERVER_MemoryType memory_type, int64_t memory_type_id, void* userp)
{
  if ((activity != TRITONSERVER_TRACE_TENSOR_QUEUE_INPUT) &&
      (activity != TRITONSERVER_TRACE_TENSOR_BACKEND_INPUT) &&
      (activity != TRITONSERVER_TRACE_TENSOR_BACKEND_OUTPUT)) {
    LOG_ERROR << "Unsupported activity: "
              << TRITONSERVER_InferenceTraceActivityString(activity);
    return;
  }

  void* buffer_base = const_cast<void*>(base);
  if (memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
    buffer_base = malloc(byte_size);
    if (buffer_base == nullptr) {
      LOG_ERROR << "Failed to malloc CPU buffer";
      return;
    }
    FAIL_IF_CUDA_ERR(
        cudaMemcpy(buffer_base, base, byte_size, cudaMemcpyDeviceToHost),
        "copying buffer into CPU memory");
#else
    LOG_ERROR << "GPU buffer is unsupported";
    return;
#endif  // TRITON_ENABLE_GPU
  }

  uint64_t id;
  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceTraceId(trace, &id), "getting trace id");

  // The function may be called with different traces but the same 'userp',
  // group the activity of the same trace together for more readable output.
  auto ts =
      reinterpret_cast<std::shared_ptr<TraceManager::Trace>*>(userp)->get();

  if (ts->setting_->mode_ == TRACE_MODE_OPENTELEMETRY) {
    LOG_ERROR << "Tensor level tracing is not supported by the mode: "
              << TraceManager::InferenceTraceModeString(ts->setting_->mode_);
  } else if (ts->setting_->mode_ == TRACE_MODE_TRITON) {
    std::lock_guard<std::mutex> lk(ts->mtx_);
    std::stringstream* ss = nullptr;
    {
      if (ts->streams_.find(id) == ts->streams_.end()) {
        std::unique_ptr<std::stringstream> stream(new std::stringstream());
        ss = stream.get();
        ts->streams_.emplace(id, std::move(stream));
        ts->spawned_traces_tracker_.emplace(id);
      } else {
        ss = ts->streams_[id].get();
        // If the string stream is not newly created, add "," as there is
        // already content in the string stream
        *ss << ",";
      }
    }

    // collect and serialize trace details.
    *ss << "{\"id\":" << id << ",\"activity\":\""
        << TRITONSERVER_InferenceTraceActivityString(activity) << "\"";
    // collect tensor
    *ss << ",\"tensor\":{";
    // collect tensor name
    *ss << "\"name\":\"" << std::string(name) << "\"";
    // collect tensor data
    *ss << ",\"data\":\"";
    size_t element_count = 1;
    for (uint64_t i = 0; i < dim_count; i++) {
      element_count *= shape[i];
    }
    switch (datatype) {
      case TRITONSERVER_TYPE_BOOL: {
        const uint8_t* bool_base =
            reinterpret_cast<const uint8_t*>(buffer_base);
        for (size_t e = 0; e < element_count; ++e) {
          *ss << ((bool_base[e] == 0) ? false : true);
          if (e < (element_count - 1))
            *ss << ",";
        }
        break;
      }
      case TRITONSERVER_TYPE_UINT8: {
        const uint8_t* cbase = reinterpret_cast<const uint8_t*>(buffer_base);
        for (size_t e = 0; e < element_count; ++e) {
          *ss << cbase[e];
          if (e < (element_count - 1))
            *ss << ",";
        }
        break;
      }
      case TRITONSERVER_TYPE_UINT16: {
        const uint16_t* cbase = reinterpret_cast<const uint16_t*>(buffer_base);
        for (size_t e = 0; e < element_count; ++e) {
          *ss << cbase[e];
          if (e < (element_count - 1))
            *ss << ",";
        }
        break;
      }
      case TRITONSERVER_TYPE_UINT32: {
        const uint32_t* cbase = reinterpret_cast<const uint32_t*>(buffer_base);
        for (size_t e = 0; e < element_count; ++e) {
          *ss << cbase[e];
          if (e < (element_count - 1))
            *ss << ",";
        }
        break;
      }
      case TRITONSERVER_TYPE_UINT64: {
        const uint64_t* cbase = reinterpret_cast<const uint64_t*>(buffer_base);
        for (size_t e = 0; e < element_count; ++e) {
          *ss << cbase[e];
          if (e < (element_count - 1))
            *ss << ",";
        }
        break;
      }
      case TRITONSERVER_TYPE_INT8: {
        const int8_t* cbase = reinterpret_cast<const int8_t*>(buffer_base);
        for (size_t e = 0; e < element_count; ++e) {
          *ss << cbase[e];
          if (e < (element_count - 1))
            *ss << ",";
        }
        break;
      }
      case TRITONSERVER_TYPE_INT16: {
        const int16_t* cbase = reinterpret_cast<const int16_t*>(buffer_base);
        for (size_t e = 0; e < element_count; ++e) {
          *ss << cbase[e];
          if (e < (element_count - 1))
            *ss << ",";
        }
        break;
      }
      case TRITONSERVER_TYPE_INT32: {
        const int32_t* cbase = reinterpret_cast<const int32_t*>(buffer_base);
        for (size_t e = 0; e < element_count; ++e) {
          *ss << cbase[e];
          if (e < (element_count - 1))
            *ss << ",";
        }
        break;
      }
      case TRITONSERVER_TYPE_INT64: {
        const int64_t* cbase = reinterpret_cast<const int64_t*>(buffer_base);
        for (size_t e = 0; e < element_count; ++e) {
          *ss << cbase[e];
          if (e < (element_count - 1))
            *ss << ",";
        }
        break;
      }

      // FP16 / BF16 already handled as binary blobs, no need to manipulate
      // here
      case TRITONSERVER_TYPE_FP16: {
        break;
      }
      case TRITONSERVER_TYPE_BF16: {
        break;
      }

      case TRITONSERVER_TYPE_FP32: {
        const float* cbase = reinterpret_cast<const float*>(buffer_base);
        for (size_t e = 0; e < element_count; ++e) {
          *ss << cbase[e];
          if (e < (element_count - 1))
            *ss << ",";
        }
        break;
      }
      case TRITONSERVER_TYPE_FP64: {
        const double* cbase = reinterpret_cast<const double*>(buffer_base);
        for (size_t e = 0; e < element_count; ++e) {
          *ss << cbase[e];
          if (e < (element_count - 1))
            *ss << ",";
        }
        break;
      }
      case TRITONSERVER_TYPE_BYTES: {
        const char* cbase = reinterpret_cast<const char*>(buffer_base);
        size_t offset = 0;
        for (size_t e = 0; e < element_count; ++e) {
          if ((offset + sizeof(uint32_t)) > byte_size) {
            return;
          }
          const size_t len =
              *(reinterpret_cast<const uint32_t*>(cbase + offset));
          offset += sizeof(uint32_t);
          if ((offset + len) > byte_size) {
            return;
          }
          std::string str(cbase + offset, len);
          *ss << "\\\"" << str << "\\\"";
          offset += len;

          if (e < (element_count - 1))
            *ss << ",";
        }
        break;
      }
      case TRITONSERVER_TYPE_INVALID: {
        return;
      }
    }
    *ss << "\",\"shape\":\"";
    for (uint64_t i = 0; i < dim_count; i++) {
      *ss << shape[i];
      if (i < (dim_count - 1)) {
        *ss << ",";
      }
    }
    *ss << "\",\"dtype\":\"" << TRITONSERVER_DataTypeString(datatype) << "\"}";
    *ss << "}";
  }

  if (memory_type == TRITONSERVER_MEMORY_GPU) {
#ifdef TRITON_ENABLE_GPU
    if (buffer_base != nullptr) {
      free(buffer_base);
    }
#endif  // TRITON_ENABLE_GPU
  }
}

TraceManager::TraceFile::~TraceFile()
{
  if (!first_write_) {
    trace_file_ << "]";
  }
}

void
TraceManager::TraceFile::SaveTraces(
    std::stringstream& trace_stream, const bool to_index_file)
{
  try {
    if (to_index_file) {
      std::string file_name =
          file_name_ + "." + std::to_string(index_.fetch_add(1));
      std::ofstream file_stream;
      file_stream.open(file_name);
      file_stream << "[";
      file_stream << trace_stream.rdbuf();
      file_stream << "]";
    } else {
      std::lock_guard<std::mutex> lock(mu_);
      if (first_write_) {
        trace_file_.open(file_name_);
        trace_file_ << "[";
        first_write_ = false;
      } else {
        trace_file_ << ",";
      }
      trace_file_ << trace_stream.rdbuf();
    }
  }
  catch (const std::ofstream::failure& e) {
    LOG_ERROR << "failed creating trace file: " << e.what();
  }
  catch (...) {
    LOG_ERROR << "failed creating trace file: reason unknown";
  }
}

std::shared_ptr<TraceManager::Trace>
TraceManager::TraceSetting::SampleTrace(bool force_sample)
{
  bool count_rate_hit = false;
  {
    std::lock_guard<std::mutex> lk(mu_);
    // [FIXME: DLIS-6033]
    // A current WAR for initiating trace based on propagated context only
    // Currently this is implemented through setting trace rate as 0
    if (rate_ != 0) {
      // If `count_` hits 0, `Valid()` returns false for this and all
      // following requests (unless `count_` is updated by a user).
      // At this point we only trace requests for which
      // `force_sample` is true.
      if (!Valid() && !force_sample) {
        return nullptr;
      }
      // `sample_` counts all requests, coming to server.
      count_rate_hit = (((++sample_) % rate_) == 0);
      if (count_rate_hit && (count_ > 0)) {
        --count_;
        ++created_;
      } else if (count_rate_hit && (count_ == 0)) {
        // This condition is reached, when `force_sample` is true,
        // `count_rate_hit` is true, but `count_` is 0. Due to the
        // latter, we explicitly set `count_rate_hit` to false.
        count_rate_hit = false;
      }
    }
  }
  if (count_rate_hit || force_sample) {
    std::shared_ptr<TraceManager::Trace> lts(new Trace());
    // Split 'Trace' management to frontend and Triton trace separately
    // to avoid dependency between frontend request and Triton trace's
    // liveness
    auto trace_userp = new std::shared_ptr<TraceManager::Trace>(lts);
    TRITONSERVER_InferenceTrace* trace;
    TRITONSERVER_Error* err = TRITONSERVER_InferenceTraceTensorNew(
        &trace, level_, 0 /* parent_id */, TraceActivity, TraceTensorActivity,
        TraceRelease, trace_userp);
    if (err != nullptr) {
      LOG_TRITONSERVER_ERROR(err, "creating inference trace object");
      delete trace_userp;
      return nullptr;
    }
    lts->trace_ = trace;
    lts->trace_userp_ = trace_userp;
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceTraceId(trace, &lts->trace_id_),
        "getting trace id");
    return lts;
  }
  return nullptr;
}

void
TraceManager::TraceSetting::WriteTrace(
    const std::unordered_map<uint64_t, std::unique_ptr<std::stringstream>>&
        streams)
{
  std::unique_lock<std::mutex> lock(mu_);

  if (sample_in_stream_ != 0) {
    trace_stream_ << ",";
  }
  ++sample_in_stream_;
  ++collected_;

  size_t stream_count = 0;
  for (const auto& stream : streams) {
    trace_stream_ << stream.second->rdbuf();
    // Need to add ',' unless it is the last trace in the group
    ++stream_count;
    if (stream_count != streams.size()) {
      trace_stream_ << ",";
    }
  }
  // Write to file with index when one of the following is true
  // 1. trace_count is specified and that number of traces has been collected
  // 2. log_frequency is specified and that number of traces has been
  // collected
  if (((count_ == 0) && (collected_ == sample_)) ||
      ((log_frequency_ != 0) && (sample_in_stream_ >= log_frequency_))) {
    // Reset variables and release lock before saving to file
    sample_in_stream_ = 0;
    std::stringstream stream;
    trace_stream_.swap(stream);
    lock.unlock();

    file_->SaveTraces(stream, true /* to_index_file */);
  }
}

TraceManager::TraceSetting::TraceSetting(
    const TRITONSERVER_InferenceTraceLevel level, const uint32_t rate,
    const int32_t count, const uint32_t log_frequency,
    const std::shared_ptr<TraceFile>& file, const InferenceTraceMode mode,
    const TraceConfigMap& config_map, const bool level_specified,
    const bool rate_specified, const bool count_specified,
    const bool log_frequency_specified, const bool filepath_specified,
    const bool mode_specified, const bool config_map_specified)
    : level_(level), rate_(rate), count_(count), log_frequency_(log_frequency),
      file_(file), mode_(mode), config_map_(config_map),
      level_specified_(level_specified), rate_specified_(rate_specified),
      count_specified_(count_specified),
      log_frequency_specified_(log_frequency_specified),
      filepath_specified_(filepath_specified), mode_specified_(mode_specified),
      config_map_specified_(config_map_specified), sample_(0), created_(0),
      collected_(0), sample_in_stream_(0)
{
  if (level_ == TRITONSERVER_TRACE_LEVEL_DISABLED) {
    invalid_reason_ = "tracing is disabled";
  } else if (rate_ == 0) {
    invalid_reason_ = "sample rate must be non-zero";
  } else if (mode_ == TRACE_MODE_TRITON && file_->FileName().empty()) {
    invalid_reason_ = "trace file name is not given";
  }
}

TraceManager::TraceSetting::~TraceSetting()
{
  // If log frequency is set, should log the remaining traces to indexed file.
  if (mode_ == TRACE_MODE_TRITON && sample_in_stream_ != 0) {
    file_->SaveTraces(trace_stream_, (log_frequency_ != 0));
  }
}
}}  // namespace triton::server
