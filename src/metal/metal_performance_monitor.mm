// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "metal_performance_monitor.h"

#include <IOKit/IOKitLib.h>
#include <IOKit/ps/IOPSKeys.h>
#include <IOKit/ps/IOPowerSources.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

#include "metal_device.h"
#include "metal_memory_manager.h"
#include "triton/core/tritonserver.h"
#include "triton/core/tritonbackend.h"
#include "../../build/triton-server/_deps/repo-core-src/src/status.h"

namespace triton { namespace metal {

using triton::core::Status;

namespace {

// Helper to convert Metal timestamp to nanoseconds
uint64_t MetalTimestampToNs(NSTimeInterval timestamp) {
  return static_cast<uint64_t>(timestamp * 1e9);
}

// Helper to get current system time in nanoseconds
uint64_t GetSystemTimeNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Helper to format metric name for Prometheus
std::string FormatMetricName(const std::string& base_name) {
  std::string formatted = base_name;
  std::replace(formatted.begin(), formatted.end(), ' ', '_');
  std::replace(formatted.begin(), formatted.end(), '-', '_');
  std::transform(formatted.begin(), formatted.end(), formatted.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return "metal_" + formatted;
}

}  // namespace

MetalPerformanceMonitor::MetalPerformanceMonitor(
    std::shared_ptr<MetalDevice> device, const PerformanceMonitorConfig& config)
    : device_(device), config_(config) {
  // Initialize memory statistics
  memory_stats_ = {};
}

MetalPerformanceMonitor::~MetalPerformanceMonitor() {
  if (is_monitoring_) {
    StopMonitoring();
  }
}

triton::core::Status MetalPerformanceMonitor::Initialize() {
  @autoreleasepool {
    std::lock_guard<std::mutex> lock(mutex_);

    // Set up Metal performance counters if available
    if (config_.enable_detailed_profiling) {
      SetupCounterSampleBuffer();
    }

    // Reserve space for metric history
    for (auto type : {MetalCounterType::GPU_UTILIZATION,
                      MetalCounterType::MEMORY_BANDWIDTH,
                      MetalCounterType::KERNEL_EXECUTION_TIME,
                      MetalCounterType::MEMORY_ALLOCATION,
                      MetalCounterType::COMMAND_BUFFER_TIME,
                      MetalCounterType::POWER_CONSUMPTION,
                      MetalCounterType::THERMAL_STATE,
                      MetalCounterType::QUEUE_UTILIZATION}) {
      metric_history_[type].reserve(config_.history_buffer_size);
    }

    return Status::Success;
  }
}

triton::core::Status MetalPerformanceMonitor::StartMonitoring() {
  if (is_monitoring_) {
    return Status(Status::Code::ALREADY_EXISTS,
                  "Performance monitoring is already active");
  }

  is_monitoring_ = true;
  monitoring_thread_ = std::make_unique<std::thread>(
      &MetalPerformanceMonitor::MonitoringThread, this);

  return Status::Success;
}

triton::core::Status MetalPerformanceMonitor::StopMonitoring() {
  if (!is_monitoring_) {
    return Status(Status::Code::UNAVAILABLE,
                  "Performance monitoring is not active");
  }

  is_monitoring_ = false;
  if (monitoring_thread_ && monitoring_thread_->joinable()) {
    monitoring_thread_->join();
  }

  return Status::Success;
}

void MetalPerformanceMonitor::BeginKernelProfiling(
    const std::string& kernel_name, id<MTLComputeCommandEncoder> encoder) {
  if (!config_.enable_kernel_profiling) {
    return;
  }

  @autoreleasepool {
    std::lock_guard<std::mutex> lock(mutex_);

    KernelProfilingData profile;
    profile.kernel_name = kernel_name;
    profile.start_time_ns = GetCurrentTimeNs();
    profile.gpu_start_time_ns = 0;  // Will be set from GPU timestamp

    // Get thread execution details
    // Note: MTLComputeCommandEncoder doesn't expose the pipeline state directly
    // This information would need to be tracked when the encoder is created
    if (encoder) {
      // TODO: Track pipeline state information when encoder is created
      profile.thread_count = 0;
      profile.threadgroup_size = 0;
      profile.shared_memory_bytes = 0;
    }

    active_kernel_profiles_[(__bridge void*)encoder] = profile;

    // Insert debug signpost if available
    if (@available(macOS 10.14, *)) {
      if (encoder && config_.enable_detailed_profiling) {
        [encoder pushDebugGroup:[NSString stringWithUTF8String:kernel_name.c_str()]];
      }
    }
  }
}

void MetalPerformanceMonitor::EndKernelProfiling(
    const std::string& kernel_name, id<MTLComputeCommandEncoder> encoder) {
  if (!config_.enable_kernel_profiling) {
    return;
  }

  @autoreleasepool {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = active_kernel_profiles_.find((__bridge void*)encoder);
    if (it != active_kernel_profiles_.end()) {
      it->second.end_time_ns = GetCurrentTimeNs();

      // Pop debug group if available
      if (@available(macOS 10.14, *)) {
        if (encoder && config_.enable_detailed_profiling) {
          [encoder popDebugGroup];
        }
      }

      // Move to completed profiles
      kernel_profiles_[kernel_name].push_back(it->second);
      active_kernel_profiles_.erase(it);

      // Keep history bounded
      auto& profiles = kernel_profiles_[kernel_name];
      if (profiles.size() > config_.history_buffer_size) {
        profiles.erase(profiles.begin());
      }
    }
  }
}

void MetalPerformanceMonitor::RegisterCommandBuffer(
    const std::string& identifier, id<MTLCommandBuffer> buffer) {
  if (!buffer) {
    return;
  }

  @autoreleasepool {
    std::lock_guard<std::mutex> lock(mutex_);

    CommandBufferProfile profile;
    profile.identifier = identifier;
    profile.submit_time_ns = GetCurrentTimeNs();

    command_buffer_profiles_[(__bridge void*)buffer] = profile;

    // Set up completion handler
    __weak id<MTLCommandBuffer> weakBuffer = buffer;
    [buffer addScheduledHandler:^(id<MTLCommandBuffer> cmdBuffer) {
      OnCommandBufferScheduled(weakBuffer);
    }];

    [buffer addCompletedHandler:^(id<MTLCommandBuffer> cmdBuffer) {
      OnCommandBufferCompleted(weakBuffer);
    }];
  }
}

void MetalPerformanceMonitor::OnCommandBufferScheduled(id<MTLCommandBuffer> buffer) {
  if (!buffer) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = command_buffer_profiles_.find((__bridge void*)buffer);
  if (it != command_buffer_profiles_.end()) {
    it->second.scheduled_time_ns = GetCurrentTimeNs();
    if (@available(macOS 10.15, *)) {
      it->second.kernel_start_time_ns = MetalTimestampToNs([buffer kernelStartTime]);
    }
  }
}

void MetalPerformanceMonitor::OnCommandBufferCompleted(id<MTLCommandBuffer> buffer) {
  if (!buffer) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = command_buffer_profiles_.find((__bridge void*)buffer);
  if (it != command_buffer_profiles_.end()) {
    it->second.completed_time_ns = GetCurrentTimeNs();
    if (@available(macOS 10.15, *)) {
      it->second.kernel_end_time_ns = MetalTimestampToNs([buffer kernelEndTime]);
    }

    // Move to completed list
    completed_command_buffers_.push_back(it->second);
    command_buffer_profiles_.erase(it);

    // Keep history bounded
    if (completed_command_buffers_.size() > config_.history_buffer_size) {
      completed_command_buffers_.erase(completed_command_buffers_.begin());
    }

    // Record command buffer latency metric
    if (it->second.kernel_end_time_ns > it->second.kernel_start_time_ns) {
      MetalPerformanceMetric metric;
      metric.type = MetalCounterType::COMMAND_BUFFER_TIME;
      metric.name = "command_buffer_gpu_time";
      metric.value = static_cast<double>(it->second.kernel_end_time_ns -
                                         it->second.kernel_start_time_ns) / 1e6;  // ms
      metric.timestamp_ns = it->second.completed_time_ns;
      metric.labels["identifier"] = it->second.identifier;
      AddMetricToHistory(metric);
    }
  }
}

void MetalPerformanceMonitor::TrackMemoryAllocation(
    void* ptr, size_t size, bool is_shared, const std::string& context) {
  std::lock_guard<std::mutex> lock(mutex_);

  MemoryAllocationInfo info;
  info.ptr = ptr;
  info.size = size;
  info.is_shared = is_shared;
  info.allocation_time_ns = GetCurrentTimeNs();
  info.deallocation_time_ns = 0;
  info.allocation_context = context;

  memory_allocations_[ptr] = info;

  // Update statistics
  memory_stats_.total_allocated_bytes += size;
  memory_stats_.current_allocated_bytes += size;
  memory_stats_.allocation_count++;
  if (is_shared) {
    memory_stats_.shared_memory_bytes += size;
  } else {
    memory_stats_.device_memory_bytes += size;
  }

  if (memory_stats_.current_allocated_bytes > memory_stats_.peak_allocated_bytes) {
    memory_stats_.peak_allocated_bytes = memory_stats_.current_allocated_bytes;
  }

  // Record allocation metric
  MetalPerformanceMetric metric;
  metric.type = MetalCounterType::MEMORY_ALLOCATION;
  metric.name = "memory_allocated_bytes";
  metric.value = static_cast<double>(memory_stats_.current_allocated_bytes);
  metric.timestamp_ns = GetCurrentTimeNs();
  metric.labels["type"] = is_shared ? "shared" : "device";
  AddMetricToHistory(metric);
}

void MetalPerformanceMonitor::TrackMemoryDeallocation(void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = memory_allocations_.find(ptr);
  if (it != memory_allocations_.end()) {
    it->second.deallocation_time_ns = GetCurrentTimeNs();

    // Update statistics
    memory_stats_.current_allocated_bytes -= it->second.size;
    memory_stats_.deallocation_count++;
    if (it->second.is_shared) {
      memory_stats_.shared_memory_bytes -= it->second.size;
    } else {
      memory_stats_.device_memory_bytes -= it->second.size;
    }

    memory_allocations_.erase(it);

    // Record deallocation metric
    MetalPerformanceMetric metric;
    metric.type = MetalCounterType::MEMORY_ALLOCATION;
    metric.name = "memory_allocated_bytes";
    metric.value = static_cast<double>(memory_stats_.current_allocated_bytes);
    metric.timestamp_ns = GetCurrentTimeNs();
    AddMetricToHistory(metric);
  }
}

std::vector<MetalPerformanceMetric> MetalPerformanceMonitor::GetCurrentMetrics() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<MetalPerformanceMetric> current_metrics;
  for (const auto& [type, history] : metric_history_) {
    if (!history.empty()) {
      current_metrics.push_back(history.back());
    }
  }

  return current_metrics;
}

std::vector<MetalPerformanceMetric> MetalPerformanceMonitor::GetMetricHistory(
    MetalCounterType type, size_t count) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = metric_history_.find(type);
  if (it == metric_history_.end()) {
    return {};
  }

  const auto& history = it->second;
  if (count == 0 || count >= history.size()) {
    return history;
  }

  return std::vector<MetalPerformanceMetric>(
      history.end() - count, history.end());
}

std::vector<KernelProfilingData> MetalPerformanceMonitor::GetKernelProfilingData(
    const std::string& kernel_name) const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<KernelProfilingData> result;
  if (kernel_name.empty()) {
    // Return all kernel profiles
    for (const auto& [name, profiles] : kernel_profiles_) {
      result.insert(result.end(), profiles.begin(), profiles.end());
    }
  } else {
    // Return profiles for specific kernel
    auto it = kernel_profiles_.find(kernel_name);
    if (it != kernel_profiles_.end()) {
      result = it->second;
    }
  }

  return result;
}

std::vector<CommandBufferProfile> MetalPerformanceMonitor::GetCommandBufferProfiles(
    size_t count) const {
  std::lock_guard<std::mutex> lock(mutex_);

  if (count == 0 || count >= completed_command_buffers_.size()) {
    return completed_command_buffers_;
  }

  return std::vector<CommandBufferProfile>(
      completed_command_buffers_.end() - count,
      completed_command_buffers_.end());
}

MetalPerformanceMonitor::MemoryStatistics
MetalPerformanceMonitor::GetMemoryStatistics() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return memory_stats_;
}

MetalPerformanceMonitor::PerformanceAnalysis
MetalPerformanceMonitor::AnalyzePerformance(uint64_t window_duration_ms) const {
  std::lock_guard<std::mutex> lock(mutex_);

  PerformanceAnalysis analysis = {};
  uint64_t current_time_ns = GetCurrentTimeNs();
  uint64_t window_start_ns = window_duration_ms > 0
      ? current_time_ns - (window_duration_ms * 1000000)
      : 0;

  // Analyze GPU utilization
  auto gpu_metrics = GetMetricHistory(MetalCounterType::GPU_UTILIZATION);
  if (!gpu_metrics.empty()) {
    double sum = 0;
    double max_val = 0;
    size_t count = 0;
    for (const auto& metric : gpu_metrics) {
      if (metric.timestamp_ns >= window_start_ns) {
        sum += metric.value;
        max_val = std::max(max_val, metric.value);
        count++;
      }
    }
    if (count > 0) {
      analysis.avg_gpu_utilization = sum / count;
      analysis.peak_gpu_utilization = max_val;
    }
  }

  // Analyze memory bandwidth
  auto bandwidth_metrics = GetMetricHistory(MetalCounterType::MEMORY_BANDWIDTH);
  if (!bandwidth_metrics.empty()) {
    double sum = 0;
    double max_val = 0;
    size_t count = 0;
    for (const auto& metric : bandwidth_metrics) {
      if (metric.timestamp_ns >= window_start_ns) {
        sum += metric.value;
        max_val = std::max(max_val, metric.value);
        count++;
      }
    }
    if (count > 0) {
      analysis.avg_memory_bandwidth_gbps = sum / count;
      analysis.peak_memory_bandwidth_gbps = max_val;
    }
  }

  // Analyze kernel execution times
  for (const auto& [kernel_name, profiles] : kernel_profiles_) {
    double total_time = 0;
    size_t count = 0;
    for (const auto& profile : profiles) {
      if (profile.end_time_ns >= window_start_ns) {
        total_time += (profile.end_time_ns - profile.start_time_ns) / 1e6;  // ms
        count++;
      }
    }
    if (count > 0) {
      double avg_time = total_time / count;
      analysis.kernel_execution_times_ms[kernel_name] = avg_time;
      analysis.avg_kernel_execution_time_ms += avg_time;
    }
  }
  if (!analysis.kernel_execution_times_ms.empty()) {
    analysis.avg_kernel_execution_time_ms /= analysis.kernel_execution_times_ms.size();
  }

  // Identify bottlenecks and generate recommendations
  AnalyzeBottlenecks(analysis);
  GenerateOptimizationRecommendations(analysis);

  return analysis;
}

#ifdef TRITON_ENABLE_METRICS
void MetalPerformanceMonitor::RegisterPrometheusMetrics(
    std::shared_ptr<prometheus::Registry> registry) {
  // GPU utilization metrics
  gpu_utilization_family_ = &prometheus::BuildGauge()
      .Name("metal_gpu_utilization_ratio")
      .Help("GPU utilization percentage (0-1)")
      .Register(*registry);
  gpu_utilization_gauge_ = &gpu_utilization_family_->Add({{"device", device_->GetName()}});

  // Memory bandwidth metrics
  memory_bandwidth_family_ = &prometheus::BuildGauge()
      .Name("metal_memory_bandwidth_gbps")
      .Help("Memory bandwidth utilization in GB/s")
      .Register(*registry);
  memory_bandwidth_gauge_ = &memory_bandwidth_family_->Add({{"device", device_->GetName()}});

  // Memory allocation metrics
  memory_allocated_family_ = &prometheus::BuildGauge()
      .Name("metal_memory_allocated_bytes")
      .Help("Currently allocated memory in bytes")
      .Register(*registry);
  memory_allocated_gauge_ = &memory_allocated_family_->Add({{"device", device_->GetName()}});

  // Power consumption metrics
  power_consumption_family_ = &prometheus::BuildGauge()
      .Name("metal_power_consumption_watts")
      .Help("Current power consumption in watts")
      .Register(*registry);
  power_consumption_gauge_ = &power_consumption_family_->Add({{"device", device_->GetName()}});

  // Thermal state metrics
  thermal_state_family_ = &prometheus::BuildGauge()
      .Name("metal_thermal_state")
      .Help("Current thermal state (0=nominal, 1=fair, 2=serious, 3=critical)")
      .Register(*registry);
  thermal_state_gauge_ = &thermal_state_family_->Add({{"device", device_->GetName()}});

  // Kernel execution metrics
  kernel_execution_count_family_ = &prometheus::BuildCounter()
      .Name("metal_kernel_execution_total")
      .Help("Total number of kernel executions")
      .Register(*registry);

  kernel_execution_time_family_ = &prometheus::BuildSummary()
      .Name("metal_kernel_execution_duration_milliseconds")
      .Help("Kernel execution duration in milliseconds")
      .Register(*registry);

  // Command buffer latency histogram
  command_buffer_latency_family_ = &prometheus::BuildHistogram()
      .Name("metal_command_buffer_latency_milliseconds")
      .Help("Command buffer submission to completion latency")
      .Register(*registry);
}

void MetalPerformanceMonitor::UpdatePrometheusMetrics() {
  std::lock_guard<std::mutex> lock(mutex_);

  // Update gauge metrics with latest values
  auto current_metrics = GetCurrentMetrics();
  for (const auto& metric : current_metrics) {
    switch (metric.type) {
      case MetalCounterType::GPU_UTILIZATION:
        if (gpu_utilization_gauge_) {
          gpu_utilization_gauge_->Set(metric.value);
        }
        break;
      case MetalCounterType::MEMORY_BANDWIDTH:
        if (memory_bandwidth_gauge_) {
          memory_bandwidth_gauge_->Set(metric.value);
        }
        break;
      case MetalCounterType::MEMORY_ALLOCATION:
        if (memory_allocated_gauge_) {
          memory_allocated_gauge_->Set(metric.value);
        }
        break;
      case MetalCounterType::POWER_CONSUMPTION:
        if (power_consumption_gauge_) {
          power_consumption_gauge_->Set(metric.value);
        }
        break;
      case MetalCounterType::THERMAL_STATE:
        if (thermal_state_gauge_) {
          thermal_state_gauge_->Set(metric.value);
        }
        break;
      default:
        break;
    }
  }

  // Update kernel execution metrics
  for (const auto& [kernel_name, profiles] : kernel_profiles_) {
    // Get or create counter/summary for this kernel
    auto counter_it = kernel_execution_counters_.find(kernel_name);
    if (counter_it == kernel_execution_counters_.end()) {
      auto& counter = kernel_execution_count_family_->Add({{"kernel", kernel_name}});
      kernel_execution_counters_[kernel_name] = &counter;
      counter_it = kernel_execution_counters_.find(kernel_name);
    }

    auto summary_it = kernel_execution_summaries_.find(kernel_name);
    if (summary_it == kernel_execution_summaries_.end()) {
      auto& summary = kernel_execution_time_family_->Add(
          {{"kernel", kernel_name}},
          prometheus::Summary::Quantiles{
              {0.5, 0.05}, {0.9, 0.01}, {0.99, 0.001}});
      kernel_execution_summaries_[kernel_name] = &summary;
      summary_it = kernel_execution_summaries_.find(kernel_name);
    }

    // Update counter with execution count
    counter_it->second->Increment(profiles.size());

    // Update summary with execution times
    for (const auto& profile : profiles) {
      double duration_ms = (profile.end_time_ns - profile.start_time_ns) / 1e6;
      summary_it->second->Observe(duration_ms);
    }
  }
}
#endif

void MetalPerformanceMonitor::SetSamplingInterval(uint64_t interval_ms) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_.sampling_interval_ms = interval_ms;
}

void MetalPerformanceMonitor::ClearHistory() {
  std::lock_guard<std::mutex> lock(mutex_);
  metric_history_.clear();
  kernel_profiles_.clear();
  completed_command_buffers_.clear();
}

void MetalPerformanceMonitor::EnableDetailedProfiling(bool enable) {
  std::lock_guard<std::mutex> lock(mutex_);
  config_.enable_detailed_profiling = enable;
  if (enable && !counter_sample_buffer_) {
    SetupCounterSampleBuffer();
  }
}

MetalPerformanceMonitor::ABTestResult MetalPerformanceMonitor::CompareKernelPerformance(
    const std::string& kernel_a, const std::string& kernel_b) const {
  std::lock_guard<std::mutex> lock(mutex_);

  ABTestResult result;
  result.variant_a_name = kernel_a;
  result.variant_b_name = kernel_b;

  // Calculate average execution times for both kernels
  auto profiles_a = kernel_profiles_.find(kernel_a);
  auto profiles_b = kernel_profiles_.find(kernel_b);

  if (profiles_a != kernel_profiles_.end() && !profiles_a->second.empty()) {
    double total_time = 0;
    for (const auto& profile : profiles_a->second) {
      total_time += (profile.end_time_ns - profile.start_time_ns) / 1e6;
    }
    result.variant_a_avg_time_ms = total_time / profiles_a->second.size();
  }

  if (profiles_b != kernel_profiles_.end() && !profiles_b->second.empty()) {
    double total_time = 0;
    for (const auto& profile : profiles_b->second) {
      total_time += (profile.end_time_ns - profile.start_time_ns) / 1e6;
    }
    result.variant_b_avg_time_ms = total_time / profiles_b->second.size();
  }

  // Calculate speedup factor
  if (result.variant_a_avg_time_ms > 0 && result.variant_b_avg_time_ms > 0) {
    result.speedup_factor = result.variant_a_avg_time_ms / result.variant_b_avg_time_ms;
    
    std::ostringstream recommendation;
    if (result.speedup_factor > 1.1) {
      recommendation << kernel_b << " is " << std::fixed << std::setprecision(2)
                     << result.speedup_factor << "x faster than " << kernel_a;
    } else if (result.speedup_factor < 0.9) {
      recommendation << kernel_a << " is " << std::fixed << std::setprecision(2)
                     << (1.0 / result.speedup_factor) << "x faster than " << kernel_b;
    } else {
      recommendation << "Both kernels have similar performance";
    }
    result.recommendation = recommendation.str();
  } else {
    result.recommendation = "Insufficient data for comparison";
  }

  return result;
}

void MetalPerformanceMonitor::MonitoringThread() {
  while (is_monitoring_) {
    auto start_time = std::chrono::steady_clock::now();

    // Collect various metrics
    if (config_.enable_gpu_utilization) {
      CollectGPUUtilization();
    }
    if (config_.enable_memory_bandwidth) {
      CollectMemoryBandwidth();
    }
    if (config_.enable_power_monitoring) {
      CollectPowerMetrics();
    }
    if (config_.enable_thermal_monitoring) {
      CollectThermalMetrics();
    }
    if (config_.enable_queue_monitoring) {
      CollectQueueUtilization();
    }

    // Process counter samples if available
    if (config_.enable_detailed_profiling && counter_sample_buffer_) {
      ProcessCounterSamples();
    }

#ifdef TRITON_ENABLE_METRICS
    // Update Prometheus metrics
    if (config_.enable_prometheus_export) {
      UpdatePrometheusMetrics();
    }
#endif

    // Sleep for the remainder of the interval
    auto elapsed = std::chrono::steady_clock::now() - start_time;
    auto sleep_duration = std::chrono::milliseconds(config_.sampling_interval_ms) - elapsed;
    if (sleep_duration.count() > 0) {
      std::this_thread::sleep_for(sleep_duration);
    }
  }
}

void MetalPerformanceMonitor::CollectGPUUtilization() {
  @autoreleasepool {
    // Get GPU utilization from system
    // Note: This is a simplified implementation. Real GPU utilization
    // would require more sophisticated system queries
    // id<MTLDevice> mtl_device = device_->GetDevice();
    
    // Check if device supports GPU utilization queries
    if (@available(macOS 10.15, *)) {
      // Get current GPU activity
      // This is a placeholder - actual implementation would query system metrics
      double utilization = 0.0;
      
      // For now, estimate based on active command buffers
      {
        std::lock_guard<std::mutex> lock(mutex_);
        utilization = static_cast<double>(command_buffer_profiles_.size()) / 10.0;
        utilization = std::min(1.0, utilization);
      }

      MetalPerformanceMetric metric;
      metric.type = MetalCounterType::GPU_UTILIZATION;
      metric.name = "gpu_utilization";
      metric.value = utilization;
      metric.timestamp_ns = GetCurrentTimeNs();
      metric.labels["device"] = device_->GetName();
      
      AddMetricToHistory(metric);
    }
  }
}

void MetalPerformanceMonitor::CollectMemoryBandwidth() {
  @autoreleasepool {
    // Estimate memory bandwidth based on recent allocations and deallocations
    // This is a simplified estimation
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Calculate bandwidth based on memory operations in the last second
    uint64_t current_time_ns = GetCurrentTimeNs();
    uint64_t one_second_ago_ns = current_time_ns - 1000000000;
    
    size_t bytes_transferred = 0;
    for (const auto& [ptr, info] : memory_allocations_) {
      if (info.allocation_time_ns >= one_second_ago_ns) {
        bytes_transferred += info.size;
      }
    }
    
    double bandwidth_gbps = static_cast<double>(bytes_transferred) / 1e9;
    
    MetalPerformanceMetric metric;
    metric.type = MetalCounterType::MEMORY_BANDWIDTH;
    metric.name = "memory_bandwidth";
    metric.value = bandwidth_gbps;
    metric.timestamp_ns = current_time_ns;
    metric.labels["device"] = device_->GetName();
    
    AddMetricToHistory(metric);
  }
}

void MetalPerformanceMonitor::CollectPowerMetrics() {
  @autoreleasepool {
    // Get power consumption data
    // Note: This is platform-specific and requires IOKit
    double power_watts = 0.0;
    
    // Placeholder for power measurement
    // Real implementation would query IOPowerSources or similar
    if (@available(macOS 10.15, *)) {
      // Estimate based on GPU utilization
      auto utilization_metrics = GetMetricHistory(MetalCounterType::GPU_UTILIZATION, 1);
      if (!utilization_metrics.empty()) {
        // Rough estimate: assume 150W max power, scale by utilization
        power_watts = utilization_metrics.back().value * 150.0;
      }
    }
    
    MetalPerformanceMetric metric;
    metric.type = MetalCounterType::POWER_CONSUMPTION;
    metric.name = "power_consumption";
    metric.value = power_watts;
    metric.timestamp_ns = GetCurrentTimeNs();
    metric.labels["device"] = device_->GetName();
    
    AddMetricToHistory(metric);
  }
}

void MetalPerformanceMonitor::CollectThermalMetrics() {
  @autoreleasepool {
    // Get thermal state
    // 0 = Nominal, 1 = Fair, 2 = Serious, 3 = Critical
    int thermal_state = 0;
    
    // Check system thermal state
    if (@available(macOS 10.10.3, *)) {
      NSProcessInfo* info = [NSProcessInfo processInfo];
      if ([info respondsToSelector:@selector(thermalState)]) {
        NSProcessInfoThermalState state = [info thermalState];
        switch (state) {
          case NSProcessInfoThermalStateNominal:
            thermal_state = 0;
            break;
          case NSProcessInfoThermalStateFair:
            thermal_state = 1;
            break;
          case NSProcessInfoThermalStateSerious:
            thermal_state = 2;
            break;
          case NSProcessInfoThermalStateCritical:
            thermal_state = 3;
            break;
        }
      }
    }
    
    MetalPerformanceMetric metric;
    metric.type = MetalCounterType::THERMAL_STATE;
    metric.name = "thermal_state";
    metric.value = static_cast<double>(thermal_state);
    metric.timestamp_ns = GetCurrentTimeNs();
    metric.labels["device"] = device_->GetName();
    
    AddMetricToHistory(metric);
  }
}

void MetalPerformanceMonitor::CollectQueueUtilization() {
  @autoreleasepool {
    // Monitor command queue utilization
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Count active command buffers
    size_t active_buffers = command_buffer_profiles_.size();
    
    // Estimate queue utilization (simplified)
    double utilization = std::min(1.0, static_cast<double>(active_buffers) / 32.0);
    
    MetalPerformanceMetric metric;
    metric.type = MetalCounterType::QUEUE_UTILIZATION;
    metric.name = "queue_utilization";
    metric.value = utilization;
    metric.timestamp_ns = GetCurrentTimeNs();
    metric.labels["device"] = device_->GetName();
    
    AddMetricToHistory(metric);
  }
}

void MetalPerformanceMonitor::SetupCounterSampleBuffer() {
  @autoreleasepool {
    id<MTLDevice> mtl_device = device_->GetDevice();
    
    if (@available(macOS 10.15, *)) {
      // Get available counter sets
      NSArray<id<MTLCounterSet>>* counterSets = [mtl_device counterSets];
      
      for (id<MTLCounterSet> counterSet in counterSets) {
        NSString* name = [counterSet name];
        if ([name containsString:@"GPU"]) {
          gpu_counter_set_ = counterSet;
        } else if ([name containsString:@"Encoder"]) {
          encoder_counter_set_ = counterSet;
        } else if ([name containsString:@"Device"]) {
          device_counter_set_ = counterSet;
        }
      }
      
      // Create counter sample buffer if we have counter sets
      if (gpu_counter_set_ || encoder_counter_set_ || device_counter_set_) {
        MTLCounterSampleBufferDescriptor* descriptor = [[MTLCounterSampleBufferDescriptor alloc] init];
        descriptor.sampleCount = 1024;
        descriptor.label = @"TritonMetalPerformanceMonitor";
        descriptor.storageMode = MTLStorageModeShared;
        
        if (gpu_counter_set_) {
          [descriptor setCounterSet:gpu_counter_set_];
        }
        
        NSError* error = nil;
        counter_sample_buffer_ = [mtl_device newCounterSampleBufferWithDescriptor:descriptor error:&error];
        
        if (error) {
          NSLog(@"Failed to create counter sample buffer: %@", error);
        }
      }
    }
  }
}

void MetalPerformanceMonitor::ProcessCounterSamples() {
  if (!counter_sample_buffer_) {
    return;
  }
  
  @autoreleasepool {
    if (@available(macOS 10.15, *)) {
      // Process counter samples
      // This is a placeholder - actual implementation would read and process
      // counter data from the sample buffer
    }
  }
}

uint64_t MetalPerformanceMonitor::GetCurrentTimeNs() const {
  return GetSystemTimeNs();
}

void MetalPerformanceMonitor::AddMetricToHistory(const MetalPerformanceMetric& metric) {
  auto& history = metric_history_[metric.type];
  history.push_back(metric);
  
  // Keep history bounded
  if (history.size() > config_.history_buffer_size) {
    history.erase(history.begin());
  }
}

void MetalPerformanceMonitor::AnalyzeBottlenecks(PerformanceAnalysis& analysis) const {
  // Identify performance bottlenecks
  
  // High GPU utilization
  if (analysis.avg_gpu_utilization > 0.9) {
    analysis.performance_bottlenecks.push_back("GPU compute bound - utilization > 90%");
  }
  
  // Memory bandwidth saturation
  if (analysis.avg_memory_bandwidth_gbps > 400) {  // Assuming ~500 GB/s max
    analysis.performance_bottlenecks.push_back("Memory bandwidth saturated");
  }
  
  // Long kernel execution times
  for (const auto& [kernel, time_ms] : analysis.kernel_execution_times_ms) {
    if (time_ms > 100) {
      analysis.performance_bottlenecks.push_back(
          "Kernel '" + kernel + "' takes " + std::to_string(time_ms) + "ms");
    }
  }
  
  // Memory pressure
  auto mem_stats = GetMemoryStatistics();
  if (mem_stats.current_allocated_bytes > 8ULL * 1024 * 1024 * 1024) {  // 8GB
    analysis.performance_bottlenecks.push_back("High memory usage > 8GB");
  }
}

void MetalPerformanceMonitor::GenerateOptimizationRecommendations(
    PerformanceAnalysis& analysis) const {
  // Generate optimization recommendations based on bottlenecks
  
  for (const auto& bottleneck : analysis.performance_bottlenecks) {
    if (bottleneck.find("GPU compute bound") != std::string::npos) {
      analysis.optimization_recommendations.push_back(
          "Consider kernel fusion or reducing arithmetic intensity");
      analysis.optimization_recommendations.push_back(
          "Profile individual kernels to find optimization opportunities");
    }
    
    if (bottleneck.find("Memory bandwidth") != std::string::npos) {
      analysis.optimization_recommendations.push_back(
          "Optimize memory access patterns for better cache utilization");
      analysis.optimization_recommendations.push_back(
          "Consider using shared memory for frequently accessed data");
    }
    
    if (bottleneck.find("takes") != std::string::npos && 
        bottleneck.find("ms") != std::string::npos) {
      analysis.optimization_recommendations.push_back(
          "Long-running kernel detected - consider splitting into smaller kernels");
      analysis.optimization_recommendations.push_back(
          "Review threadgroup size and grid dimensions for better occupancy");
    }
    
    if (bottleneck.find("High memory usage") != std::string::npos) {
      analysis.optimization_recommendations.push_back(
          "Implement memory pooling to reduce allocation overhead");
      analysis.optimization_recommendations.push_back(
          "Consider streaming or tiling large operations");
    }
  }
}

// PerformanceMonitoringScope implementation
PerformanceMonitoringScope::PerformanceMonitoringScope(
    MetalPerformanceMonitor* monitor,
    const std::string& scope_name,
    id<MTLComputeCommandEncoder> encoder)
    : monitor_(monitor), scope_name_(scope_name), encoder_(encoder) {
  if (monitor_) {
    start_time_ns_ = GetSystemTimeNs();
    monitor_->BeginKernelProfiling(scope_name_, encoder_);
  }
}

PerformanceMonitoringScope::~PerformanceMonitoringScope() {
  if (monitor_) {
    monitor_->EndKernelProfiling(scope_name_, encoder_);
  }
}

// PerformanceMonitorManager implementation
PerformanceMonitorManager& PerformanceMonitorManager::Instance() {
  static PerformanceMonitorManager instance;
  return instance;
}

void PerformanceMonitorManager::RegisterMonitor(
    const std::string& name,
    std::shared_ptr<MetalPerformanceMonitor> monitor) {
  std::lock_guard<std::mutex> lock(mutex_);
  monitors_[name] = monitor;
}

void PerformanceMonitorManager::UnregisterMonitor(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  monitors_.erase(name);
}

std::shared_ptr<MetalPerformanceMonitor> PerformanceMonitorManager::GetMonitor(
    const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = monitors_.find(name);
  return it != monitors_.end() ? it->second : nullptr;
}

std::vector<std::string> PerformanceMonitorManager::GetMonitorNames() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::string> names;
  names.reserve(monitors_.size());
  for (const auto& [name, _] : monitors_) {
    names.push_back(name);
  }
  return names;
}

std::vector<MetalPerformanceMetric> PerformanceMonitorManager::GetAggregatedMetrics() const {
  std::lock_guard<std::mutex> lock(mutex_);
  
  std::vector<MetalPerformanceMetric> aggregated;
  for (const auto& [name, monitor] : monitors_) {
    if (monitor) {
      auto metrics = monitor->GetCurrentMetrics();
      aggregated.insert(aggregated.end(), metrics.begin(), metrics.end());
    }
  }
  
  return aggregated;
}

MetalPerformanceMonitor::PerformanceAnalysis
PerformanceMonitorManager::GetAggregatedAnalysis() const {
  std::lock_guard<std::mutex> lock(mutex_);
  
  MetalPerformanceMonitor::PerformanceAnalysis aggregated = {};
  size_t monitor_count = 0;
  
  for (const auto& [name, monitor] : monitors_) {
    if (monitor) {
      auto analysis = monitor->AnalyzePerformance();
      
      // Aggregate metrics
      aggregated.avg_gpu_utilization += analysis.avg_gpu_utilization;
      aggregated.peak_gpu_utilization = std::max(
          aggregated.peak_gpu_utilization, analysis.peak_gpu_utilization);
      aggregated.avg_memory_bandwidth_gbps += analysis.avg_memory_bandwidth_gbps;
      aggregated.peak_memory_bandwidth_gbps = std::max(
          aggregated.peak_memory_bandwidth_gbps, analysis.peak_memory_bandwidth_gbps);
      aggregated.avg_kernel_execution_time_ms += analysis.avg_kernel_execution_time_ms;
      
      // Merge kernel execution times
      for (const auto& [kernel, time] : analysis.kernel_execution_times_ms) {
        aggregated.kernel_execution_times_ms[kernel] = time;
      }
      
      // Merge bottlenecks and recommendations
      aggregated.performance_bottlenecks.insert(
          aggregated.performance_bottlenecks.end(),
          analysis.performance_bottlenecks.begin(),
          analysis.performance_bottlenecks.end());
      aggregated.optimization_recommendations.insert(
          aggregated.optimization_recommendations.end(),
          analysis.optimization_recommendations.begin(),
          analysis.optimization_recommendations.end());
      
      monitor_count++;
    }
  }
  
  // Average the averaged metrics
  if (monitor_count > 0) {
    aggregated.avg_gpu_utilization /= monitor_count;
    aggregated.avg_memory_bandwidth_gbps /= monitor_count;
    aggregated.avg_kernel_execution_time_ms /= monitor_count;
  }
  
  return aggregated;
}

}}  // namespace triton::metal