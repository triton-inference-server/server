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

#pragma once

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef TRITON_ENABLE_METRICS
#include "prometheus/counter.h"
#include "prometheus/gauge.h"
#include "prometheus/histogram.h"
#include "prometheus/registry.h"
#include "prometheus/summary.h"
#endif

namespace triton { namespace metal {

// Forward declarations
class MetalDevice;
class MetalMemoryManager;

// Performance counter types
enum class MetalCounterType {
  GPU_UTILIZATION,
  MEMORY_BANDWIDTH,
  KERNEL_EXECUTION_TIME,
  MEMORY_ALLOCATION,
  COMMAND_BUFFER_TIME,
  POWER_CONSUMPTION,
  THERMAL_STATE,
  QUEUE_UTILIZATION
};

// Performance metric structure
struct MetalPerformanceMetric {
  MetalCounterType type;
  std::string name;
  double value;
  uint64_t timestamp_ns;
  std::map<std::string, std::string> labels;
};

// Kernel profiling data
struct KernelProfilingData {
  std::string kernel_name;
  uint64_t start_time_ns;
  uint64_t end_time_ns;
  uint64_t gpu_start_time_ns;
  uint64_t gpu_end_time_ns;
  size_t thread_count;
  size_t threadgroup_size;
  size_t shared_memory_bytes;
  std::map<std::string, std::string> metadata;
};

// Memory allocation tracking
struct MemoryAllocationInfo {
  void* ptr;
  size_t size;
  bool is_shared;
  uint64_t allocation_time_ns;
  uint64_t deallocation_time_ns;
  std::string allocation_context;
};

// Command buffer profiling
struct CommandBufferProfile {
  std::string identifier;
  uint64_t submit_time_ns;
  uint64_t scheduled_time_ns;
  uint64_t kernel_start_time_ns;
  uint64_t kernel_end_time_ns;
  uint64_t completed_time_ns;
  std::vector<KernelProfilingData> kernel_profiles;
};

// Performance monitoring configuration
struct PerformanceMonitorConfig {
  bool enable_gpu_utilization = true;
  bool enable_memory_bandwidth = true;
  bool enable_kernel_profiling = true;
  bool enable_power_monitoring = true;
  bool enable_thermal_monitoring = true;
  bool enable_queue_monitoring = true;
  bool enable_detailed_profiling = false;
  uint64_t sampling_interval_ms = 100;  // 100ms default
  size_t history_buffer_size = 10000;   // Keep last 10k samples
  bool enable_prometheus_export = true;
};

// Metal Performance Monitor class
class MetalPerformanceMonitor {
 public:
  MetalPerformanceMonitor(
      std::shared_ptr<MetalDevice> device,
      const PerformanceMonitorConfig& config = PerformanceMonitorConfig());
  ~MetalPerformanceMonitor();

  // Initialize monitoring
  Status Initialize();

  // Start/stop monitoring
  Status StartMonitoring();
  Status StopMonitoring();
  bool IsMonitoring() const { return is_monitoring_; }

  // Kernel profiling
  void BeginKernelProfiling(
      const std::string& kernel_name, id<MTLComputeCommandEncoder> encoder);
  void EndKernelProfiling(
      const std::string& kernel_name, id<MTLComputeCommandEncoder> encoder);

  // Command buffer profiling
  void RegisterCommandBuffer(
      const std::string& identifier, id<MTLCommandBuffer> buffer);
  void OnCommandBufferScheduled(id<MTLCommandBuffer> buffer);
  void OnCommandBufferCompleted(id<MTLCommandBuffer> buffer);

  // Memory tracking
  void TrackMemoryAllocation(
      void* ptr, size_t size, bool is_shared, const std::string& context);
  void TrackMemoryDeallocation(void* ptr);

  // Get current metrics
  std::vector<MetalPerformanceMetric> GetCurrentMetrics() const;
  std::vector<MetalPerformanceMetric> GetMetricHistory(
      MetalCounterType type, size_t count = 0) const;

  // Get profiling data
  std::vector<KernelProfilingData> GetKernelProfilingData(
      const std::string& kernel_name = "") const;
  std::vector<CommandBufferProfile> GetCommandBufferProfiles(
      size_t count = 0) const;

  // Memory statistics
  struct MemoryStatistics {
    size_t total_allocated_bytes;
    size_t current_allocated_bytes;
    size_t peak_allocated_bytes;
    size_t allocation_count;
    size_t deallocation_count;
    size_t shared_memory_bytes;
    size_t device_memory_bytes;
  };
  MemoryStatistics GetMemoryStatistics() const;

  // Performance analysis
  struct PerformanceAnalysis {
    double avg_gpu_utilization;
    double peak_gpu_utilization;
    double avg_memory_bandwidth_gbps;
    double peak_memory_bandwidth_gbps;
    double avg_kernel_execution_time_ms;
    std::map<std::string, double> kernel_execution_times_ms;
    std::vector<std::string> performance_bottlenecks;
    std::vector<std::string> optimization_recommendations;
  };
  PerformanceAnalysis AnalyzePerformance(
      uint64_t window_duration_ms = 0) const;

  // Export metrics for Prometheus
#ifdef TRITON_ENABLE_METRICS
  void RegisterPrometheusMetrics(
      std::shared_ptr<prometheus::Registry> registry);
  void UpdatePrometheusMetrics();
#endif

  // Utility functions
  void SetSamplingInterval(uint64_t interval_ms);
  void ClearHistory();
  void EnableDetailedProfiling(bool enable);

  // A/B testing support
  struct ABTestResult {
    std::string variant_a_name;
    std::string variant_b_name;
    double variant_a_avg_time_ms;
    double variant_b_avg_time_ms;
    double speedup_factor;
    std::string recommendation;
  };
  ABTestResult CompareKernelPerformance(
      const std::string& kernel_a, const std::string& kernel_b) const;

 private:
  // Internal monitoring methods
  void MonitoringThread();
  void CollectGPUUtilization();
  void CollectMemoryBandwidth();
  void CollectPowerMetrics();
  void CollectThermalMetrics();
  void CollectQueueUtilization();

  // Metal counter sample buffer support
  void SetupCounterSampleBuffer();
  void ProcessCounterSamples();

  // Helper methods
  uint64_t GetCurrentTimeNs() const;
  void AddMetricToHistory(const MetalPerformanceMetric& metric);
  void AnalyzeBottlenecks(PerformanceAnalysis& analysis) const;
  void GenerateOptimizationRecommendations(PerformanceAnalysis& analysis) const;

 private:
  // Device and configuration
  std::shared_ptr<MetalDevice> device_;
  PerformanceMonitorConfig config_;

  // Monitoring state
  std::atomic<bool> is_monitoring_{false};
  std::unique_ptr<std::thread> monitoring_thread_;
  mutable std::mutex mutex_;

  // Performance metrics storage
  std::map<MetalCounterType, std::vector<MetalPerformanceMetric>>
      metric_history_;
  
  // Kernel profiling data
  std::unordered_map<std::string, std::vector<KernelProfilingData>>
      kernel_profiles_;
  std::unordered_map<id<MTLComputeCommandEncoder>, KernelProfilingData>
      active_kernel_profiles_;

  // Command buffer tracking
  std::unordered_map<id<MTLCommandBuffer>, CommandBufferProfile>
      command_buffer_profiles_;
  std::vector<CommandBufferProfile> completed_command_buffers_;

  // Memory tracking
  std::unordered_map<void*, MemoryAllocationInfo> memory_allocations_;
  MemoryStatistics memory_stats_;

  // Metal performance counter support
  id<MTLCounterSampleBuffer> counter_sample_buffer_;
  id<MTLCounterSet> gpu_counter_set_;
  id<MTLCounterSet> encoder_counter_set_;
  id<MTLCounterSet> device_counter_set_;

#ifdef TRITON_ENABLE_METRICS
  // Prometheus metrics families
  prometheus::Family<prometheus::Gauge>* gpu_utilization_family_;
  prometheus::Family<prometheus::Gauge>* memory_bandwidth_family_;
  prometheus::Family<prometheus::Gauge>* memory_allocated_family_;
  prometheus::Family<prometheus::Gauge>* power_consumption_family_;
  prometheus::Family<prometheus::Gauge>* thermal_state_family_;
  prometheus::Family<prometheus::Counter>* kernel_execution_count_family_;
  prometheus::Family<prometheus::Summary>* kernel_execution_time_family_;
  prometheus::Family<prometheus::Histogram>* command_buffer_latency_family_;

  // Individual metric instances
  prometheus::Gauge* gpu_utilization_gauge_;
  prometheus::Gauge* memory_bandwidth_gauge_;
  prometheus::Gauge* memory_allocated_gauge_;
  prometheus::Gauge* power_consumption_gauge_;
  prometheus::Gauge* thermal_state_gauge_;
  std::unordered_map<std::string, prometheus::Counter*> kernel_execution_counters_;
  std::unordered_map<std::string, prometheus::Summary*> kernel_execution_summaries_;
#endif
};

// Performance monitoring utilities
class PerformanceMonitoringScope {
 public:
  PerformanceMonitoringScope(
      MetalPerformanceMonitor* monitor,
      const std::string& scope_name,
      id<MTLComputeCommandEncoder> encoder = nil);
  ~PerformanceMonitoringScope();

 private:
  MetalPerformanceMonitor* monitor_;
  std::string scope_name_;
  id<MTLComputeCommandEncoder> encoder_;
  uint64_t start_time_ns_;
};

// Global performance monitor instance management
class PerformanceMonitorManager {
 public:
  static PerformanceMonitorManager& Instance();

  void RegisterMonitor(
      const std::string& name,
      std::shared_ptr<MetalPerformanceMonitor> monitor);
  void UnregisterMonitor(const std::string& name);
  
  std::shared_ptr<MetalPerformanceMonitor> GetMonitor(
      const std::string& name) const;
  std::vector<std::string> GetMonitorNames() const;

  // Global metrics aggregation
  std::vector<MetalPerformanceMetric> GetAggregatedMetrics() const;
  PerformanceAnalysis GetAggregatedAnalysis() const;

 private:
  PerformanceMonitorManager() = default;
  ~PerformanceMonitorManager() = default;

  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<MetalPerformanceMonitor>>
      monitors_;
};

}}  // namespace triton::metal