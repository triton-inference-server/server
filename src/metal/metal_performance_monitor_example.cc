// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Example usage of Metal Performance Monitoring System

#include <iostream>
#include <thread>
#include <chrono>

#include "metal_performance_monitor.h"
#include "metal_performance_visualizer.h"
#include "metal_device.h"
#include "metal_memory_manager.h"
#include "metal_command.h"

using namespace triton::metal;

// Example kernel execution simulation
void SimulateKernelExecution(
    MetalPerformanceMonitor* monitor,
    const std::string& kernel_name,
    double execution_time_ms) {
  
  // Begin profiling
  monitor->BeginKernelProfiling(kernel_name, nullptr);
  
  // Simulate kernel execution
  std::this_thread::sleep_for(
      std::chrono::microseconds(static_cast<int64_t>(execution_time_ms * 1000)));
  
  // End profiling
  monitor->EndKernelProfiling(kernel_name, nullptr);
}

// Example memory allocation simulation
void SimulateMemoryOperations(MetalPerformanceMonitor* monitor) {
  // Simulate allocations
  void* ptr1 = reinterpret_cast<void*>(0x1000);
  void* ptr2 = reinterpret_cast<void*>(0x2000);
  void* ptr3 = reinterpret_cast<void*>(0x3000);
  
  monitor->TrackMemoryAllocation(ptr1, 1024 * 1024, false, "buffer_1");
  monitor->TrackMemoryAllocation(ptr2, 2048 * 1024, true, "shared_buffer");
  monitor->TrackMemoryAllocation(ptr3, 512 * 1024, false, "buffer_3");
  
  // Simulate some delay
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  
  // Deallocate some memory
  monitor->TrackMemoryDeallocation(ptr1);
  monitor->TrackMemoryDeallocation(ptr3);
}

int main(int argc, char** argv) {
  std::cout << "Metal Performance Monitoring Example\n";
  std::cout << "===================================\n\n";

  // Create Metal device
  auto device = std::make_shared<MetalDevice>();
  if (!device->Initialize().IsOk()) {
    std::cerr << "Failed to initialize Metal device\n";
    return 1;
  }

  // Configure performance monitoring
  PerformanceMonitorConfig config;
  config.enable_gpu_utilization = true;
  config.enable_memory_bandwidth = true;
  config.enable_kernel_profiling = true;
  config.enable_power_monitoring = true;
  config.enable_thermal_monitoring = true;
  config.enable_queue_monitoring = true;
  config.sampling_interval_ms = 100;  // 100ms sampling

  // Create performance monitor
  auto monitor = std::make_shared<MetalPerformanceMonitor>(device, config);
  if (!monitor->Initialize().IsOk()) {
    std::cerr << "Failed to initialize performance monitor\n";
    return 1;
  }

  // Register with global manager
  PerformanceMonitorManager::Instance().RegisterMonitor("main", monitor);

  // Start monitoring
  std::cout << "Starting performance monitoring...\n";
  monitor->StartMonitoring();

  // Simulate various workloads
  std::cout << "\nSimulating workloads...\n";

  // Simulate memory operations
  SimulateMemoryOperations(monitor.get());

  // Simulate kernel executions with varying performance
  for (int i = 0; i < 10; ++i) {
    SimulateKernelExecution(monitor.get(), "gemm_kernel", 5.0 + (i % 3) * 2.0);
    SimulateKernelExecution(monitor.get(), "conv2d_kernel", 8.0 + (i % 4) * 1.5);
    SimulateKernelExecution(monitor.get(), "reduce_kernel", 2.0 + (i % 2) * 0.5);
    
    // Add some variation in timing
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  // Simulate A/B testing scenario
  std::cout << "\nRunning A/B test between kernels...\n";
  for (int i = 0; i < 20; ++i) {
    if (i % 2 == 0) {
      SimulateKernelExecution(monitor.get(), "gemm_v1", 10.0);
    } else {
      SimulateKernelExecution(monitor.get(), "gemm_v2", 7.5);
    }
  }

  // Let monitoring collect data
  std::this_thread::sleep_for(std::chrono::seconds(2));

  // Analyze performance
  std::cout << "\nAnalyzing performance...\n";
  auto analysis = monitor->AnalyzePerformance();

  // Create visualizer
  VisualizationConfig viz_config;
  viz_config.format = VisualizationFormat::TERMINAL;
  auto visualizer = std::make_shared<MetalPerformanceVisualizer>(monitor, viz_config);

  // Print performance summary to terminal
  visualizer->PrintPerformanceSummary();
  visualizer->PrintKernelProfilingTable();
  visualizer->PrintMemoryStatistics();
  visualizer->PrintBottlenecksAndRecommendations();

  // A/B test results
  auto ab_result = monitor->CompareKernelPerformance("gemm_v1", "gemm_v2");
  std::cout << "\n=== A/B Test Results ===\n";
  std::cout << "Variant A (" << ab_result.variant_a_name << "): " 
            << ab_result.variant_a_avg_time_ms << " ms\n";
  std::cout << "Variant B (" << ab_result.variant_b_name << "): " 
            << ab_result.variant_b_avg_time_ms << " ms\n";
  std::cout << "Speedup Factor: " << ab_result.speedup_factor << "x\n";
  std::cout << "Recommendation: " << ab_result.recommendation << "\n";

  // Generate HTML report
  std::cout << "\nGenerating HTML performance report...\n";
  viz_config.format = VisualizationFormat::HTML;
  viz_config.output_path = "metal_performance_report.html";
  auto html_visualizer = std::make_shared<MetalPerformanceVisualizer>(monitor, viz_config);
  if (html_visualizer->GenerateReport().IsOk()) {
    std::cout << "HTML report saved to: " << viz_config.output_path << "\n";
  }

  // Generate JSON report
  std::cout << "\nGenerating JSON performance data...\n";
  viz_config.format = VisualizationFormat::JSON;
  viz_config.output_path = "metal_performance_data.json";
  viz_config.include_raw_data = true;
  auto json_visualizer = std::make_shared<MetalPerformanceVisualizer>(monitor, viz_config);
  if (json_visualizer->GenerateReport().IsOk()) {
    std::cout << "JSON data saved to: " << viz_config.output_path << "\n";
  }

  // Generate Markdown report
  std::cout << "\nGenerating Markdown performance report...\n";
  viz_config.format = VisualizationFormat::MARKDOWN;
  viz_config.output_path = "metal_performance_report.md";
  auto md_visualizer = std::make_shared<MetalPerformanceVisualizer>(monitor, viz_config);
  if (md_visualizer->GenerateReport().IsOk()) {
    std::cout << "Markdown report saved to: " << viz_config.output_path << "\n";
  }

  // Stop monitoring
  std::cout << "\nStopping performance monitoring...\n";
  monitor->StopMonitoring();

  // Demonstrate aggregated metrics from multiple monitors
  std::cout << "\n=== Aggregated Performance Metrics ===\n";
  auto aggregated = PerformanceMonitorManager::Instance().GetAggregatedAnalysis();
  std::cout << "Overall GPU Utilization: " 
            << (aggregated.avg_gpu_utilization * 100) << "%\n";

  // Cleanup
  PerformanceMonitorManager::Instance().UnregisterMonitor("main");

  std::cout << "\nExample completed successfully!\n";
  return 0;
}