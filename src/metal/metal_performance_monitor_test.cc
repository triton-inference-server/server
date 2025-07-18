// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Unit tests for Metal Performance Monitoring System

#include <gtest/gtest.h>
#include <chrono>
#include <thread>

#include "metal_performance_monitor.h"
#include "metal_performance_visualizer.h"
#include "metal_device.h"

namespace triton { namespace metal {

class MetalPerformanceMonitorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = std::make_shared<MetalDevice>();
    ASSERT_TRUE(device_->Initialize().IsOk());
    
    PerformanceMonitorConfig config;
    config.sampling_interval_ms = 10;  // Fast sampling for tests
    config.history_buffer_size = 100;
    monitor_ = std::make_shared<MetalPerformanceMonitor>(device_, config);
    ASSERT_TRUE(monitor_->Initialize().IsOk());
  }

  void TearDown() override {
    if (monitor_->IsMonitoring()) {
      monitor_->StopMonitoring();
    }
  }

  std::shared_ptr<MetalDevice> device_;
  std::shared_ptr<MetalPerformanceMonitor> monitor_;
};

TEST_F(MetalPerformanceMonitorTest, InitializationTest) {
  EXPECT_FALSE(monitor_->IsMonitoring());
  EXPECT_TRUE(monitor_->GetCurrentMetrics().empty());
}

TEST_F(MetalPerformanceMonitorTest, StartStopMonitoringTest) {
  EXPECT_FALSE(monitor_->IsMonitoring());
  
  EXPECT_TRUE(monitor_->StartMonitoring().IsOk());
  EXPECT_TRUE(monitor_->IsMonitoring());
  
  // Wait for some metrics to be collected
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  
  EXPECT_TRUE(monitor_->StopMonitoring().IsOk());
  EXPECT_FALSE(monitor_->IsMonitoring());
  
  // Should not be able to stop again
  EXPECT_FALSE(monitor_->StopMonitoring().IsOk());
}

TEST_F(MetalPerformanceMonitorTest, KernelProfilingTest) {
  monitor_->BeginKernelProfiling("test_kernel", nullptr);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  monitor_->EndKernelProfiling("test_kernel", nullptr);
  
  auto profiles = monitor_->GetKernelProfilingData("test_kernel");
  ASSERT_EQ(profiles.size(), 1);
  EXPECT_EQ(profiles[0].kernel_name, "test_kernel");
  EXPECT_GT(profiles[0].end_time_ns, profiles[0].start_time_ns);
}

TEST_F(MetalPerformanceMonitorTest, MemoryTrackingTest) {
  void* ptr1 = reinterpret_cast<void*>(0x1000);
  void* ptr2 = reinterpret_cast<void*>(0x2000);
  
  monitor_->TrackMemoryAllocation(ptr1, 1024, false, "test_buffer_1");
  monitor_->TrackMemoryAllocation(ptr2, 2048, true, "test_buffer_2");
  
  auto stats = monitor_->GetMemoryStatistics();
  EXPECT_EQ(stats.current_allocated_bytes, 3072);
  EXPECT_EQ(stats.allocation_count, 2);
  EXPECT_EQ(stats.shared_memory_bytes, 2048);
  EXPECT_EQ(stats.device_memory_bytes, 1024);
  
  monitor_->TrackMemoryDeallocation(ptr1);
  stats = monitor_->GetMemoryStatistics();
  EXPECT_EQ(stats.current_allocated_bytes, 2048);
  EXPECT_EQ(stats.deallocation_count, 1);
}

TEST_F(MetalPerformanceMonitorTest, MetricHistoryTest) {
  EXPECT_TRUE(monitor_->StartMonitoring().IsOk());
  
  // Wait for metrics to be collected
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  
  auto gpu_metrics = monitor_->GetMetricHistory(MetalCounterType::GPU_UTILIZATION);
  EXPECT_GT(gpu_metrics.size(), 0);
  
  // Test limited history
  auto limited_metrics = monitor_->GetMetricHistory(MetalCounterType::GPU_UTILIZATION, 5);
  EXPECT_LE(limited_metrics.size(), 5);
  
  monitor_->StopMonitoring();
}

TEST_F(MetalPerformanceMonitorTest, PerformanceAnalysisTest) {
  // Simulate some kernel executions
  for (int i = 0; i < 5; ++i) {
    monitor_->BeginKernelProfiling("analyze_kernel", nullptr);
    std::this_thread::sleep_for(std::chrono::milliseconds(5 + i));
    monitor_->EndKernelProfiling("analyze_kernel", nullptr);
  }
  
  auto analysis = monitor_->AnalyzePerformance();
  EXPECT_GE(analysis.kernel_execution_times_ms.size(), 0);
  
  if (analysis.kernel_execution_times_ms.count("analyze_kernel") > 0) {
    EXPECT_GT(analysis.kernel_execution_times_ms["analyze_kernel"], 0);
  }
}

TEST_F(MetalPerformanceMonitorTest, ABTestingTest) {
  // Simulate kernel A executions
  for (int i = 0; i < 10; ++i) {
    monitor_->BeginKernelProfiling("kernel_a", nullptr);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    monitor_->EndKernelProfiling("kernel_a", nullptr);
  }
  
  // Simulate kernel B executions (faster)
  for (int i = 0; i < 10; ++i) {
    monitor_->BeginKernelProfiling("kernel_b", nullptr);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    monitor_->EndKernelProfiling("kernel_b", nullptr);
  }
  
  auto result = monitor_->CompareKernelPerformance("kernel_a", "kernel_b");
  EXPECT_EQ(result.variant_a_name, "kernel_a");
  EXPECT_EQ(result.variant_b_name, "kernel_b");
  EXPECT_GT(result.variant_a_avg_time_ms, result.variant_b_avg_time_ms);
  EXPECT_GT(result.speedup_factor, 1.0);
}

TEST_F(MetalPerformanceMonitorTest, ClearHistoryTest) {
  // Add some data
  monitor_->BeginKernelProfiling("clear_test", nullptr);
  monitor_->EndKernelProfiling("clear_test", nullptr);
  
  EXPECT_FALSE(monitor_->GetKernelProfilingData().empty());
  
  monitor_->ClearHistory();
  
  EXPECT_TRUE(monitor_->GetKernelProfilingData().empty());
}

TEST_F(MetalPerformanceMonitorTest, PerformanceMonitorManagerTest) {
  auto& manager = PerformanceMonitorManager::Instance();
  
  manager.RegisterMonitor("test_monitor", monitor_);
  
  auto names = manager.GetMonitorNames();
  EXPECT_TRUE(std::find(names.begin(), names.end(), "test_monitor") != names.end());
  
  auto retrieved = manager.GetMonitor("test_monitor");
  EXPECT_EQ(retrieved, monitor_);
  
  manager.UnregisterMonitor("test_monitor");
  EXPECT_EQ(manager.GetMonitor("test_monitor"), nullptr);
}

// Visualizer Tests
class MetalPerformanceVisualizerTest : public MetalPerformanceMonitorTest {
 protected:
  void SetUp() override {
    MetalPerformanceMonitorTest::SetUp();
    
    VisualizationConfig config;
    config.format = VisualizationFormat::JSON;
    visualizer_ = std::make_shared<MetalPerformanceVisualizer>(monitor_, config);
  }

  std::shared_ptr<MetalPerformanceVisualizer> visualizer_;
};

TEST_F(MetalPerformanceVisualizerTest, GenerateReportsTest) {
  // Add some test data
  monitor_->BeginKernelProfiling("viz_kernel", nullptr);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  monitor_->EndKernelProfiling("viz_kernel", nullptr);
  
  monitor_->TrackMemoryAllocation(reinterpret_cast<void*>(0x1000), 1024, false, "viz_buffer");
  
  // Test JSON export
  std::string json_path = "/tmp/test_perf_report.json";
  EXPECT_TRUE(visualizer_->ExportMetrics(VisualizationFormat::JSON, json_path).IsOk());
  
  // Test CSV export
  std::string csv_path = "/tmp/test_perf_report.csv";
  EXPECT_TRUE(visualizer_->ExportMetrics(VisualizationFormat::CSV, csv_path).IsOk());
  
  // Test Markdown export
  std::string md_path = "/tmp/test_perf_report.md";
  EXPECT_TRUE(visualizer_->ExportMetrics(VisualizationFormat::MARKDOWN, md_path).IsOk());
}

TEST_F(MetalPerformanceVisualizerTest, TerminalVisualizationTest) {
  // Add test data
  monitor_->StartMonitoring();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  
  for (int i = 0; i < 3; ++i) {
    monitor_->BeginKernelProfiling("term_kernel_" + std::to_string(i), nullptr);
    std::this_thread::sleep_for(std::chrono::milliseconds(10 * (i + 1)));
    monitor_->EndKernelProfiling("term_kernel_" + std::to_string(i), nullptr);
  }
  
  // These should not crash
  visualizer_->PrintPerformanceSummary();
  visualizer_->PrintKernelProfilingTable();
  visualizer_->PrintMemoryStatistics();
  visualizer_->PrintBottlenecksAndRecommendations();
  
  monitor_->StopMonitoring();
}

TEST_F(MetalPerformanceVisualizerTest, ComparisonVisualizerTest) {
  PerformanceComparisonVisualizer comparator;
  
  // Create two analysis results
  MetalPerformanceMonitor::PerformanceAnalysis analysis1;
  analysis1.avg_gpu_utilization = 0.75;
  analysis1.avg_memory_bandwidth_gbps = 350.0;
  analysis1.avg_kernel_execution_time_ms = 5.5;
  
  MetalPerformanceMonitor::PerformanceAnalysis analysis2;
  analysis2.avg_gpu_utilization = 0.85;
  analysis2.avg_memory_bandwidth_gbps = 400.0;
  analysis2.avg_kernel_execution_time_ms = 4.2;
  
  comparator.AddDataset("Config A", analysis1);
  comparator.AddDataset("Config B", analysis2);
  
  auto table = comparator.GenerateComparisonTable();
  EXPECT_FALSE(table.empty());
  EXPECT_NE(table.find("Config A"), std::string::npos);
  EXPECT_NE(table.find("Config B"), std::string::npos);
}

TEST_F(MetalPerformanceVisualizerTest, ReportSchedulerTest) {
  VisualizationConfig config;
  config.format = VisualizationFormat::JSON;
  config.output_path = "/tmp/scheduled_report.json";
  
  auto scheduled_visualizer = std::make_shared<MetalPerformanceVisualizer>(monitor_, config);
  PerformanceReportScheduler scheduler(scheduled_visualizer, 1);  // 1 second interval
  
  scheduler.Start();
  
  // Wait for at least one report to be generated
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  
  scheduler.Stop();
  
  // Check if file was created
  std::ifstream file(config.output_path);
  EXPECT_TRUE(file.is_open());
  file.close();
}

}}  // namespace triton::metal