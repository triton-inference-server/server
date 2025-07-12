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

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "metal_performance_monitor.h"

namespace triton { namespace metal {

// Performance visualization formats
enum class VisualizationFormat {
  HTML,      // Interactive HTML dashboard
  JSON,      // Raw JSON data
  CSV,       // CSV for spreadsheet analysis
  MARKDOWN,  // Markdown report
  TERMINAL   // Terminal-based visualization
};

// Visualization configuration
struct VisualizationConfig {
  VisualizationFormat format = VisualizationFormat::HTML;
  std::string output_path = "metal_performance_report.html";
  bool include_raw_data = false;
  bool auto_refresh = true;
  uint32_t refresh_interval_ms = 1000;
  size_t max_data_points = 1000;
  bool dark_theme = true;
};

// Performance report generator
class MetalPerformanceVisualizer {
 public:
  explicit MetalPerformanceVisualizer(
      std::shared_ptr<MetalPerformanceMonitor> monitor,
      const VisualizationConfig& config = VisualizationConfig());
  ~MetalPerformanceVisualizer();

  // Generate performance report
  Status GenerateReport();
  Status GenerateReport(const std::string& output_path);

  // Generate live dashboard (HTML only)
  Status StartLiveDashboard(uint16_t port = 8080);
  Status StopLiveDashboard();

  // Export data in various formats
  Status ExportMetrics(VisualizationFormat format, const std::string& output_path);

  // Generate specific visualizations
  std::string GenerateGPUUtilizationChart();
  std::string GenerateMemoryUsageChart();
  std::string GenerateKernelPerformanceChart();
  std::string GenerateBandwidthChart();
  std::string GeneratePowerThermalChart();

  // Terminal visualization
  void PrintPerformanceSummary();
  void PrintKernelProfilingTable();
  void PrintMemoryStatistics();
  void PrintBottlenecksAndRecommendations();

 private:
  // HTML generation helpers
  std::string GenerateHTMLDashboard();
  std::string GenerateHTMLHeader();
  std::string GenerateHTMLCharts();
  std::string GenerateHTMLTables();
  std::string GenerateHTMLFooter();
  std::string GenerateChartJS(const std::string& chart_id, 
                               const std::string& chart_type,
                               const std::string& data_json);

  // JSON generation helpers
  std::string GenerateJSONReport();
  std::string MetricsToJSON(const std::vector<MetalPerformanceMetric>& metrics);
  std::string ProfilingDataToJSON(const std::vector<KernelProfilingData>& data);

  // CSV generation helpers
  std::string GenerateCSVReport();
  std::string MetricsToCSV(const std::vector<MetalPerformanceMetric>& metrics);
  std::string ProfilingDataToCSV(const std::vector<KernelProfilingData>& data);

  // Markdown generation helpers
  std::string GenerateMarkdownReport();
  std::string GenerateMarkdownTable(const std::vector<std::vector<std::string>>& data);

  // Terminal visualization helpers
  std::string GenerateASCIIChart(const std::vector<double>& values,
                                 size_t width = 60, size_t height = 20);
  std::string GenerateProgressBar(double value, size_t width = 30);
  std::string FormatTableRow(const std::vector<std::string>& columns,
                             const std::vector<size_t>& widths);

  // Data processing helpers
  std::vector<double> ExtractMetricValues(MetalCounterType type, size_t count);
  std::map<std::string, double> AggregateKernelTimes();
  std::vector<std::pair<std::string, double>> GetTopKernelsByTime(size_t count = 10);

  // Live dashboard implementation
  class DashboardServer;
  std::unique_ptr<DashboardServer> dashboard_server_;

 private:
  std::shared_ptr<MetalPerformanceMonitor> monitor_;
  VisualizationConfig config_;
};

// Utility class for performance report scheduling
class PerformanceReportScheduler {
 public:
  PerformanceReportScheduler(
      std::shared_ptr<MetalPerformanceVisualizer> visualizer,
      uint64_t interval_seconds = 60);
  ~PerformanceReportScheduler();

  void Start();
  void Stop();
  void SetInterval(uint64_t interval_seconds);

 private:
  void SchedulerThread();

 private:
  std::shared_ptr<MetalPerformanceVisualizer> visualizer_;
  uint64_t interval_seconds_;
  std::atomic<bool> is_running_{false};
  std::unique_ptr<std::thread> scheduler_thread_;
};

// Performance comparison visualizer
class PerformanceComparisonVisualizer {
 public:
  PerformanceComparisonVisualizer();
  ~PerformanceComparisonVisualizer();

  // Add performance data for comparison
  void AddDataset(const std::string& name,
                  const MetalPerformanceMonitor::PerformanceAnalysis& analysis);

  // Generate comparison reports
  std::string GenerateComparisonChart(const std::string& metric_name);
  std::string GenerateComparisonTable();
  std::string GenerateComparisonReport(VisualizationFormat format);

  // A/B test visualization
  std::string VisualizeABTestResult(
      const MetalPerformanceMonitor::ABTestResult& result);

 private:
  std::map<std::string, MetalPerformanceMonitor::PerformanceAnalysis> datasets_;
};

}}  // namespace triton::metal