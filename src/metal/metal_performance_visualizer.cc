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

#include "metal_performance_visualizer.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace triton { namespace metal {

namespace {

// Terminal color codes
const char* RESET = "\033[0m";
const char* BOLD = "\033[1m";
const char* RED = "\033[31m";
const char* GREEN = "\033[32m";
const char* YELLOW = "\033[33m";
const char* BLUE = "\033[34m";
const char* CYAN = "\033[36m";

// Helper to format timestamp
std::string FormatTimestamp(uint64_t timestamp_ns) {
  auto time_t = std::chrono::system_clock::to_time_t(
      std::chrono::system_clock::time_point(
          std::chrono::nanoseconds(timestamp_ns)));
  std::stringstream ss;
  ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
  return ss.str();
}

// Helper to format duration
std::string FormatDuration(double ms) {
  std::stringstream ss;
  if (ms < 1.0) {
    ss << std::fixed << std::setprecision(2) << (ms * 1000) << " μs";
  } else if (ms < 1000.0) {
    ss << std::fixed << std::setprecision(2) << ms << " ms";
  } else {
    ss << std::fixed << std::setprecision(2) << (ms / 1000) << " s";
  }
  return ss.str();
}

// Helper to format bytes
std::string FormatBytes(size_t bytes) {
  const char* units[] = {"B", "KB", "MB", "GB", "TB"};
  int unit_index = 0;
  double size = static_cast<double>(bytes);
  
  while (size >= 1024 && unit_index < 4) {
    size /= 1024;
    unit_index++;
  }
  
  std::stringstream ss;
  ss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
  return ss.str();
}

}  // namespace

MetalPerformanceVisualizer::MetalPerformanceVisualizer(
    std::shared_ptr<MetalPerformanceMonitor> monitor,
    const VisualizationConfig& config)
    : monitor_(monitor), config_(config) {}

MetalPerformanceVisualizer::~MetalPerformanceVisualizer() {
  if (dashboard_server_) {
    StopLiveDashboard();
  }
}

Status MetalPerformanceVisualizer::GenerateReport() {
  return GenerateReport(config_.output_path);
}

Status MetalPerformanceVisualizer::GenerateReport(const std::string& output_path) {
  std::string content;
  
  switch (config_.format) {
    case VisualizationFormat::HTML:
      content = GenerateHTMLDashboard();
      break;
    case VisualizationFormat::JSON:
      content = GenerateJSONReport();
      break;
    case VisualizationFormat::CSV:
      content = GenerateCSVReport();
      break;
    case VisualizationFormat::MARKDOWN:
      content = GenerateMarkdownReport();
      break;
    case VisualizationFormat::TERMINAL:
      PrintPerformanceSummary();
      PrintKernelProfilingTable();
      PrintMemoryStatistics();
      PrintBottlenecksAndRecommendations();
      return Status::Success;
  }
  
  // Write to file
  std::ofstream file(output_path);
  if (!file.is_open()) {
    return Status(Status::Code::INTERNAL, "Failed to open output file: " + output_path);
  }
  
  file << content;
  file.close();
  
  return Status::Success;
}

std::string MetalPerformanceVisualizer::GenerateHTMLDashboard() {
  std::stringstream html;
  
  html << GenerateHTMLHeader();
  html << "<body>\n";
  html << "<div class='container'>\n";
  html << "<h1>Metal Performance Dashboard</h1>\n";
  html << "<div class='timestamp'>Generated: " << FormatTimestamp(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count()) 
       << "</div>\n";
  
  // Performance summary
  auto analysis = monitor_->AnalyzePerformance();
  html << "<div class='section'>\n";
  html << "<h2>Performance Summary</h2>\n";
  html << "<div class='metrics-grid'>\n";
  html << "<div class='metric-card'>\n";
  html << "<div class='metric-label'>Average GPU Utilization</div>\n";
  html << "<div class='metric-value'>" << std::fixed << std::setprecision(1) 
       << (analysis.avg_gpu_utilization * 100) << "%</div>\n";
  html << "</div>\n";
  html << "<div class='metric-card'>\n";
  html << "<div class='metric-label'>Peak GPU Utilization</div>\n";
  html << "<div class='metric-value'>" << std::fixed << std::setprecision(1)
       << (analysis.peak_gpu_utilization * 100) << "%</div>\n";
  html << "</div>\n";
  html << "<div class='metric-card'>\n";
  html << "<div class='metric-label'>Average Memory Bandwidth</div>\n";
  html << "<div class='metric-value'>" << std::fixed << std::setprecision(1)
       << analysis.avg_memory_bandwidth_gbps << " GB/s</div>\n";
  html << "</div>\n";
  html << "<div class='metric-card'>\n";
  html << "<div class='metric-label'>Average Kernel Time</div>\n";
  html << "<div class='metric-value'>" << FormatDuration(analysis.avg_kernel_execution_time_ms) 
       << "</div>\n";
  html << "</div>\n";
  html << "</div>\n";
  html << "</div>\n";
  
  // Charts
  html << GenerateHTMLCharts();
  
  // Tables
  html << GenerateHTMLTables();
  
  // Bottlenecks and recommendations
  if (!analysis.performance_bottlenecks.empty() || 
      !analysis.optimization_recommendations.empty()) {
    html << "<div class='section'>\n";
    html << "<h2>Performance Analysis</h2>\n";
    
    if (!analysis.performance_bottlenecks.empty()) {
      html << "<div class='subsection'>\n";
      html << "<h3>Identified Bottlenecks</h3>\n";
      html << "<ul class='bottlenecks'>\n";
      for (const auto& bottleneck : analysis.performance_bottlenecks) {
        html << "<li class='bottleneck'>" << bottleneck << "</li>\n";
      }
      html << "</ul>\n";
      html << "</div>\n";
    }
    
    if (!analysis.optimization_recommendations.empty()) {
      html << "<div class='subsection'>\n";
      html << "<h3>Optimization Recommendations</h3>\n";
      html << "<ul class='recommendations'>\n";
      for (const auto& recommendation : analysis.optimization_recommendations) {
        html << "<li class='recommendation'>" << recommendation << "</li>\n";
      }
      html << "</ul>\n";
      html << "</div>\n";
    }
    
    html << "</div>\n";
  }
  
  html << "</div>\n";  // container
  html << "</body>\n";
  html << GenerateHTMLFooter();
  
  return html.str();
}

std::string MetalPerformanceVisualizer::GenerateHTMLHeader() {
  std::stringstream html;
  
  html << "<!DOCTYPE html>\n";
  html << "<html lang='en'>\n";
  html << "<head>\n";
  html << "<meta charset='UTF-8'>\n";
  html << "<meta name='viewport' content='width=device-width, initial-scale=1.0'>\n";
  html << "<title>Metal Performance Dashboard</title>\n";
  
  // Include Chart.js
  html << "<script src='https://cdn.jsdelivr.net/npm/chart.js'></script>\n";
  
  // CSS styles
  html << "<style>\n";
  if (config_.dark_theme) {
    html << ":root {\n";
    html << "  --bg-primary: #1a1a1a;\n";
    html << "  --bg-secondary: #2a2a2a;\n";
    html << "  --text-primary: #e0e0e0;\n";
    html << "  --text-secondary: #b0b0b0;\n";
    html << "  --accent: #4a9eff;\n";
    html << "  --success: #4caf50;\n";
    html << "  --warning: #ff9800;\n";
    html << "  --danger: #f44336;\n";
    html << "}\n";
  } else {
    html << ":root {\n";
    html << "  --bg-primary: #ffffff;\n";
    html << "  --bg-secondary: #f5f5f5;\n";
    html << "  --text-primary: #333333;\n";
    html << "  --text-secondary: #666666;\n";
    html << "  --accent: #2196f3;\n";
    html << "  --success: #4caf50;\n";
    html << "  --warning: #ff9800;\n";
    html << "  --danger: #f44336;\n";
    html << "}\n";
  }
  
  html << "body {\n";
  html << "  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;\n";
  html << "  background-color: var(--bg-primary);\n";
  html << "  color: var(--text-primary);\n";
  html << "  margin: 0;\n";
  html << "  padding: 0;\n";
  html << "}\n";
  
  html << ".container {\n";
  html << "  max-width: 1200px;\n";
  html << "  margin: 0 auto;\n";
  html << "  padding: 20px;\n";
  html << "}\n";
  
  html << "h1, h2, h3 {\n";
  html << "  color: var(--text-primary);\n";
  html << "}\n";
  
  html << ".timestamp {\n";
  html << "  color: var(--text-secondary);\n";
  html << "  font-size: 0.9em;\n";
  html << "  margin-bottom: 20px;\n";
  html << "}\n";
  
  html << ".section {\n";
  html << "  background-color: var(--bg-secondary);\n";
  html << "  border-radius: 8px;\n";
  html << "  padding: 20px;\n";
  html << "  margin-bottom: 20px;\n";
  html << "}\n";
  
  html << ".metrics-grid {\n";
  html << "  display: grid;\n";
  html << "  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));\n";
  html << "  gap: 20px;\n";
  html << "  margin-bottom: 20px;\n";
  html << "}\n";
  
  html << ".metric-card {\n";
  html << "  background-color: var(--bg-primary);\n";
  html << "  border-radius: 8px;\n";
  html << "  padding: 20px;\n";
  html << "  text-align: center;\n";
  html << "}\n";
  
  html << ".metric-label {\n";
  html << "  color: var(--text-secondary);\n";
  html << "  font-size: 0.9em;\n";
  html << "  margin-bottom: 10px;\n";
  html << "}\n";
  
  html << ".metric-value {\n";
  html << "  color: var(--accent);\n";
  html << "  font-size: 2em;\n";
  html << "  font-weight: bold;\n";
  html << "}\n";
  
  html << ".chart-container {\n";
  html << "  position: relative;\n";
  html << "  height: 400px;\n";
  html << "  margin-bottom: 20px;\n";
  html << "}\n";
  
  html << "table {\n";
  html << "  width: 100%;\n";
  html << "  border-collapse: collapse;\n";
  html << "  margin-top: 10px;\n";
  html << "}\n";
  
  html << "th, td {\n";
  html << "  text-align: left;\n";
  html << "  padding: 12px;\n";
  html << "  border-bottom: 1px solid var(--bg-primary);\n";
  html << "}\n";
  
  html << "th {\n";
  html << "  background-color: var(--bg-primary);\n";
  html << "  font-weight: bold;\n";
  html << "  color: var(--text-secondary);\n";
  html << "}\n";
  
  html << ".bottleneck {\n";
  html << "  color: var(--warning);\n";
  html << "  margin: 5px 0;\n";
  html << "}\n";
  
  html << ".recommendation {\n";
  html << "  color: var(--success);\n";
  html << "  margin: 5px 0;\n";
  html << "}\n";
  
  html << "</style>\n";
  html << "</head>\n";
  
  return html.str();
}

std::string MetalPerformanceVisualizer::GenerateHTMLCharts() {
  std::stringstream html;
  
  html << "<div class='section'>\n";
  html << "<h2>Performance Metrics</h2>\n";
  
  // GPU Utilization Chart
  html << "<div class='subsection'>\n";
  html << "<h3>GPU Utilization Over Time</h3>\n";
  html << "<div class='chart-container'>\n";
  html << "<canvas id='gpuUtilizationChart'></canvas>\n";
  html << "</div>\n";
  html << "</div>\n";
  
  // Memory Bandwidth Chart
  html << "<div class='subsection'>\n";
  html << "<h3>Memory Bandwidth Over Time</h3>\n";
  html << "<div class='chart-container'>\n";
  html << "<canvas id='memoryBandwidthChart'></canvas>\n";
  html << "</div>\n";
  html << "</div>\n";
  
  // Kernel Performance Chart
  html << "<div class='subsection'>\n";
  html << "<h3>Kernel Execution Times</h3>\n";
  html << "<div class='chart-container'>\n";
  html << "<canvas id='kernelPerformanceChart'></canvas>\n";
  html << "</div>\n";
  html << "</div>\n";
  
  html << "</div>\n";
  
  // Generate chart data and scripts
  html << "<script>\n";
  
  // GPU Utilization Chart
  auto gpu_values = ExtractMetricValues(MetalCounterType::GPU_UTILIZATION, 100);
  html << "const gpuUtilizationData = {\n";
  html << "  labels: [";
  for (size_t i = 0; i < gpu_values.size(); ++i) {
    if (i > 0) html << ", ";
    html << i;
  }
  html << "],\n";
  html << "  datasets: [{\n";
  html << "    label: 'GPU Utilization (%)',\n";
  html << "    data: [";
  for (size_t i = 0; i < gpu_values.size(); ++i) {
    if (i > 0) html << ", ";
    html << (gpu_values[i] * 100);
  }
  html << "],\n";
  html << "    borderColor: 'rgb(74, 158, 255)',\n";
  html << "    backgroundColor: 'rgba(74, 158, 255, 0.1)',\n";
  html << "    tension: 0.1\n";
  html << "  }]\n";
  html << "};\n";
  
  html << "new Chart(document.getElementById('gpuUtilizationChart'), {\n";
  html << "  type: 'line',\n";
  html << "  data: gpuUtilizationData,\n";
  html << "  options: {\n";
  html << "    responsive: true,\n";
  html << "    maintainAspectRatio: false,\n";
  html << "    scales: {\n";
  html << "      y: {\n";
  html << "        beginAtZero: true,\n";
  html << "        max: 100\n";
  html << "      }\n";
  html << "    }\n";
  html << "  }\n";
  html << "});\n";
  
  // Memory Bandwidth Chart
  auto bandwidth_values = ExtractMetricValues(MetalCounterType::MEMORY_BANDWIDTH, 100);
  html << "const memoryBandwidthData = {\n";
  html << "  labels: [";
  for (size_t i = 0; i < bandwidth_values.size(); ++i) {
    if (i > 0) html << ", ";
    html << i;
  }
  html << "],\n";
  html << "  datasets: [{\n";
  html << "    label: 'Memory Bandwidth (GB/s)',\n";
  html << "    data: [";
  for (size_t i = 0; i < bandwidth_values.size(); ++i) {
    if (i > 0) html << ", ";
    html << bandwidth_values[i];
  }
  html << "],\n";
  html << "    borderColor: 'rgb(255, 152, 0)',\n";
  html << "    backgroundColor: 'rgba(255, 152, 0, 0.1)',\n";
  html << "    tension: 0.1\n";
  html << "  }]\n";
  html << "};\n";
  
  html << "new Chart(document.getElementById('memoryBandwidthChart'), {\n";
  html << "  type: 'line',\n";
  html << "  data: memoryBandwidthData,\n";
  html << "  options: {\n";
  html << "    responsive: true,\n";
  html << "    maintainAspectRatio: false,\n";
  html << "    scales: {\n";
  html << "      y: {\n";
  html << "        beginAtZero: true\n";
  html << "      }\n";
  html << "    }\n";
  html << "  }\n";
  html << "});\n";
  
  // Kernel Performance Chart
  auto kernel_times = AggregateKernelTimes();
  auto top_kernels = GetTopKernelsByTime(10);
  
  html << "const kernelPerformanceData = {\n";
  html << "  labels: [";
  for (size_t i = 0; i < top_kernels.size(); ++i) {
    if (i > 0) html << ", ";
    html << "'" << top_kernels[i].first << "'";
  }
  html << "],\n";
  html << "  datasets: [{\n";
  html << "    label: 'Average Execution Time (ms)',\n";
  html << "    data: [";
  for (size_t i = 0; i < top_kernels.size(); ++i) {
    if (i > 0) html << ", ";
    html << top_kernels[i].second;
  }
  html << "],\n";
  html << "    backgroundColor: 'rgba(76, 175, 80, 0.8)',\n";
  html << "    borderColor: 'rgb(76, 175, 80)',\n";
  html << "    borderWidth: 1\n";
  html << "  }]\n";
  html << "};\n";
  
  html << "new Chart(document.getElementById('kernelPerformanceChart'), {\n";
  html << "  type: 'bar',\n";
  html << "  data: kernelPerformanceData,\n";
  html << "  options: {\n";
  html << "    responsive: true,\n";
  html << "    maintainAspectRatio: false,\n";
  html << "    scales: {\n";
  html << "      y: {\n";
  html << "        beginAtZero: true\n";
  html << "      }\n";
  html << "    }\n";
  html << "  }\n";
  html << "});\n";
  
  html << "</script>\n";
  
  return html.str();
}

std::string MetalPerformanceVisualizer::GenerateHTMLTables() {
  std::stringstream html;
  
  html << "<div class='section'>\n";
  html << "<h2>Detailed Metrics</h2>\n";
  
  // Memory Statistics Table
  auto mem_stats = monitor_->GetMemoryStatistics();
  html << "<div class='subsection'>\n";
  html << "<h3>Memory Statistics</h3>\n";
  html << "<table>\n";
  html << "<tr><th>Metric</th><th>Value</th></tr>\n";
  html << "<tr><td>Total Allocated</td><td>" << FormatBytes(mem_stats.total_allocated_bytes) << "</td></tr>\n";
  html << "<tr><td>Current Allocated</td><td>" << FormatBytes(mem_stats.current_allocated_bytes) << "</td></tr>\n";
  html << "<tr><td>Peak Allocated</td><td>" << FormatBytes(mem_stats.peak_allocated_bytes) << "</td></tr>\n";
  html << "<tr><td>Allocation Count</td><td>" << mem_stats.allocation_count << "</td></tr>\n";
  html << "<tr><td>Deallocation Count</td><td>" << mem_stats.deallocation_count << "</td></tr>\n";
  html << "<tr><td>Shared Memory</td><td>" << FormatBytes(mem_stats.shared_memory_bytes) << "</td></tr>\n";
  html << "<tr><td>Device Memory</td><td>" << FormatBytes(mem_stats.device_memory_bytes) << "</td></tr>\n";
  html << "</table>\n";
  html << "</div>\n";
  
  // Top Kernels Table
  auto top_kernels = GetTopKernelsByTime(20);
  if (!top_kernels.empty()) {
    html << "<div class='subsection'>\n";
    html << "<h3>Top Kernels by Execution Time</h3>\n";
    html << "<table>\n";
    html << "<tr><th>Kernel Name</th><th>Average Time</th><th>Total Executions</th></tr>\n";
    
    auto all_profiles = monitor_->GetKernelProfilingData();
    std::map<std::string, size_t> kernel_counts;
    for (const auto& profile : all_profiles) {
      kernel_counts[profile.kernel_name]++;
    }
    
    for (const auto& [kernel, time] : top_kernels) {
      html << "<tr>";
      html << "<td>" << kernel << "</td>";
      html << "<td>" << FormatDuration(time) << "</td>";
      html << "<td>" << kernel_counts[kernel] << "</td>";
      html << "</tr>\n";
    }
    html << "</table>\n";
    html << "</div>\n";
  }
  
  html << "</div>\n";
  
  return html.str();
}

std::string MetalPerformanceVisualizer::GenerateHTMLFooter() {
  std::stringstream html;
  
  if (config_.auto_refresh) {
    html << "<script>\n";
    html << "setTimeout(function() {\n";
    html << "  window.location.reload();\n";
    html << "}, " << config_.refresh_interval_ms << ");\n";
    html << "</script>\n";
  }
  
  html << "</html>\n";
  
  return html.str();
}

std::string MetalPerformanceVisualizer::GenerateJSONReport() {
  std::stringstream json;
  
  json << "{\n";
  
  // Timestamp
  json << "  \"timestamp\": \"" << FormatTimestamp(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count()) << "\",\n";
  
  // Performance analysis
  auto analysis = monitor_->AnalyzePerformance();
  json << "  \"performance_analysis\": {\n";
  json << "    \"avg_gpu_utilization\": " << analysis.avg_gpu_utilization << ",\n";
  json << "    \"peak_gpu_utilization\": " << analysis.peak_gpu_utilization << ",\n";
  json << "    \"avg_memory_bandwidth_gbps\": " << analysis.avg_memory_bandwidth_gbps << ",\n";
  json << "    \"peak_memory_bandwidth_gbps\": " << analysis.peak_memory_bandwidth_gbps << ",\n";
  json << "    \"avg_kernel_execution_time_ms\": " << analysis.avg_kernel_execution_time_ms << ",\n";
  json << "    \"kernel_execution_times_ms\": {\n";
  size_t count = 0;
  for (const auto& [kernel, time] : analysis.kernel_execution_times_ms) {
    if (count++ > 0) json << ",\n";
    json << "      \"" << kernel << "\": " << time;
  }
  json << "\n    },\n";
  json << "    \"performance_bottlenecks\": [\n";
  for (size_t i = 0; i < analysis.performance_bottlenecks.size(); ++i) {
    if (i > 0) json << ",\n";
    json << "      \"" << analysis.performance_bottlenecks[i] << "\"";
  }
  json << "\n    ],\n";
  json << "    \"optimization_recommendations\": [\n";
  for (size_t i = 0; i < analysis.optimization_recommendations.size(); ++i) {
    if (i > 0) json << ",\n";
    json << "      \"" << analysis.optimization_recommendations[i] << "\"";
  }
  json << "\n    ]\n";
  json << "  },\n";
  
  // Memory statistics
  auto mem_stats = monitor_->GetMemoryStatistics();
  json << "  \"memory_statistics\": {\n";
  json << "    \"total_allocated_bytes\": " << mem_stats.total_allocated_bytes << ",\n";
  json << "    \"current_allocated_bytes\": " << mem_stats.current_allocated_bytes << ",\n";
  json << "    \"peak_allocated_bytes\": " << mem_stats.peak_allocated_bytes << ",\n";
  json << "    \"allocation_count\": " << mem_stats.allocation_count << ",\n";
  json << "    \"deallocation_count\": " << mem_stats.deallocation_count << ",\n";
  json << "    \"shared_memory_bytes\": " << mem_stats.shared_memory_bytes << ",\n";
  json << "    \"device_memory_bytes\": " << mem_stats.device_memory_bytes << "\n";
  json << "  },\n";
  
  // Current metrics
  auto current_metrics = monitor_->GetCurrentMetrics();
  json << "  \"current_metrics\": " << MetricsToJSON(current_metrics);
  
  if (config_.include_raw_data) {
    json << ",\n";
    
    // Kernel profiling data
    auto kernel_profiles = monitor_->GetKernelProfilingData();
    json << "  \"kernel_profiling_data\": " << ProfilingDataToJSON(kernel_profiles);
  }
  
  json << "\n}\n";
  
  return json.str();
}

std::string MetalPerformanceVisualizer::MetricsToJSON(
    const std::vector<MetalPerformanceMetric>& metrics) {
  std::stringstream json;
  
  json << "[\n";
  for (size_t i = 0; i < metrics.size(); ++i) {
    if (i > 0) json << ",\n";
    const auto& metric = metrics[i];
    json << "    {\n";
    json << "      \"type\": \"" << static_cast<int>(metric.type) << "\",\n";
    json << "      \"name\": \"" << metric.name << "\",\n";
    json << "      \"value\": " << metric.value << ",\n";
    json << "      \"timestamp_ns\": " << metric.timestamp_ns << ",\n";
    json << "      \"labels\": {\n";
    size_t label_count = 0;
    for (const auto& [key, value] : metric.labels) {
      if (label_count++ > 0) json << ",\n";
      json << "        \"" << key << "\": \"" << value << "\"";
    }
    json << "\n      }\n";
    json << "    }";
  }
  json << "\n  ]";
  
  return json.str();
}

std::string MetalPerformanceVisualizer::ProfilingDataToJSON(
    const std::vector<KernelProfilingData>& data) {
  std::stringstream json;
  
  json << "[\n";
  for (size_t i = 0; i < data.size(); ++i) {
    if (i > 0) json << ",\n";
    const auto& profile = data[i];
    json << "    {\n";
    json << "      \"kernel_name\": \"" << profile.kernel_name << "\",\n";
    json << "      \"start_time_ns\": " << profile.start_time_ns << ",\n";
    json << "      \"end_time_ns\": " << profile.end_time_ns << ",\n";
    json << "      \"gpu_start_time_ns\": " << profile.gpu_start_time_ns << ",\n";
    json << "      \"gpu_end_time_ns\": " << profile.gpu_end_time_ns << ",\n";
    json << "      \"thread_count\": " << profile.thread_count << ",\n";
    json << "      \"threadgroup_size\": " << profile.threadgroup_size << ",\n";
    json << "      \"shared_memory_bytes\": " << profile.shared_memory_bytes << "\n";
    json << "    }";
  }
  json << "\n  ]";
  
  return json.str();
}

std::string MetalPerformanceVisualizer::GenerateCSVReport() {
  std::stringstream csv;
  
  // Header
  csv << "Timestamp,Metric Type,Metric Name,Value,Labels\n";
  
  // All metrics
  for (auto type : {MetalCounterType::GPU_UTILIZATION,
                    MetalCounterType::MEMORY_BANDWIDTH,
                    MetalCounterType::KERNEL_EXECUTION_TIME,
                    MetalCounterType::MEMORY_ALLOCATION,
                    MetalCounterType::COMMAND_BUFFER_TIME,
                    MetalCounterType::POWER_CONSUMPTION,
                    MetalCounterType::THERMAL_STATE,
                    MetalCounterType::QUEUE_UTILIZATION}) {
    auto metrics = monitor_->GetMetricHistory(type);
    for (const auto& metric : metrics) {
      csv << metric.timestamp_ns << ",";
      csv << static_cast<int>(metric.type) << ",";
      csv << metric.name << ",";
      csv << metric.value << ",";
      csv << "\"";
      size_t label_count = 0;
      for (const auto& [key, value] : metric.labels) {
        if (label_count++ > 0) csv << ";";
        csv << key << "=" << value;
      }
      csv << "\"\n";
    }
  }
  
  return csv.str();
}

std::string MetalPerformanceVisualizer::GenerateMarkdownReport() {
  std::stringstream md;
  
  md << "# Metal Performance Report\n\n";
  md << "Generated: " << FormatTimestamp(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count()) << "\n\n";
  
  // Performance Summary
  auto analysis = monitor_->AnalyzePerformance();
  md << "## Performance Summary\n\n";
  md << "| Metric | Value |\n";
  md << "|--------|-------|\n";
  md << "| Average GPU Utilization | " << std::fixed << std::setprecision(1) 
     << (analysis.avg_gpu_utilization * 100) << "% |\n";
  md << "| Peak GPU Utilization | " << std::fixed << std::setprecision(1)
     << (analysis.peak_gpu_utilization * 100) << "% |\n";
  md << "| Average Memory Bandwidth | " << std::fixed << std::setprecision(1)
     << analysis.avg_memory_bandwidth_gbps << " GB/s |\n";
  md << "| Peak Memory Bandwidth | " << std::fixed << std::setprecision(1)
     << analysis.peak_memory_bandwidth_gbps << " GB/s |\n";
  md << "| Average Kernel Execution Time | " << FormatDuration(analysis.avg_kernel_execution_time_ms) << " |\n";
  md << "\n";
  
  // Memory Statistics
  auto mem_stats = monitor_->GetMemoryStatistics();
  md << "## Memory Statistics\n\n";
  md << "| Metric | Value |\n";
  md << "|--------|-------|\n";
  md << "| Total Allocated | " << FormatBytes(mem_stats.total_allocated_bytes) << " |\n";
  md << "| Current Allocated | " << FormatBytes(mem_stats.current_allocated_bytes) << " |\n";
  md << "| Peak Allocated | " << FormatBytes(mem_stats.peak_allocated_bytes) << " |\n";
  md << "| Allocation Count | " << mem_stats.allocation_count << " |\n";
  md << "| Deallocation Count | " << mem_stats.deallocation_count << " |\n";
  md << "\n";
  
  // Top Kernels
  auto top_kernels = GetTopKernelsByTime(10);
  if (!top_kernels.empty()) {
    md << "## Top Kernels by Execution Time\n\n";
    md << "| Kernel Name | Average Time |\n";
    md << "|-------------|-------------|\n";
    for (const auto& [kernel, time] : top_kernels) {
      md << "| " << kernel << " | " << FormatDuration(time) << " |\n";
    }
    md << "\n";
  }
  
  // Bottlenecks
  if (!analysis.performance_bottlenecks.empty()) {
    md << "## Identified Bottlenecks\n\n";
    for (const auto& bottleneck : analysis.performance_bottlenecks) {
      md << "- " << bottleneck << "\n";
    }
    md << "\n";
  }
  
  // Recommendations
  if (!analysis.optimization_recommendations.empty()) {
    md << "## Optimization Recommendations\n\n";
    for (const auto& recommendation : analysis.optimization_recommendations) {
      md << "- " << recommendation << "\n";
    }
    md << "\n";
  }
  
  return md.str();
}

void MetalPerformanceVisualizer::PrintPerformanceSummary() {
  auto analysis = monitor_->AnalyzePerformance();
  
  std::cout << "\n" << BOLD << "=== Metal Performance Summary ===" << RESET << "\n\n";
  
  std::cout << CYAN << "GPU Utilization:" << RESET << "\n";
  std::cout << "  Average: " << std::fixed << std::setprecision(1) 
            << (analysis.avg_gpu_utilization * 100) << "%\n";
  std::cout << "  Peak:    " << std::fixed << std::setprecision(1)
            << (analysis.peak_gpu_utilization * 100) << "%\n\n";
  
  std::cout << CYAN << "Memory Bandwidth:" << RESET << "\n";
  std::cout << "  Average: " << std::fixed << std::setprecision(1)
            << analysis.avg_memory_bandwidth_gbps << " GB/s\n";
  std::cout << "  Peak:    " << std::fixed << std::setprecision(1)
            << analysis.peak_memory_bandwidth_gbps << " GB/s\n\n";
  
  std::cout << CYAN << "Kernel Execution:" << RESET << "\n";
  std::cout << "  Average Time: " << FormatDuration(analysis.avg_kernel_execution_time_ms) << "\n\n";
}

void MetalPerformanceVisualizer::PrintKernelProfilingTable() {
  auto top_kernels = GetTopKernelsByTime(15);
  if (top_kernels.empty()) {
    return;
  }
  
  std::cout << BOLD << "=== Top Kernels by Execution Time ===" << RESET << "\n\n";
  
  // Calculate column widths
  size_t kernel_col_width = 40;
  size_t time_col_width = 15;
  size_t count_col_width = 10;
  
  // Header
  std::cout << std::left << std::setw(kernel_col_width) << "Kernel Name"
            << std::setw(time_col_width) << "Avg Time"
            << std::setw(count_col_width) << "Count" << "\n";
  std::cout << std::string(kernel_col_width + time_col_width + count_col_width, '-') << "\n";
  
  // Get execution counts
  auto all_profiles = monitor_->GetKernelProfilingData();
  std::map<std::string, size_t> kernel_counts;
  for (const auto& profile : all_profiles) {
    kernel_counts[profile.kernel_name]++;
  }
  
  // Print rows
  for (const auto& [kernel, time] : top_kernels) {
    std::string kernel_display = kernel;
    if (kernel_display.length() > kernel_col_width - 2) {
      kernel_display = kernel_display.substr(0, kernel_col_width - 5) + "...";
    }
    
    std::cout << std::left << std::setw(kernel_col_width) << kernel_display
              << std::setw(time_col_width) << FormatDuration(time)
              << std::setw(count_col_width) << kernel_counts[kernel] << "\n";
  }
  
  std::cout << "\n";
}

void MetalPerformanceVisualizer::PrintMemoryStatistics() {
  auto mem_stats = monitor_->GetMemoryStatistics();
  
  std::cout << BOLD << "=== Memory Statistics ===" << RESET << "\n\n";
  
  std::cout << "Total Allocated:   " << GREEN << FormatBytes(mem_stats.total_allocated_bytes) << RESET << "\n";
  std::cout << "Current Allocated: " << YELLOW << FormatBytes(mem_stats.current_allocated_bytes) << RESET << "\n";
  std::cout << "Peak Allocated:    " << RED << FormatBytes(mem_stats.peak_allocated_bytes) << RESET << "\n";
  std::cout << "Allocations:       " << mem_stats.allocation_count << "\n";
  std::cout << "Deallocations:     " << mem_stats.deallocation_count << "\n";
  std::cout << "Shared Memory:     " << FormatBytes(mem_stats.shared_memory_bytes) << "\n";
  std::cout << "Device Memory:     " << FormatBytes(mem_stats.device_memory_bytes) << "\n\n";
}

void MetalPerformanceVisualizer::PrintBottlenecksAndRecommendations() {
  auto analysis = monitor_->AnalyzePerformance();
  
  if (!analysis.performance_bottlenecks.empty()) {
    std::cout << BOLD << YELLOW << "=== Identified Bottlenecks ===" << RESET << "\n\n";
    for (const auto& bottleneck : analysis.performance_bottlenecks) {
      std::cout << YELLOW << "• " << bottleneck << RESET << "\n";
    }
    std::cout << "\n";
  }
  
  if (!analysis.optimization_recommendations.empty()) {
    std::cout << BOLD << GREEN << "=== Optimization Recommendations ===" << RESET << "\n\n";
    for (const auto& recommendation : analysis.optimization_recommendations) {
      std::cout << GREEN << "✓ " << recommendation << RESET << "\n";
    }
    std::cout << "\n";
  }
}

std::vector<double> MetalPerformanceVisualizer::ExtractMetricValues(
    MetalCounterType type, size_t count) {
  auto metrics = monitor_->GetMetricHistory(type, count);
  std::vector<double> values;
  values.reserve(metrics.size());
  
  for (const auto& metric : metrics) {
    values.push_back(metric.value);
  }
  
  return values;
}

std::map<std::string, double> MetalPerformanceVisualizer::AggregateKernelTimes() {
  auto profiles = monitor_->GetKernelProfilingData();
  std::map<std::string, std::vector<double>> kernel_times;
  
  for (const auto& profile : profiles) {
    double time_ms = (profile.end_time_ns - profile.start_time_ns) / 1e6;
    kernel_times[profile.kernel_name].push_back(time_ms);
  }
  
  std::map<std::string, double> avg_times;
  for (const auto& [kernel, times] : kernel_times) {
    if (!times.empty()) {
      double sum = std::accumulate(times.begin(), times.end(), 0.0);
      avg_times[kernel] = sum / times.size();
    }
  }
  
  return avg_times;
}

std::vector<std::pair<std::string, double>> 
MetalPerformanceVisualizer::GetTopKernelsByTime(size_t count) {
  auto kernel_times = AggregateKernelTimes();
  
  std::vector<std::pair<std::string, double>> sorted_kernels(
      kernel_times.begin(), kernel_times.end());
  
  std::sort(sorted_kernels.begin(), sorted_kernels.end(),
            [](const auto& a, const auto& b) {
              return a.second > b.second;
            });
  
  if (count > 0 && sorted_kernels.size() > count) {
    sorted_kernels.resize(count);
  }
  
  return sorted_kernels;
}

// PerformanceReportScheduler implementation
PerformanceReportScheduler::PerformanceReportScheduler(
    std::shared_ptr<MetalPerformanceVisualizer> visualizer,
    uint64_t interval_seconds)
    : visualizer_(visualizer), interval_seconds_(interval_seconds) {}

PerformanceReportScheduler::~PerformanceReportScheduler() {
  Stop();
}

void PerformanceReportScheduler::Start() {
  if (is_running_) {
    return;
  }
  
  is_running_ = true;
  scheduler_thread_ = std::make_unique<std::thread>(
      &PerformanceReportScheduler::SchedulerThread, this);
}

void PerformanceReportScheduler::Stop() {
  if (!is_running_) {
    return;
  }
  
  is_running_ = false;
  if (scheduler_thread_ && scheduler_thread_->joinable()) {
    scheduler_thread_->join();
  }
}

void PerformanceReportScheduler::SetInterval(uint64_t interval_seconds) {
  interval_seconds_ = interval_seconds;
}

void PerformanceReportScheduler::SchedulerThread() {
  while (is_running_) {
    // Generate report
    visualizer_->GenerateReport();
    
    // Sleep for interval
    for (uint64_t i = 0; i < interval_seconds_ && is_running_; ++i) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}

// PerformanceComparisonVisualizer implementation
PerformanceComparisonVisualizer::PerformanceComparisonVisualizer() {}

PerformanceComparisonVisualizer::~PerformanceComparisonVisualizer() {}

void PerformanceComparisonVisualizer::AddDataset(
    const std::string& name,
    const MetalPerformanceMonitor::PerformanceAnalysis& analysis) {
  datasets_[name] = analysis;
}

std::string PerformanceComparisonVisualizer::GenerateComparisonChart(
    const std::string& metric_name) {
  // Implementation would generate comparison charts
  // This is a placeholder
  return "Comparison chart for " + metric_name;
}

std::string PerformanceComparisonVisualizer::GenerateComparisonTable() {
  std::stringstream table;
  
  table << "| Dataset | GPU Util (%) | Mem BW (GB/s) | Avg Kernel (ms) |\n";
  table << "|---------|--------------|---------------|----------------|\n";
  
  for (const auto& [name, analysis] : datasets_) {
    table << "| " << name;
    table << " | " << std::fixed << std::setprecision(1) 
          << (analysis.avg_gpu_utilization * 100);
    table << " | " << std::fixed << std::setprecision(1)
          << analysis.avg_memory_bandwidth_gbps;
    table << " | " << std::fixed << std::setprecision(2)
          << analysis.avg_kernel_execution_time_ms;
    table << " |\n";
  }
  
  return table.str();
}

std::string PerformanceComparisonVisualizer::VisualizeABTestResult(
    const MetalPerformanceMonitor::ABTestResult& result) {
  std::stringstream viz;
  
  viz << "\n" << BOLD << "=== A/B Test Results ===" << RESET << "\n\n";
  viz << "Variant A: " << CYAN << result.variant_a_name << RESET << "\n";
  viz << "  Average Time: " << result.variant_a_avg_time_ms << " ms\n\n";
  viz << "Variant B: " << CYAN << result.variant_b_name << RESET << "\n";
  viz << "  Average Time: " << result.variant_b_avg_time_ms << " ms\n\n";
  
  if (result.speedup_factor > 1.0) {
    viz << GREEN << "✓ " << result.recommendation << RESET << "\n";
  } else if (result.speedup_factor < 1.0) {
    viz << YELLOW << "⚠ " << result.recommendation << RESET << "\n";
  } else {
    viz << BLUE << "= " << result.recommendation << RESET << "\n";
  }
  
  return viz.str();
}

}}  // namespace triton::metal