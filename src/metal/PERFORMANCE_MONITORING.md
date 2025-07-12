# Metal Performance Monitoring System

## Overview

The Metal Performance Monitoring System provides comprehensive performance tracking and analysis capabilities for Metal-based operations in Triton Inference Server. It enables real-time monitoring of GPU utilization, memory bandwidth, kernel execution times, and other critical performance metrics.

## Features

### Core Monitoring Capabilities

1. **GPU Utilization Tracking**
   - Real-time GPU usage percentage
   - Historical utilization data
   - Peak and average utilization metrics

2. **Memory Bandwidth Monitoring**
   - Memory transfer rates (GB/s)
   - Read/write bandwidth separation
   - Bandwidth saturation detection

3. **Kernel Execution Profiling**
   - Per-kernel execution time tracking
   - Thread and threadgroup configuration
   - Shared memory usage statistics
   - Kernel launch overhead measurement

4. **Memory Allocation Tracking**
   - Real-time memory usage
   - Allocation/deallocation patterns
   - Peak memory usage
   - Shared vs. device memory breakdown

5. **Power and Thermal Monitoring**
   - Power consumption in watts
   - Thermal state tracking
   - Thermal throttling detection

6. **Command Buffer Profiling**
   - Submission to completion latency
   - GPU vs. CPU timeline analysis
   - Queue utilization metrics

### Analysis and Visualization

1. **Performance Analysis**
   - Automatic bottleneck detection
   - Optimization recommendations
   - Performance trend analysis

2. **A/B Testing Support**
   - Kernel performance comparison
   - Statistical analysis
   - Speedup factor calculation

3. **Multiple Output Formats**
   - Interactive HTML dashboards
   - JSON data export
   - CSV for spreadsheet analysis
   - Markdown reports
   - Terminal visualization

4. **Prometheus Integration**
   - Export metrics to Prometheus
   - Grafana dashboard compatibility
   - Real-time monitoring support

## Architecture

### Components

1. **MetalPerformanceMonitor**
   - Core monitoring engine
   - Metric collection and storage
   - Real-time data sampling

2. **MetalPerformanceVisualizer**
   - Report generation
   - Data visualization
   - Export functionality

3. **PerformanceMonitorManager**
   - Global monitor registry
   - Multi-device support
   - Aggregated metrics

## Usage

### Basic Setup

```cpp
#include "metal_performance_monitor.h"
#include "metal_performance_visualizer.h"

// Create Metal device
auto device = std::make_shared<MetalDevice>();
device->Initialize();

// Configure monitoring
PerformanceMonitorConfig config;
config.enable_gpu_utilization = true;
config.enable_memory_bandwidth = true;
config.enable_kernel_profiling = true;
config.sampling_interval_ms = 100;  // 100ms

// Create monitor
auto monitor = std::make_shared<MetalPerformanceMonitor>(device, config);
monitor->Initialize();

// Start monitoring
monitor->StartMonitoring();
```

### Kernel Profiling

```cpp
// Profile kernel execution
monitor->BeginKernelProfiling("gemm_kernel", encoder);
// ... dispatch kernel ...
monitor->EndKernelProfiling("gemm_kernel", encoder);

// Get profiling data
auto profiles = monitor->GetKernelProfilingData("gemm_kernel");
for (const auto& profile : profiles) {
    double exec_time_ms = (profile.end_time_ns - profile.start_time_ns) / 1e6;
    std::cout << "Kernel execution time: " << exec_time_ms << " ms\n";
}
```

### Memory Tracking

```cpp
// Track memory allocations
void* buffer = /* allocate memory */;
monitor->TrackMemoryAllocation(buffer, size, is_shared, "buffer_name");

// ... use buffer ...

// Track deallocation
monitor->TrackMemoryDeallocation(buffer);

// Get memory statistics
auto stats = monitor->GetMemoryStatistics();
std::cout << "Current allocated: " << stats.current_allocated_bytes << " bytes\n";
std::cout << "Peak allocated: " << stats.peak_allocated_bytes << " bytes\n";
```

### Performance Analysis

```cpp
// Analyze performance over time window
auto analysis = monitor->AnalyzePerformance(5000);  // Last 5 seconds

std::cout << "Average GPU utilization: " << (analysis.avg_gpu_utilization * 100) << "%\n";
std::cout << "Peak memory bandwidth: " << analysis.peak_memory_bandwidth_gbps << " GB/s\n";

// Print bottlenecks
for (const auto& bottleneck : analysis.performance_bottlenecks) {
    std::cout << "Bottleneck: " << bottleneck << "\n";
}

// Print recommendations
for (const auto& recommendation : analysis.optimization_recommendations) {
    std::cout << "Recommendation: " << recommendation << "\n";
}
```

### A/B Testing

```cpp
// Compare two kernel implementations
auto result = monitor->CompareKernelPerformance("kernel_v1", "kernel_v2");

std::cout << "Kernel v1 average: " << result.variant_a_avg_time_ms << " ms\n";
std::cout << "Kernel v2 average: " << result.variant_b_avg_time_ms << " ms\n";
std::cout << "Speedup: " << result.speedup_factor << "x\n";
std::cout << "Recommendation: " << result.recommendation << "\n";
```

### Visualization

```cpp
// Create visualizer
VisualizationConfig viz_config;
viz_config.format = VisualizationFormat::HTML;
viz_config.output_path = "performance_report.html";
viz_config.auto_refresh = true;
viz_config.refresh_interval_ms = 1000;

auto visualizer = std::make_shared<MetalPerformanceVisualizer>(monitor, viz_config);

// Generate report
visualizer->GenerateReport();

// Terminal visualization
visualizer->PrintPerformanceSummary();
visualizer->PrintKernelProfilingTable();
visualizer->PrintMemoryStatistics();
```

### Prometheus Integration

```cpp
#ifdef TRITON_ENABLE_METRICS
// Register Prometheus metrics
auto registry = prometheus::Registry::Create();
monitor->RegisterPrometheusMetrics(registry);

// Metrics are automatically updated during monitoring
// Access via Prometheus endpoint (e.g., http://localhost:9090/metrics)
#endif
```

## Performance Metrics

### Available Metrics

| Metric | Type | Description | Unit |
|--------|------|-------------|------|
| `metal_gpu_utilization_ratio` | Gauge | GPU utilization percentage | 0-1 |
| `metal_memory_bandwidth_gbps` | Gauge | Memory bandwidth usage | GB/s |
| `metal_memory_allocated_bytes` | Gauge | Currently allocated memory | bytes |
| `metal_power_consumption_watts` | Gauge | Power consumption | watts |
| `metal_thermal_state` | Gauge | Thermal state (0-3) | enum |
| `metal_kernel_execution_total` | Counter | Total kernel executions | count |
| `metal_kernel_execution_duration_milliseconds` | Summary | Kernel execution time | ms |
| `metal_command_buffer_latency_milliseconds` | Histogram | Command buffer latency | ms |

### Metric Labels

- `device`: Metal device name
- `kernel`: Kernel name (for kernel-specific metrics)
- `type`: Memory type (shared/device)
- `identifier`: Command buffer identifier

## Optimization Recommendations

The system automatically generates optimization recommendations based on detected bottlenecks:

### GPU Compute Bound
- Consider kernel fusion to reduce launch overhead
- Optimize arithmetic intensity
- Review threadgroup configuration

### Memory Bandwidth Bound
- Optimize memory access patterns
- Use shared memory for frequently accessed data
- Consider data compression

### High Memory Usage
- Implement memory pooling
- Use streaming for large operations
- Review memory allocation patterns

### Long Kernel Execution
- Split into smaller kernels
- Optimize algorithm complexity
- Review parallelization strategy

## Configuration Options

```cpp
struct PerformanceMonitorConfig {
    bool enable_gpu_utilization = true;
    bool enable_memory_bandwidth = true;
    bool enable_kernel_profiling = true;
    bool enable_power_monitoring = true;
    bool enable_thermal_monitoring = true;
    bool enable_queue_monitoring = true;
    bool enable_detailed_profiling = false;  // Enables MTLCounterSampleBuffer
    uint64_t sampling_interval_ms = 100;     // Metric sampling interval
    size_t history_buffer_size = 10000;      // Max samples to retain
    bool enable_prometheus_export = true;     // Prometheus metrics export
};
```

## Best Practices

1. **Sampling Interval**
   - Use 100ms for general monitoring
   - Reduce to 10-50ms for detailed profiling
   - Increase to 1000ms for long-running applications

2. **Memory Management**
   - Always track allocations and deallocations in pairs
   - Use consistent context names for related allocations
   - Monitor peak memory usage to set appropriate limits

3. **Kernel Profiling**
   - Profile representative workloads
   - Collect multiple samples for statistical validity
   - Compare different implementations using A/B testing

4. **Performance Analysis**
   - Analyze over appropriate time windows
   - Consider warm-up periods
   - Look for patterns in bottlenecks

5. **Visualization**
   - Use HTML dashboards for interactive analysis
   - Export to JSON for custom processing
   - Use terminal visualization for quick checks

## Troubleshooting

### High CPU Usage from Monitoring
- Increase sampling interval
- Disable detailed profiling
- Reduce history buffer size

### Missing Metrics
- Verify Metal device supports requested counters
- Check macOS version (10.15+ required for some features)
- Ensure monitoring is started before operations

### Inaccurate Measurements
- Allow warm-up period before analysis
- Increase sample count for averages
- Check for thermal throttling

## Future Enhancements

1. **Machine Learning Integration**
   - Automatic performance anomaly detection
   - Predictive optimization suggestions
   - Workload classification

2. **Advanced Profiling**
   - Instruction-level profiling
   - Cache performance metrics
   - Warp occupancy analysis

3. **Cloud Integration**
   - Remote monitoring dashboards
   - Multi-instance aggregation
   - Performance regression tracking

## Example Output

### Terminal Visualization

```
=== Metal Performance Summary ===

GPU Utilization:
  Average: 75.3%
  Peak:    92.1%

Memory Bandwidth:
  Average: 285.6 GB/s
  Peak:    412.3 GB/s

Kernel Execution:
  Average Time: 5.42 ms

=== Top Kernels by Execution Time ===

Kernel Name                    Avg Time      Count
----------------------------------------     -----
gemm_kernel_fp32              8.32 ms       1523
conv2d_3x3_kernel            6.74 ms        987
reduce_sum_kernel            2.15 ms       3241

=== Memory Statistics ===

Total Allocated:   15.6 GB
Current Allocated: 8.2 GB
Peak Allocated:    12.4 GB
```

### HTML Dashboard

The HTML dashboard provides:
- Real-time charts for all metrics
- Interactive kernel performance analysis
- Memory usage timeline
- Bottleneck identification
- Optimization recommendations

## License

This performance monitoring system is part of the Triton Inference Server and is subject to the same license terms.