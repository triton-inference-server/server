// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Comprehensive Performance Benchmarks for Apple Silicon

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

#include "../apple/amx_provider.h"
#include "../apple/amx_kernels.h"
#include "../apple/amx_metal_interop.h"
#include "../apple/ane_provider.h"
#include "../metal/metal_backend_utils.h"
#include "../metal/kernels/gemm_kernel.h"

using namespace triton::apple;
using namespace triton::metal;

// Benchmark configuration
struct BenchmarkConfig {
    bool enable_amx = true;
    bool enable_metal = true;
    bool enable_ane = true;
    bool enable_cpu = true;
    int warmup_iterations = 10;
    int benchmark_iterations = 100;
    bool export_csv = true;
    bool export_json = true;
    std::string output_dir = "./benchmark_results";
};

// Benchmark result
struct BenchmarkResult {
    std::string operation;
    std::string processor;
    std::vector<size_t> dimensions;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double std_dev_ms;
    double gflops;
    double gb_per_sec;
    double power_watts;
    double efficiency_gflops_per_watt;
    size_t memory_bytes;
};

// Benchmark suite
class AppleSiliconBenchmarks {
public:
    AppleSiliconBenchmarks(const BenchmarkConfig& config) : config_(config) {
        InitializeProviders();
    }
    
    void RunAllBenchmarks() {
        std::cout << "\n========================================\n";
        std::cout << "Apple Silicon Performance Benchmarks\n";
        std::cout << "========================================\n\n";
        
        // GEMM benchmarks
        RunGEMMBenchmarks();
        
        // Convolution benchmarks
        RunConvolutionBenchmarks();
        
        // Transformer benchmarks
        RunTransformerBenchmarks();
        
        // Memory bandwidth benchmarks
        RunMemoryBenchmarks();
        
        // Mixed precision benchmarks
        RunMixedPrecisionBenchmarks();
        
        // Power efficiency benchmarks
        RunPowerBenchmarks();
        
        // Export results
        ExportResults();
        
        // Print summary
        PrintSummary();
    }
    
private:
    BenchmarkConfig config_;
    std::vector<BenchmarkResult> results_;
    
    void InitializeProviders() {
        // Initialize AMX
        if (config_.enable_amx) {
            auto err = AMXProvider::Instance().Initialize();
            if (err) {
                std::cout << "Warning: AMX initialization failed\n";
                config_.enable_amx = false;
                TRITONSERVER_ErrorDelete(err);
            }
        }
        
        // Initialize Metal
        if (config_.enable_metal) {
            metal::MetalBackendUtils::Initialize();
        }
        
        // Initialize ANE
        if (config_.enable_ane) {
            auto err = ANEProvider::Instance().Initialize();
            if (err) {
                std::cout << "Warning: ANE initialization failed\n";
                config_.enable_ane = false;
                TRITONSERVER_ErrorDelete(err);
            }
        }
        
        // Initialize interop
        AMXMetalInterop::Instance().Initialize();
    }
    
    // ======================
    // GEMM Benchmarks
    // ======================
    
    void RunGEMMBenchmarks() {
        std::cout << "\n--- GEMM Benchmarks ---\n";
        
        std::vector<std::vector<size_t>> gemm_sizes = {
            // Small (AMX-friendly)
            {32, 32, 32},
            {64, 64, 64},
            {128, 128, 128},
            
            // Medium
            {256, 256, 256},
            {512, 512, 512},
            
            // Large (GPU-friendly)
            {1024, 1024, 1024},
            {2048, 2048, 2048},
            {4096, 4096, 4096},
            
            // Rectangular
            {1024, 256, 512},
            {4096, 1024, 256},
            
            // Transformer-like
            {512, 768, 768},    // BERT
            {1024, 1024, 1024}, // GPT-2
            {2048, 1600, 1600}, // GPT-3 like
        };
        
        for (const auto& size : gemm_sizes) {
            size_t M = size[0], N = size[1], K = size[2];
            std::cout << "\nGEMM " << M << "x" << N << "x" << K << ":\n";
            
            // Allocate matrices
            auto A = AllocateMatrix(M * K);
            auto B = AllocateMatrix(K * N);
            auto C = AllocateMatrix(M * N);
            
            // Initialize with random data
            InitializeRandom(A.get(), M * K);
            InitializeRandom(B.get(), K * N);
            
            // Benchmark on different processors
            if (config_.enable_cpu) {
                BenchmarkCPUGEMM(A.get(), B.get(), C.get(), M, N, K);
            }
            
            if (config_.enable_amx) {
                BenchmarkAMXGEMM(A.get(), B.get(), C.get(), M, N, K);
            }
            
            if (config_.enable_metal) {
                BenchmarkMetalGEMM(A.get(), B.get(), C.get(), M, N, K);
            }
            
            // Benchmark hybrid execution
            BenchmarkHybridGEMM(A.get(), B.get(), C.get(), M, N, K);
        }
    }
    
    void BenchmarkAMXGEMM(float* A, float* B, float* C, 
                         size_t M, size_t N, size_t K) {
        auto& provider = AMXProvider::Instance();
        
        // Warmup
        for (int i = 0; i < config_.warmup_iterations; ++i) {
            provider.ExecuteGEMM(A, B, C, M, N, K);
        }
        
        // Benchmark
        std::vector<double> times;
        for (int i = 0; i < config_.benchmark_iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            provider.ExecuteGEMM(A, B, C, M, N, K);
            auto end = std::chrono::high_resolution_clock::now();
            
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            times.push_back(time_ms);
        }
        
        // Calculate statistics
        BenchmarkResult result;
        result.operation = "GEMM";
        result.processor = "AMX";
        result.dimensions = {M, N, K};
        CalculateStats(times, result);
        
        // Calculate GFLOPS
        double ops = 2.0 * M * N * K;
        result.gflops = (ops / 1e9) / (result.avg_time_ms / 1000.0);
        result.memory_bytes = (M * K + K * N + M * N) * sizeof(float);
        result.gb_per_sec = (result.memory_bytes / 1e9) / (result.avg_time_ms / 1000.0);
        
        results_.push_back(result);
        PrintResult(result);
    }
    
    // ======================
    // Convolution Benchmarks
    // ======================
    
    void RunConvolutionBenchmarks() {
        std::cout << "\n--- Convolution Benchmarks ---\n";
        
        struct ConvConfig {
            size_t batch, height, width, in_channels, out_channels;
            size_t kernel_h, kernel_w;
            size_t stride;
            std::string name;
        };
        
        std::vector<ConvConfig> conv_configs = {
            // Common CNN layers
            {1, 224, 224, 3, 64, 7, 7, 2, "ResNet-FirstLayer"},
            {1, 56, 56, 64, 64, 3, 3, 1, "ResNet-Conv3x3"},
            {1, 56, 56, 64, 256, 1, 1, 1, "ResNet-Conv1x1"},
            
            // MobileNet depthwise
            {1, 112, 112, 32, 32, 3, 3, 1, "MobileNet-Depthwise"},
            
            // Large convolutions
            {8, 224, 224, 3, 64, 3, 3, 1, "Batch8-Conv3x3"},
            {16, 112, 112, 64, 128, 3, 3, 1, "Batch16-Conv3x3"},
        };
        
        for (const auto& config : conv_configs) {
            std::cout << "\n" << config.name << ":\n";
            BenchmarkConvolution(config);
        }
    }
    
    // ======================
    // Transformer Benchmarks
    // ======================
    
    void RunTransformerBenchmarks() {
        std::cout << "\n--- Transformer Benchmarks ---\n";
        
        if (!config_.enable_ane) {
            std::cout << "ANE not available, skipping transformer benchmarks\n";
            return;
        }
        
        struct TransformerConfig {
            std::string name;
            size_t batch_size;
            size_t seq_length;
            size_t hidden_dim;
            size_t num_heads;
            size_t num_layers;
        };
        
        std::vector<TransformerConfig> configs = {
            {"BERT-Base", 1, 128, 768, 12, 12},
            {"BERT-Base-Long", 1, 512, 768, 12, 12},
            {"GPT-2-Small", 1, 1024, 768, 12, 12},
            {"GPT-2-Medium", 1, 1024, 1024, 16, 24},
            {"T5-Small", 1, 512, 512, 8, 6},
        };
        
        for (const auto& config : configs) {
            std::cout << "\n" << config.name << ":\n";
            BenchmarkTransformer(config);
        }
    }
    
    // ======================
    // Memory Benchmarks
    // ======================
    
    void RunMemoryBenchmarks() {
        std::cout << "\n--- Memory Bandwidth Benchmarks ---\n";
        
        std::vector<size_t> sizes = {
            1 * 1024 * 1024,      // 1 MB
            10 * 1024 * 1024,     // 10 MB
            100 * 1024 * 1024,    // 100 MB
            1024 * 1024 * 1024,   // 1 GB
        };
        
        for (size_t size : sizes) {
            std::cout << "\nMemory size: " << (size / (1024.0 * 1024.0)) << " MB\n";
            
            // Test different access patterns
            BenchmarkSequentialAccess(size);
            BenchmarkStridedAccess(size);
            BenchmarkRandomAccess(size);
            
            // Test unified memory transfer
            if (config_.enable_amx && config_.enable_metal) {
                BenchmarkUnifiedMemory(size);
            }
        }
    }
    
    // ======================
    // Mixed Precision Benchmarks
    // ======================
    
    void RunMixedPrecisionBenchmarks() {
        std::cout << "\n--- Mixed Precision Benchmarks ---\n";
        
        size_t M = 1024, N = 1024, K = 1024;
        
        // FP32 baseline
        BenchmarkPrecision<float>("FP32", M, N, K);
        
        // FP16
        BenchmarkPrecision<uint16_t>("FP16", M, N, K);
        
        // INT8
        BenchmarkPrecision<int8_t>("INT8", M, N, K);
    }
    
    // ======================
    // Power Efficiency Benchmarks
    // ======================
    
    void RunPowerBenchmarks() {
        std::cout << "\n--- Power Efficiency Benchmarks ---\n";
        
        // Run sustained workload and measure power
        std::cout << "Running 30-second sustained workload...\n";
        
        struct PowerResult {
            std::string processor;
            double avg_power_watts;
            double peak_power_watts;
            double operations_per_watt;
        };
        
        std::vector<PowerResult> power_results;
        
        // Test each processor
        if (config_.enable_amx) {
            auto result = MeasurePowerAMX();
            power_results.push_back(result);
        }
        
        if (config_.enable_metal) {
            auto result = MeasurePowerMetal();
            power_results.push_back(result);
        }
        
        if (config_.enable_ane) {
            auto result = MeasurePowerANE();
            power_results.push_back(result);
        }
        
        // Print power comparison
        std::cout << "\nPower Efficiency Summary:\n";
        std::cout << std::setw(15) << "Processor" 
                  << std::setw(15) << "Avg Power (W)"
                  << std::setw(15) << "Peak Power (W)"
                  << std::setw(20) << "Ops/Watt\n";
        std::cout << std::string(65, '-') << "\n";
        
        for (const auto& result : power_results) {
            std::cout << std::setw(15) << result.processor
                      << std::setw(15) << std::fixed << std::setprecision(1) 
                      << result.avg_power_watts
                      << std::setw(15) << result.peak_power_watts
                      << std::setw(20) << std::scientific << std::setprecision(2)
                      << result.operations_per_watt << "\n";
        }
    }
    
    // ======================
    // Helper Functions
    // ======================
    
    std::unique_ptr<float[]> AllocateMatrix(size_t size) {
        return std::unique_ptr<float[]>(new float[size]);
    }
    
    void InitializeRandom(float* data, size_t size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
    }
    
    void CalculateStats(const std::vector<double>& times, BenchmarkResult& result) {
        result.min_time_ms = *std::min_element(times.begin(), times.end());
        result.max_time_ms = *std::max_element(times.begin(), times.end());
        
        double sum = 0.0;
        for (double t : times) {
            sum += t;
        }
        result.avg_time_ms = sum / times.size();
        
        // Calculate standard deviation
        double variance = 0.0;
        for (double t : times) {
            variance += (t - result.avg_time_ms) * (t - result.avg_time_ms);
        }
        result.std_dev_ms = std::sqrt(variance / times.size());
    }
    
    void PrintResult(const BenchmarkResult& result) {
        std::cout << "  " << result.processor << ": "
                  << std::fixed << std::setprecision(2)
                  << result.avg_time_ms << " ms"
                  << " (" << result.gflops << " GFLOPS, "
                  << result.gb_per_sec << " GB/s)\n";
    }
    
    // ======================
    // Export Functions
    // ======================
    
    void ExportResults() {
        if (config_.export_csv) {
            ExportCSV();
        }
        
        if (config_.export_json) {
            ExportJSON();
        }
    }
    
    void ExportCSV() {
        std::ofstream file(config_.output_dir + "/benchmark_results.csv");
        
        // Header
        file << "Operation,Processor,Dimensions,Avg_Time_ms,Min_Time_ms,Max_Time_ms,"
             << "Std_Dev_ms,GFLOPS,GB_per_sec,Power_W,GFLOPS_per_W,Memory_MB\n";
        
        // Data
        for (const auto& r : results_) {
            file << r.operation << ","
                 << r.processor << ",";
            
            // Dimensions
            file << "\"";
            for (size_t i = 0; i < r.dimensions.size(); ++i) {
                if (i > 0) file << "x";
                file << r.dimensions[i];
            }
            file << "\",";
            
            file << r.avg_time_ms << ","
                 << r.min_time_ms << ","
                 << r.max_time_ms << ","
                 << r.std_dev_ms << ","
                 << r.gflops << ","
                 << r.gb_per_sec << ","
                 << r.power_watts << ","
                 << r.efficiency_gflops_per_watt << ","
                 << (r.memory_bytes / (1024.0 * 1024.0)) << "\n";
        }
        
        file.close();
        std::cout << "\nResults exported to: " << config_.output_dir << "/benchmark_results.csv\n";
    }
    
    void ExportJSON() {
        std::ofstream file(config_.output_dir + "/benchmark_results.json");
        
        file << "{\n  \"results\": [\n";
        
        for (size_t i = 0; i < results_.size(); ++i) {
            const auto& r = results_[i];
            
            file << "    {\n";
            file << "      \"operation\": \"" << r.operation << "\",\n";
            file << "      \"processor\": \"" << r.processor << "\",\n";
            file << "      \"dimensions\": [";
            
            for (size_t j = 0; j < r.dimensions.size(); ++j) {
                if (j > 0) file << ", ";
                file << r.dimensions[j];
            }
            file << "],\n";
            
            file << "      \"avg_time_ms\": " << r.avg_time_ms << ",\n";
            file << "      \"min_time_ms\": " << r.min_time_ms << ",\n";
            file << "      \"max_time_ms\": " << r.max_time_ms << ",\n";
            file << "      \"std_dev_ms\": " << r.std_dev_ms << ",\n";
            file << "      \"gflops\": " << r.gflops << ",\n";
            file << "      \"gb_per_sec\": " << r.gb_per_sec << ",\n";
            file << "      \"power_watts\": " << r.power_watts << ",\n";
            file << "      \"efficiency_gflops_per_watt\": " << r.efficiency_gflops_per_watt << ",\n";
            file << "      \"memory_mb\": " << (r.memory_bytes / (1024.0 * 1024.0)) << "\n";
            file << "    }";
            
            if (i < results_.size() - 1) {
                file << ",";
            }
            file << "\n";
        }
        
        file << "  ]\n}\n";
        file.close();
        
        std::cout << "Results exported to: " << config_.output_dir << "/benchmark_results.json\n";
    }
    
    void PrintSummary() {
        std::cout << "\n========================================\n";
        std::cout << "Benchmark Summary\n";
        std::cout << "========================================\n";
        
        // Find best processor for each operation type
        std::unordered_map<std::string, std::pair<std::string, double>> best_perf;
        
        for (const auto& r : results_) {
            auto key = r.operation;
            if (best_perf.find(key) == best_perf.end() || 
                r.gflops > best_perf[key].second) {
                best_perf[key] = {r.processor, r.gflops};
            }
        }
        
        std::cout << "\nBest Performance by Operation:\n";
        for (const auto& [op, perf] : best_perf) {
            std::cout << "  " << op << ": " << perf.first 
                      << " (" << perf.second << " GFLOPS)\n";
        }
        
        // Overall statistics
        double total_time = 0.0;
        for (const auto& r : results_) {
            total_time += r.avg_time_ms * config_.benchmark_iterations;
        }
        
        std::cout << "\nTotal benchmark time: " 
                  << (total_time / 1000.0) << " seconds\n";
        std::cout << "Total operations benchmarked: " << results_.size() << "\n";
    }
};

// Main benchmark runner
int main(int argc, char* argv[]) {
    BenchmarkConfig config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        
        if (arg == "--no-amx") {
            config.enable_amx = false;
        } else if (arg == "--no-metal") {
            config.enable_metal = false;
        } else if (arg == "--no-ane") {
            config.enable_ane = false;
        } else if (arg == "--iterations" && i + 1 < argc) {
            config.benchmark_iterations = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_dir = argv[++i];
        }
    }
    
    // Create output directory
    system(("mkdir -p " + config.output_dir).c_str());
    
    // Run benchmarks
    AppleSiliconBenchmarks benchmarks(config);
    benchmarks.RunAllBenchmarks();
    
    return 0;
}