#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <thread>
#include <iomanip>
#include <cstring>
#include <Accelerate/Accelerate.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <sys/types.h>
#include <sys/sysctl.h>

class RealTimeMonitor {
private:
    mach_timebase_info_data_t timebase_info;
    uint64_t start_time;
    uint64_t start_cpu_time;
    
public:
    RealTimeMonitor() {
        mach_timebase_info(&timebase_info);
    }
    
    void start() {
        start_time = mach_absolute_time();
        start_cpu_time = get_cpu_time();
    }
    
    void print_status(const std::string& operation, int iteration, int total) {
        uint64_t current_time = mach_absolute_time();
        uint64_t elapsed_ns = (current_time - start_time) * timebase_info.numer / timebase_info.denom;
        double elapsed_s = elapsed_ns / 1e9;
        
        uint64_t cpu_time = get_cpu_time() - start_cpu_time;
        double cpu_percent = (cpu_time / 1e9) / elapsed_s * 100.0;
        
        std::cout << "\r" << operation << ": " << iteration << "/" << total 
                  << " | Time: " << std::fixed << std::setprecision(2) << elapsed_s << "s"
                  << " | CPU: " << std::setprecision(1) << cpu_percent << "%  " << std::flush;
    }
    
private:
    uint64_t get_cpu_time() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        return usage.ru_utime.tv_sec * 1e9 + usage.ru_utime.tv_usec * 1e3;
    }
};

void benchmark_with_monitoring(const std::string& name, int size, int iterations) {
    std::cout << "\n=== " << name << " (Matrix size: " << size << "x" << size << ") ===" << std::endl;
    
    // Allocate aligned memory for optimal AMX performance
    float* A = (float*)aligned_alloc(64, size * size * sizeof(float));
    float* B = (float*)aligned_alloc(64, size * size * sizeof(float));
    float* C = (float*)aligned_alloc(64, size * size * sizeof(float));
    
    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < size * size; ++i) {
        A[i] = dis(gen);
        B[i] = dis(gen);
        C[i] = 0.0f;
    }
    
    // Real-time monitoring
    RealTimeMonitor monitor;
    monitor.start();
    
    // Warmup
    std::cout << "Warming up..." << std::flush;
    for (int i = 0; i < 3; ++i) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    size, size, size, 1.0f, A, size, B, size, 0.0f, C, size);
    }
    std::cout << " done" << std::endl;
    
    // Benchmark with monitoring
    auto bench_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    size, size, size, 1.0f, A, size, B, size, 0.0f, C, size);
        
        monitor.print_status("Progress", i + 1, iterations);
    }
    
    auto bench_end = std::chrono::high_resolution_clock::now();
    std::cout << std::endl;
    
    // Calculate performance
    double elapsed = std::chrono::duration<double>(bench_end - bench_start).count();
    double gflops = (2.0 * size * size * size * iterations) / (elapsed * 1e9);
    double bandwidth = (3.0 * size * size * sizeof(float) * iterations) / (elapsed * 1e9);
    
    std::cout << "Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
    std::cout << "Memory Bandwidth: " << bandwidth << " GB/s" << std::endl;
    std::cout << "Time per iteration: " << (elapsed / iterations) * 1000 << " ms" << std::endl;
    
    // Verify result correctness (spot check)
    float expected = A[0] * size * B[0];  // Simplified check
    std::cout << "Result verification: C[0,0] = " << C[0] << " (non-zero: " 
              << (std::abs(C[0]) > 0.0001 ? "✓" : "✗") << ")" << std::endl;
    
    free(A);
    free(B);
    free(C);
}

void check_system_state() {
    std::cout << "\n=== System State Check ===" << std::endl;
    
    // CPU frequency
    size_t size = sizeof(uint64_t);
    uint64_t freq;
    if (sysctlbyname("hw.cpufrequency", &freq, &size, NULL, 0) == 0) {
        std::cout << "CPU Frequency: " << (freq / 1e9) << " GHz" << std::endl;
    }
    
    // Thermal state
    int thermal_state;
    size = sizeof(thermal_state);
    if (sysctlbyname("machdep.xcpm.cpu_thermal_level", &thermal_state, &size, NULL, 0) == 0) {
        std::cout << "Thermal State: " << thermal_state << std::endl;
    }
    
    // Memory pressure
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t host_size = sizeof(vm_stat) / sizeof(natural_t);
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm_stat, &host_size) == KERN_SUCCESS) {
        double total_pages = vm_stat.free_count + vm_stat.active_count + vm_stat.inactive_count + 
                            vm_stat.wire_count + vm_stat.compressor_page_count;
        double pressure = (vm_stat.compressor_page_count / total_pages) * 100.0;
        std::cout << "Memory Pressure: " << std::fixed << std::setprecision(1) << pressure << "%" << std::endl;
    }
}

int main() {
    std::cout << "Apple Silicon Hardware Acceleration Verification" << std::endl;
    std::cout << "===============================================" << std::endl;
    
    // System info
    char cpu_brand[256];
    size_t size = sizeof(cpu_brand);
    sysctlbyname("machdep.cpu.brand_string", &cpu_brand, &size, NULL, 0);
    std::cout << "CPU: " << cpu_brand << std::endl;
    
    // Check initial system state
    check_system_state();
    
    // Run benchmarks with real-time monitoring
    std::vector<int> sizes = {256, 512, 1024, 2048};
    
    for (int size : sizes) {
        benchmark_with_monitoring("Accelerate SGEMM", size, 20);
        
        // Cool down between tests
        std::cout << "\nCooling down..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // Check system state after benchmark
        check_system_state();
    }
    
    // Final analysis
    std::cout << "\n=== Analysis ===" << std::endl;
    std::cout << "Expected indicators of AMX usage:" << std::endl;
    std::cout << "1. Performance >1000 GFLOPS for large matrices" << std::endl;
    std::cout << "2. Low CPU usage (<50%) during computation" << std::endl;
    std::cout << "3. High memory bandwidth (>50 GB/s)" << std::endl;
    std::cout << "4. Minimal thermal impact" << std::endl;
    
    return 0;
}