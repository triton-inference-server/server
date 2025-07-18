// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <atomic>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#include <Metal/Metal.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

namespace triton {
namespace metal {

// Forward declarations
class MetalBuffer;
class MetalDevice;

// Memory access pattern types
enum class AccessPattern {
    SEQUENTIAL,      // Linear access pattern
    STRIDED,        // Fixed stride access
    RANDOM,         // Random access pattern
    STREAMING,      // One-time sequential read
    TEMPORAL,       // Frequently accessed
    SPATIAL,        // Spatially local access
    UNKNOWN         // Pattern not yet determined
};

// Prefetch hint levels
enum class PrefetchHint {
    NONE,           // No prefetching
    CONSERVATIVE,   // Light prefetching
    MODERATE,       // Balanced prefetching
    AGGRESSIVE,     // Heavy prefetching
    ADAPTIVE        // Learn from access patterns
};

// Memory access statistics
struct AccessStats {
    size_t total_accesses = 0;
    size_t cache_hits = 0;
    size_t cache_misses = 0;
    size_t prefetch_hits = 0;
    size_t prefetch_misses = 0;
    std::chrono::nanoseconds total_latency{0};
    AccessPattern detected_pattern = AccessPattern::UNKNOWN;
    double stride_confidence = 0.0;
    size_t detected_stride = 0;
};

// Prefetch request
struct PrefetchRequest {
    void* address;
    size_t size;
    std::chrono::steady_clock::time_point deadline;
    PrefetchHint hint;
    int priority;
};

// Memory region info
struct MemoryRegion {
    void* base_address;
    size_t size;
    AccessPattern pattern;
    size_t access_count;
    std::chrono::steady_clock::time_point last_access;
    double temperature; // Hot/cold indicator
    std::deque<size_t> access_history; // Recent access offsets
};

// Metal memory prefetcher
class MetalMemoryPrefetcher {
public:
    MetalMemoryPrefetcher();
    ~MetalMemoryPrefetcher();
    
    // Initialize prefetcher with device
    void Initialize(MetalDevice* device);
    
    // Shutdown prefetcher
    void Shutdown();
    
    // Register memory region for tracking
    void RegisterMemoryRegion(void* base_address, size_t size,
                             AccessPattern initial_pattern = AccessPattern::UNKNOWN);
    
    // Unregister memory region
    void UnregisterMemoryRegion(void* base_address);
    
    // Record memory access
    void RecordAccess(void* address, size_t size, bool is_read = true);
    
    // Explicit prefetch request
    void Prefetch(void* address, size_t size, 
                  PrefetchHint hint = PrefetchHint::MODERATE);
    
    // Batch prefetch for multiple addresses
    void BatchPrefetch(const std::vector<std::pair<void*, size_t>>& requests,
                      PrefetchHint hint = PrefetchHint::MODERATE);
    
    // Set global prefetch policy
    void SetPrefetchPolicy(PrefetchHint policy) { global_policy_ = policy; }
    
    // Get access statistics
    AccessStats GetStats(void* base_address) const;
    AccessStats GetGlobalStats() const { return global_stats_; }
    
    // Reset statistics
    void ResetStats();
    
    // Pattern detection and prediction
    AccessPattern DetectPattern(void* base_address) const;
    void* PredictNextAccess(void* base_address, size_t& predicted_size) const;
    
    // Performance tuning
    void EnableAdaptivePrefetching(bool enable) { adaptive_enabled_ = enable; }
    void SetPrefetchDistance(size_t distance) { prefetch_distance_ = distance; }
    void SetPrefetchDegree(size_t degree) { prefetch_degree_ = degree; }
    
    // Cache management hints
    void MarkAsStreamingData(void* address, size_t size);
    void MarkAsTemporalData(void* address, size_t size);
    void EvictFromCache(void* address, size_t size);
    
#ifdef __APPLE__
    // Apple-specific optimizations
    void SetMemoryAdvise(void* address, size_t size, int advice);
    void PinMemoryPages(void* address, size_t size);
    void UnpinMemoryPages(void* address, size_t size);
#endif
    
private:
    // Pattern detection algorithms
    void UpdateAccessPattern(MemoryRegion& region, size_t offset);
    bool DetectStridePattern(const std::deque<size_t>& history, 
                            size_t& stride, double& confidence);
    bool DetectStreamingPattern(const std::deque<size_t>& history);
    
    // Prefetch execution
    void PrefetchWorker();
    void ExecutePrefetch(const PrefetchRequest& request);
    void AdaptivePrefetchAdjustment();
    
    // Memory region lookup
    MemoryRegion* FindRegion(void* address);
    const MemoryRegion* FindRegion(void* address) const;
    
    // Hardware-specific prefetch
#ifdef __APPLE__
    void MetalPrefetch(void* address, size_t size);
    void AppleUnifiedMemoryPrefetch(void* address, size_t size);
#endif
    
    // Member variables
    MetalDevice* device_ = nullptr;
    
    // Memory region tracking
    std::unordered_map<void*, MemoryRegion> regions_;
    mutable std::mutex regions_mutex_;
    
    // Prefetch queue
    std::deque<PrefetchRequest> prefetch_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // Worker thread
    std::thread prefetch_thread_;
    std::atomic<bool> running_{false};
    
    // Configuration
    PrefetchHint global_policy_ = PrefetchHint::ADAPTIVE;
    std::atomic<bool> adaptive_enabled_{true};
    std::atomic<size_t> prefetch_distance_{64 * 1024}; // 64KB ahead
    std::atomic<size_t> prefetch_degree_{4}; // Prefetch 4 blocks
    
    // Statistics
    AccessStats global_stats_;
    mutable std::mutex stats_mutex_;
    
    // Adaptive parameters
    double hit_rate_threshold_ = 0.8;
    double miss_penalty_ns_ = 100.0;
    std::chrono::steady_clock::time_point last_adaptation_;
};

// Prefetch-aware memory buffer
class PrefetchAwareBuffer : public MetalBuffer {
public:
    PrefetchAwareBuffer(size_t size, MetalMemoryPrefetcher* prefetcher);
    ~PrefetchAwareBuffer();
    
    // Override access methods to trigger prefetching
    void* Data() override;
    void Read(void* dst, size_t offset, size_t size) override;
    void Write(const void* src, size_t offset, size_t size) override;
    
    // Prefetch hints
    void SetAccessPattern(AccessPattern pattern);
    void PrefetchRange(size_t offset, size_t size);
    
private:
    MetalMemoryPrefetcher* prefetcher_;
    AccessPattern access_pattern_ = AccessPattern::UNKNOWN;
    size_t last_access_offset_ = 0;
};

// Global prefetcher instance
class PrefetcherManager {
public:
    static PrefetcherManager& Instance();
    
    MetalMemoryPrefetcher* GetPrefetcher() { return &prefetcher_; }
    
    void Enable(bool enabled) { enabled_ = enabled; }
    bool IsEnabled() const { return enabled_; }
    
    // Global configuration
    void SetPolicy(PrefetchHint policy) { prefetcher_.SetPrefetchPolicy(policy); }
    void SetAdaptive(bool adaptive) { prefetcher_.EnableAdaptivePrefetching(adaptive); }
    
private:
    PrefetcherManager() = default;
    ~PrefetcherManager() = default;
    
    MetalMemoryPrefetcher prefetcher_;
    bool enabled_ = true;
};

// Helper functions for common prefetch patterns
namespace PrefetchPatterns {
    // Matrix multiplication prefetch
    void PrefetchMatrixMultiply(void* a, void* b, void* c,
                               size_t m, size_t n, size_t k,
                               size_t element_size);
    
    // Convolution prefetch
    void PrefetchConvolution(void* input, void* kernel, void* output,
                            size_t batch, size_t height, size_t width,
                            size_t channels, size_t filters);
    
    // Reduction operation prefetch
    void PrefetchReduction(void* input, void* output,
                          size_t size, size_t element_size);
    
    // Transformer attention prefetch
    void PrefetchAttention(void* q, void* k, void* v,
                          size_t batch, size_t heads, 
                          size_t seq_len, size_t dim);
}

} // namespace metal
} // namespace triton