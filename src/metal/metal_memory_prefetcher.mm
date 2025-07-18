// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Metal memory prefetcher implementation

#include "metal_memory_prefetcher.h"
#include "metal_device.h"
#include "metal_memory.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#ifdef __APPLE__
#include <mach/mach_init.h>
#include <mach/vm_map.h>
#endif

namespace triton {
namespace metal {

// MetalMemoryPrefetcher implementation
MetalMemoryPrefetcher::MetalMemoryPrefetcher() {
    last_adaptation_ = std::chrono::steady_clock::now();
}

MetalMemoryPrefetcher::~MetalMemoryPrefetcher() {
    Shutdown();
}

void MetalMemoryPrefetcher::Initialize(MetalDevice* device) {
    device_ = device;
    running_ = true;
    
    // Start prefetch worker thread
    prefetch_thread_ = std::thread(&MetalMemoryPrefetcher::PrefetchWorker, this);
}

void MetalMemoryPrefetcher::Shutdown() {
    if (running_) {
        running_ = false;
        queue_cv_.notify_all();
        
        if (prefetch_thread_.joinable()) {
            prefetch_thread_.join();
        }
    }
}

void MetalMemoryPrefetcher::RegisterMemoryRegion(
    void* base_address, size_t size, AccessPattern initial_pattern) {
    
    std::lock_guard<std::mutex> lock(regions_mutex_);
    
    MemoryRegion region;
    region.base_address = base_address;
    region.size = size;
    region.pattern = initial_pattern;
    region.access_count = 0;
    region.last_access = std::chrono::steady_clock::now();
    region.temperature = 0.0;
    
    regions_[base_address] = region;
}

void MetalMemoryPrefetcher::UnregisterMemoryRegion(void* base_address) {
    std::lock_guard<std::mutex> lock(regions_mutex_);
    regions_.erase(base_address);
}

void MetalMemoryPrefetcher::RecordAccess(void* address, size_t size, bool is_read) {
    auto region = FindRegion(address);
    if (!region) return;
    
    size_t offset = static_cast<char*>(address) - static_cast<char*>(region->base_address);
    
    // Update access statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        global_stats_.total_accesses++;
        region->access_count++;
    }
    
    // Update access pattern
    UpdateAccessPattern(*region, offset);
    
    // Update temperature (hot/cold)
    auto now = std::chrono::steady_clock::now();
    auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - region->last_access).count();
    
    // Exponential decay with new access
    region->temperature = region->temperature * 0.9 + (1000.0 / (time_since_last + 1));
    region->last_access = now;
    
    // Trigger adaptive prefetching if enabled
    if (adaptive_enabled_ && region->pattern != AccessPattern::RANDOM) {
        // Predict next access
        size_t predicted_size;
        void* next_addr = PredictNextAccess(region->base_address, predicted_size);
        
        if (next_addr) {
            Prefetch(next_addr, predicted_size, PrefetchHint::ADAPTIVE);
        }
    }
}

void MetalMemoryPrefetcher::Prefetch(void* address, size_t size, PrefetchHint hint) {
    if (!enabled_ || hint == PrefetchHint::NONE) return;
    
    PrefetchRequest request;
    request.address = address;
    request.size = size;
    request.hint = hint;
    request.priority = (hint == PrefetchHint::AGGRESSIVE) ? 10 : 5;
    request.deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(100);
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        prefetch_queue_.push_back(request);
    }
    
    queue_cv_.notify_one();
}

void MetalMemoryPrefetcher::BatchPrefetch(
    const std::vector<std::pair<void*, size_t>>& requests, PrefetchHint hint) {
    
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    for (const auto& [address, size] : requests) {
        PrefetchRequest request;
        request.address = address;
        request.size = size;
        request.hint = hint;
        request.priority = 5;
        request.deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(200);
        
        prefetch_queue_.push_back(request);
    }
    
    queue_cv_.notify_one();
}

AccessStats MetalMemoryPrefetcher::GetStats(void* base_address) const {
    std::lock_guard<std::mutex> lock(regions_mutex_);
    
    auto it = regions_.find(base_address);
    if (it != regions_.end()) {
        AccessStats stats;
        stats.total_accesses = it->second.access_count;
        stats.detected_pattern = it->second.pattern;
        return stats;
    }
    
    return AccessStats{};
}

AccessPattern MetalMemoryPrefetcher::DetectPattern(void* base_address) const {
    auto region = FindRegion(base_address);
    return region ? region->pattern : AccessPattern::UNKNOWN;
}

void* MetalMemoryPrefetcher::PredictNextAccess(
    void* base_address, size_t& predicted_size) const {
    
    auto region = FindRegion(base_address);
    if (!region || region->access_history.size() < 3) {
        return nullptr;
    }
    
    predicted_size = 4096; // Default page size
    
    switch (region->pattern) {
        case AccessPattern::SEQUENTIAL: {
            // Predict next sequential block
            if (!region->access_history.empty()) {
                size_t last_offset = region->access_history.back();
                size_t next_offset = last_offset + prefetch_distance_;
                
                if (next_offset < region->size) {
                    predicted_size = std::min(prefetch_distance_, region->size - next_offset);
                    return static_cast<char*>(base_address) + next_offset;
                }
            }
            break;
        }
        
        case AccessPattern::STRIDED: {
            // Predict based on detected stride
            size_t stride;
            double confidence;
            if (DetectStridePattern(region->access_history, stride, confidence) && 
                confidence > 0.8) {
                
                size_t last_offset = region->access_history.back();
                size_t next_offset = last_offset + stride;
                
                if (next_offset < region->size) {
                    predicted_size = std::min(stride, region->size - next_offset);
                    return static_cast<char*>(base_address) + next_offset;
                }
            }
            break;
        }
        
        case AccessPattern::TEMPORAL: {
            // Prefetch frequently accessed regions
            if (region->temperature > 100.0) {
                // Hot data - prefetch surrounding areas
                size_t last_offset = region->access_history.back();
                size_t aligned_offset = (last_offset / 4096) * 4096;
                predicted_size = 16384; // 4 pages
                
                if (aligned_offset + predicted_size <= region->size) {
                    return static_cast<char*>(base_address) + aligned_offset;
                }
            }
            break;
        }
        
        default:
            break;
    }
    
    return nullptr;
}

void MetalMemoryPrefetcher::MarkAsStreamingData(void* address, size_t size) {
    auto region = FindRegion(address);
    if (region) {
        region->pattern = AccessPattern::STREAMING;
        
#ifdef __APPLE__
        // Advise kernel about streaming access
        SetMemoryAdvise(address, size, MADV_SEQUENTIAL);
#endif
    }
}

void MetalMemoryPrefetcher::MarkAsTemporalData(void* address, size_t size) {
    auto region = FindRegion(address);
    if (region) {
        region->pattern = AccessPattern::TEMPORAL;
        
#ifdef __APPLE__
        // Advise kernel to keep in cache
        SetMemoryAdvise(address, size, MADV_WILLNEED);
#endif
    }
}

#ifdef __APPLE__
void MetalMemoryPrefetcher::SetMemoryAdvise(void* address, size_t size, int advice) {
    madvise(address, size, advice);
}

void MetalMemoryPrefetcher::PinMemoryPages(void* address, size_t size) {
    // Pin pages to prevent swapping
    mlock(address, size);
}

void MetalMemoryPrefetcher::UnpinMemoryPages(void* address, size_t size) {
    munlock(address, size);
}

void MetalMemoryPrefetcher::MetalPrefetch(void* address, size_t size) {
    // On Apple Silicon, we can leverage unified memory prefetch
    if (device_ && device_->HasUnifiedMemory()) {
        // Touch pages to trigger prefetch
        volatile char* ptr = static_cast<volatile char*>(address);
        for (size_t i = 0; i < size; i += 4096) {
            volatile char dummy = ptr[i];
            (void)dummy;
        }
    }
}
#endif

void MetalMemoryPrefetcher::UpdateAccessPattern(MemoryRegion& region, size_t offset) {
    region.access_history.push_back(offset);
    
    // Keep only recent history
    if (region.access_history.size() > 32) {
        region.access_history.pop_front();
    }
    
    // Need enough history to detect patterns
    if (region.access_history.size() < 4) {
        return;
    }
    
    // Try to detect patterns
    size_t stride;
    double confidence;
    
    if (DetectStridePattern(region.access_history, stride, confidence)) {
        if (confidence > 0.9) {
            region.pattern = (stride == 1) ? AccessPattern::SEQUENTIAL : AccessPattern::STRIDED;
            region.detected_stride = stride;
            region.stride_confidence = confidence;
        }
    } else if (DetectStreamingPattern(region.access_history)) {
        region.pattern = AccessPattern::STREAMING;
    } else if (region.temperature > 50.0) {
        region.pattern = AccessPattern::TEMPORAL;
    } else {
        region.pattern = AccessPattern::RANDOM;
    }
}

bool MetalMemoryPrefetcher::DetectStridePattern(
    const std::deque<size_t>& history, size_t& stride, double& confidence) {
    
    if (history.size() < 3) return false;
    
    std::vector<int64_t> deltas;
    for (size_t i = 1; i < history.size(); ++i) {
        deltas.push_back(static_cast<int64_t>(history[i]) - 
                        static_cast<int64_t>(history[i-1]));
    }
    
    // Check if all deltas are the same
    int64_t first_delta = deltas[0];
    size_t matches = 0;
    
    for (const auto& delta : deltas) {
        if (delta == first_delta) matches++;
    }
    
    confidence = static_cast<double>(matches) / deltas.size();
    
    if (confidence > 0.8 && first_delta > 0) {
        stride = static_cast<size_t>(first_delta);
        return true;
    }
    
    return false;
}

bool MetalMemoryPrefetcher::DetectStreamingPattern(
    const std::deque<size_t>& history) {
    
    if (history.size() < 4) return false;
    
    // Check if accesses are monotonically increasing
    for (size_t i = 1; i < history.size(); ++i) {
        if (history[i] <= history[i-1]) {
            return false;
        }
    }
    
    // Check if we're not revisiting old offsets
    std::unordered_set<size_t> unique_offsets(history.begin(), history.end());
    return unique_offsets.size() == history.size();
}

void MetalMemoryPrefetcher::PrefetchWorker() {
    while (running_) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        queue_cv_.wait(lock, [this] {
            return !prefetch_queue_.empty() || !running_;
        });
        
        if (!running_) break;
        
        // Process prefetch requests
        while (!prefetch_queue_.empty()) {
            auto request = prefetch_queue_.front();
            prefetch_queue_.pop_front();
            lock.unlock();
            
            ExecutePrefetch(request);
            
            lock.lock();
        }
        
        // Periodic adaptation
        auto now = std::chrono::steady_clock::now();
        if (adaptive_enabled_ && 
            std::chrono::duration_cast<std::chrono::seconds>(
                now - last_adaptation_).count() > 1) {
            
            lock.unlock();
            AdaptivePrefetchAdjustment();
            last_adaptation_ = now;
            lock.lock();
        }
    }
}

void MetalMemoryPrefetcher::ExecutePrefetch(const PrefetchRequest& request) {
#ifdef __APPLE__
    // Check if this is a Metal buffer
    if (device_ && device_->HasUnifiedMemory()) {
        MetalPrefetch(request.address, request.size);
    } else {
        // Standard prefetch
        SetMemoryAdvise(request.address, request.size, MADV_WILLNEED);
    }
#endif
    
    // Update statistics
    std::lock_guard<std::mutex> lock(stats_mutex_);
    // In a real implementation, we would track prefetch effectiveness
}

void MetalMemoryPrefetcher::AdaptivePrefetchAdjustment() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    // Calculate hit rate
    double hit_rate = (global_stats_.total_accesses > 0) ?
        static_cast<double>(global_stats_.cache_hits) / global_stats_.total_accesses : 0.0;
    
    // Adjust prefetch distance based on hit rate
    if (hit_rate < hit_rate_threshold_ * 0.8) {
        // Increase prefetch distance
        prefetch_distance_ = std::min<size_t>(prefetch_distance_ * 2, 1024 * 1024);
    } else if (hit_rate > hit_rate_threshold_ * 1.2) {
        // Decrease prefetch distance to save bandwidth
        prefetch_distance_ = std::max<size_t>(prefetch_distance_ / 2, 4096);
    }
}

MemoryRegion* MetalMemoryPrefetcher::FindRegion(void* address) {
    std::lock_guard<std::mutex> lock(regions_mutex_);
    
    for (auto& [base, region] : regions_) {
        char* base_ptr = static_cast<char*>(base);
        char* addr_ptr = static_cast<char*>(address);
        
        if (addr_ptr >= base_ptr && addr_ptr < base_ptr + region.size) {
            return &region;
        }
    }
    
    return nullptr;
}

const MemoryRegion* MetalMemoryPrefetcher::FindRegion(void* address) const {
    std::lock_guard<std::mutex> lock(regions_mutex_);
    
    for (const auto& [base, region] : regions_) {
        const char* base_ptr = static_cast<const char*>(base);
        const char* addr_ptr = static_cast<const char*>(address);
        
        if (addr_ptr >= base_ptr && addr_ptr < base_ptr + region.size) {
            return &region;
        }
    }
    
    return nullptr;
}

// PrefetcherManager implementation
PrefetcherManager& PrefetcherManager::Instance() {
    static PrefetcherManager instance;
    return instance;
}

// Prefetch pattern implementations
namespace PrefetchPatterns {

void PrefetchMatrixMultiply(void* a, void* b, void* c,
                           size_t m, size_t n, size_t k,
                           size_t element_size) {
    auto& prefetcher = PrefetcherManager::Instance().GetPrefetcher();
    
    // Prefetch strategy for matrix multiply:
    // - A is accessed row-wise
    // - B is accessed column-wise (may benefit from transposition)
    // - C is written row-wise
    
    // Mark access patterns
    prefetcher->MarkAsStreamingData(a, m * k * element_size);
    prefetcher->MarkAsTemporalData(b, k * n * element_size);
    prefetcher->MarkAsStreamingData(c, m * n * element_size);
    
    // Prefetch first blocks
    size_t block_size = 64 * 1024; // 64KB blocks
    prefetcher->Prefetch(a, std::min(block_size, m * k * element_size));
    prefetcher->Prefetch(b, std::min(block_size, k * n * element_size));
}

void PrefetchConvolution(void* input, void* kernel, void* output,
                        size_t batch, size_t height, size_t width,
                        size_t channels, size_t filters) {
    auto& prefetcher = PrefetcherManager::Instance().GetPrefetcher();
    
    // Convolution access patterns:
    // - Input: spatial locality within each channel
    // - Kernel: fully reused for each spatial position
    // - Output: sequential write
    
    size_t input_size = batch * height * width * channels * sizeof(float);
    size_t kernel_size = filters * channels * 3 * 3 * sizeof(float); // Assuming 3x3
    size_t output_size = batch * height * width * filters * sizeof(float);
    
    prefetcher->MarkAsTemporalData(kernel, kernel_size);
    prefetcher->MarkAsStreamingData(output, output_size);
    
    // Prefetch kernel entirely (it's reused)
    prefetcher->Prefetch(kernel, kernel_size, PrefetchHint::AGGRESSIVE);
}

} // namespace PrefetchPatterns

} // namespace metal
} // namespace triton