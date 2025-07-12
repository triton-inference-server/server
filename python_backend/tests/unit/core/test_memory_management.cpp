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

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <random>

#ifdef __APPLE__
#include <mach/mach.h>
#include <malloc/malloc.h>
#endif

namespace triton { namespace backend { namespace python {

class MemoryManagementTest : public ::testing::Test {
protected:
    void SetUp() override {
        initial_memory_ = GetCurrentMemoryUsage();
    }

    void TearDown() override {
        // Force garbage collection if needed
        size_t final_memory = GetCurrentMemoryUsage();
        
        // Log memory usage for debugging
        if (::testing::Test::HasFailure()) {
            std::cout << "Memory usage - Initial: " << initial_memory_ 
                     << ", Final: " << final_memory 
                     << ", Difference: " << (int64_t)(final_memory - initial_memory_) 
                     << " bytes" << std::endl;
        }
    }
    
    size_t GetCurrentMemoryUsage() {
#ifdef __APPLE__
        struct mach_task_basic_info info;
        mach_msg_type_number_t size = MACH_TASK_BASIC_INFO_COUNT;
        kern_return_t kerr = task_info(mach_task_self(),
                                       MACH_TASK_BASIC_INFO,
                                       (task_info_t)&info,
                                       &size);
        
        if (kerr == KERN_SUCCESS) {
            return info.resident_size;
        }
#endif
        return 0;
    }
    
    size_t initial_memory_;
};

// Test basic memory allocation and deallocation
TEST_F(MemoryManagementTest, BasicAllocation) {
    const size_t alloc_size = 1024 * 1024; // 1MB
    
    // Test malloc/free
    {
        void* ptr = malloc(alloc_size);
        ASSERT_NE(ptr, nullptr);
        
        // Initialize memory to ensure it's actually allocated
        memset(ptr, 0xAB, alloc_size);
        
        // Verify memory is accessible
        EXPECT_EQ(static_cast<uint8_t*>(ptr)[0], 0xAB);
        EXPECT_EQ(static_cast<uint8_t*>(ptr)[alloc_size - 1], 0xAB);
        
        free(ptr);
    }
    
    // Test new/delete
    {
        char* ptr = new char[alloc_size];
        ASSERT_NE(ptr, nullptr);
        
        memset(ptr, 0xCD, alloc_size);
        EXPECT_EQ(ptr[0], static_cast<char>(0xCD));
        
        delete[] ptr;
    }
    
    // Test aligned allocation
    {
        const size_t alignment = 64;
        void* ptr = nullptr;
        
#ifdef __APPLE__
        // macOS uses posix_memalign
        int result = posix_memalign(&ptr, alignment, alloc_size);
        ASSERT_EQ(result, 0);
        ASSERT_NE(ptr, nullptr);
#else
        ptr = aligned_alloc(alignment, alloc_size);
        ASSERT_NE(ptr, nullptr);
#endif
        
        // Verify alignment
        EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % alignment, 0);
        
        free(ptr);
    }
}

// Test memory pool behavior
TEST_F(MemoryManagementTest, MemoryPooling) {
    const size_t small_size = 64;
    const size_t medium_size = 1024;
    const size_t large_size = 1024 * 1024;
    const int iterations = 100;
    
    // Allocate and free many small objects
    std::vector<void*> ptrs;
    
    for (int i = 0; i < iterations; ++i) {
        void* ptr = malloc(small_size);
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }
    
    // Free in LIFO order (common pattern)
    while (!ptrs.empty()) {
        free(ptrs.back());
        ptrs.pop_back();
    }
    
    // Allocate again - should reuse memory
    for (int i = 0; i < iterations; ++i) {
        void* ptr = malloc(small_size);
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }
    
    // Free in random order
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(ptrs.begin(), ptrs.end(), gen);
    
    for (void* ptr : ptrs) {
        free(ptr);
    }
    ptrs.clear();
}

#ifdef TRITON_PLATFORM_MACOS
// Test macOS-specific memory features
TEST_F(MemoryManagementTest, MacOSMemoryZones) {
    // Get default malloc zone
    malloc_zone_t* zone = malloc_default_zone();
    ASSERT_NE(zone, nullptr);
    
    // Allocate using zone
    const size_t size = 1024;
    void* ptr = malloc_zone_malloc(zone, size);
    ASSERT_NE(ptr, nullptr);
    
    // Get allocation size
    size_t actual_size = malloc_size(ptr);
    EXPECT_GE(actual_size, size);
    
    // Check if pointer belongs to zone
    EXPECT_NE(malloc_zone_from_ptr(ptr), nullptr);
    
    // Free using zone
    malloc_zone_free(zone, ptr);
    
    // Test zone statistics
    malloc_statistics_t stats;
    malloc_zone_statistics(zone, &stats);
    
    // Verify statistics are reasonable
    EXPECT_GT(stats.blocks_in_use, 0u);
    EXPECT_GT(stats.size_in_use, 0u);
}

// Test memory pressure handling
TEST_F(MemoryManagementTest, MacOSMemoryPressure) {
    // This test simulates memory pressure scenarios
    std::vector<std::unique_ptr<char[]>> allocations;
    const size_t chunk_size = 10 * 1024 * 1024; // 10MB chunks
    const int max_chunks = 10;
    
    // Allocate memory until we hit a limit or system pressure
    for (int i = 0; i < max_chunks; ++i) {
        try {
            auto chunk = std::make_unique<char[]>(chunk_size);
            // Touch memory to ensure it's allocated
            memset(chunk.get(), i, chunk_size);
            allocations.push_back(std::move(chunk));
        } catch (const std::bad_alloc&) {
            // Expected when memory is exhausted
            break;
        }
        
        // Check current memory usage
        size_t current = GetCurrentMemoryUsage();
        if (current > initial_memory_ + (100 * 1024 * 1024)) {
            // Stop if we've allocated more than 100MB
            break;
        }
    }
    
    // Verify we allocated some memory
    EXPECT_GT(allocations.size(), 0u);
    
    // Clear allocations
    allocations.clear();
}
#endif

// Test memory leak detection patterns
TEST_F(MemoryManagementTest, LeakDetection) {
    // Pattern 1: Forgetting to free memory (intentional for test)
    std::vector<void*> leaked_ptrs;
    const size_t leak_size = 1024;
    const int num_leaks = 5;
    
    // Track allocated memory
    size_t before_leaks = GetCurrentMemoryUsage();
    
    // Create intentional leaks
    for (int i = 0; i < num_leaks; ++i) {
        void* ptr = malloc(leak_size);
        ASSERT_NE(ptr, nullptr);
        leaked_ptrs.push_back(ptr);
    }
    
    size_t after_leaks = GetCurrentMemoryUsage();
    
    // Clean up the "leaks" for test hygiene
    for (void* ptr : leaked_ptrs) {
        free(ptr);
    }
    
    // Memory usage should have increased during leak
    EXPECT_GT(after_leaks, before_leaks);
}

// Test custom allocators
TEST_F(MemoryManagementTest, CustomAllocator) {
    // Simple custom allocator that tracks allocations
    class TrackingAllocator {
    public:
        struct AllocationInfo {
            size_t size;
            void* ptr;
        };
        
        void* allocate(size_t size) {
            void* ptr = malloc(size);
            if (ptr) {
                allocations_.push_back({size, ptr});
                total_allocated_ += size;
            }
            return ptr;
        }
        
        void deallocate(void* ptr) {
            auto it = std::find_if(allocations_.begin(), allocations_.end(),
                                  [ptr](const AllocationInfo& info) {
                                      return info.ptr == ptr;
                                  });
            
            if (it != allocations_.end()) {
                total_allocated_ -= it->size;
                allocations_.erase(it);
                free(ptr);
            }
        }
        
        size_t getTotalAllocated() const { return total_allocated_; }
        size_t getAllocationCount() const { return allocations_.size(); }
        
    private:
        std::vector<AllocationInfo> allocations_;
        size_t total_allocated_ = 0;
    };
    
    TrackingAllocator allocator;
    
    // Test allocations
    void* p1 = allocator.allocate(100);
    void* p2 = allocator.allocate(200);
    void* p3 = allocator.allocate(300);
    
    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p2, nullptr);
    ASSERT_NE(p3, nullptr);
    
    EXPECT_EQ(allocator.getTotalAllocated(), 600u);
    EXPECT_EQ(allocator.getAllocationCount(), 3u);
    
    // Test deallocations
    allocator.deallocate(p2);
    EXPECT_EQ(allocator.getTotalAllocated(), 400u);
    EXPECT_EQ(allocator.getAllocationCount(), 2u);
    
    allocator.deallocate(p1);
    allocator.deallocate(p3);
    EXPECT_EQ(allocator.getTotalAllocated(), 0u);
    EXPECT_EQ(allocator.getAllocationCount(), 0u);
}

// Test memory fragmentation
TEST_F(MemoryManagementTest, FragmentationTest) {
    const int num_iterations = 100;
    std::vector<void*> allocations;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> size_dist(64, 4096);
    
    // Create fragmentation pattern
    for (int i = 0; i < num_iterations; ++i) {
        // Allocate random sized blocks
        size_t size = size_dist(gen);
        void* ptr = malloc(size);
        ASSERT_NE(ptr, nullptr);
        allocations.push_back(ptr);
    }
    
    // Free every other allocation
    for (size_t i = 0; i < allocations.size(); i += 2) {
        free(allocations[i]);
        allocations[i] = nullptr;
    }
    
    // Try to allocate large block (may fail due to fragmentation)
    void* large_ptr = malloc(1024 * 1024);
    if (large_ptr) {
        free(large_ptr);
    }
    
    // Clean up remaining allocations
    for (void* ptr : allocations) {
        if (ptr) {
            free(ptr);
        }
    }
}

// Test thread-safe memory allocation
TEST_F(MemoryManagementTest, ThreadSafeAllocation) {
    const int num_threads = 4;
    const int allocations_per_thread = 1000;
    std::atomic<int> successful_allocs(0);
    
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&successful_allocs, allocations_per_thread]() {
            std::vector<void*> thread_allocs;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> size_dist(64, 1024);
            
            for (int j = 0; j < allocations_per_thread; ++j) {
                size_t size = size_dist(gen);
                void* ptr = malloc(size);
                
                if (ptr) {
                    // Write pattern to memory
                    memset(ptr, j % 256, size);
                    thread_allocs.push_back(ptr);
                    successful_allocs++;
                }
            }
            
            // Free all allocations
            for (void* ptr : thread_allocs) {
                free(ptr);
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    EXPECT_EQ(successful_allocs.load(), num_threads * allocations_per_thread);
}

}}} // namespace triton::backend::python