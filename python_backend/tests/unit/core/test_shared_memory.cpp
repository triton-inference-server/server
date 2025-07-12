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
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <thread>
#include <chrono>
#include <random>

#include "shm_manager.h"

namespace triton { namespace backend { namespace python {

class SharedMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate unique shared memory names
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1000, 9999);
        
        shm_name_ = "/triton_test_shm_" + std::to_string(dis(gen));
    }

    void TearDown() override {
        // Clean up shared memory
        shm_unlink(shm_name_.c_str());
    }
    
    std::string shm_name_;
};

// Test basic shared memory creation and access
TEST_F(SharedMemoryTest, CreateAndAccess) {
    const size_t shm_size = 4096;
    
    // Create shared memory
    int fd = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
    ASSERT_NE(fd, -1) << "Failed to create shared memory: " << strerror(errno);
    
    // Set size
    ASSERT_EQ(ftruncate(fd, shm_size), 0) << "Failed to set size: " << strerror(errno);
    
    // Map memory
    void* ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ASSERT_NE(ptr, MAP_FAILED) << "Failed to map memory: " << strerror(errno);
    
    // Write test data
    const char* test_data = "Hello from macOS shared memory!";
    strcpy(static_cast<char*>(ptr), test_data);
    
    // Read back data
    EXPECT_STREQ(static_cast<char*>(ptr), test_data);
    
    // Cleanup
    munmap(ptr, shm_size);
    close(fd);
}

// Test concurrent access to shared memory
TEST_F(SharedMemoryTest, ConcurrentAccess) {
    const size_t shm_size = 1024 * 1024; // 1MB
    const int num_threads = 4;
    
    // Create and initialize shared memory
    int fd = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
    ASSERT_NE(fd, -1);
    ASSERT_EQ(ftruncate(fd, shm_size), 0);
    
    void* base_ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ASSERT_NE(base_ptr, MAP_FAILED);
    
    // Initialize memory
    std::memset(base_ptr, 0, shm_size);
    
    std::vector<std::thread> threads;
    std::atomic<int> ready_count(0);
    
    // Launch threads to write to different regions
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([this, i, num_threads, shm_size, &ready_count]() {
            // Open shared memory in thread
            int thread_fd = shm_open(shm_name_.c_str(), O_RDWR, 0666);
            ASSERT_NE(thread_fd, -1);
            
            void* thread_ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, 
                                   MAP_SHARED, thread_fd, 0);
            ASSERT_NE(thread_ptr, MAP_FAILED);
            
            // Signal ready
            ready_count++;
            
            // Wait for all threads
            while (ready_count < num_threads) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            
            // Write to assigned region
            size_t region_size = shm_size / num_threads;
            size_t offset = i * region_size;
            char* region = static_cast<char*>(thread_ptr) + offset;
            
            for (size_t j = 0; j < region_size; ++j) {
                region[j] = static_cast<char>('A' + i);
            }
            
            // Cleanup
            munmap(thread_ptr, shm_size);
            close(thread_fd);
        });
    }
    
    // Wait for threads
    for (auto& t : threads) {
        t.join();
    }
    
    // Verify data
    char* data = static_cast<char*>(base_ptr);
    size_t region_size = shm_size / num_threads;
    
    for (int i = 0; i < num_threads; ++i) {
        for (size_t j = 0; j < region_size; ++j) {
            EXPECT_EQ(data[i * region_size + j], static_cast<char>('A' + i));
        }
    }
    
    // Cleanup
    munmap(base_ptr, shm_size);
    close(fd);
}

// Test shared memory limits
TEST_F(SharedMemoryTest, MemoryLimits) {
#ifdef TRITON_PLATFORM_MACOS
    // macOS has different shared memory limits than Linux
    // Test various sizes
    std::vector<size_t> test_sizes = {
        1024,           // 1KB
        1024 * 1024,    // 1MB
        10 * 1024 * 1024, // 10MB
        100 * 1024 * 1024 // 100MB
    };
    
    for (size_t size : test_sizes) {
        std::string test_shm = shm_name_ + "_" + std::to_string(size);
        
        int fd = shm_open(test_shm.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd == -1) {
            // Hit system limit
            break;
        }
        
        // Try to set size
        if (ftruncate(fd, size) == 0) {
            // Success - try to map
            void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (ptr != MAP_FAILED) {
                // Write pattern to verify
                std::memset(ptr, 0xAB, std::min(size, size_t(4096)));
                munmap(ptr, size);
            }
        }
        
        close(fd);
        shm_unlink(test_shm.c_str());
    }
#else
    GTEST_SKIP() << "macOS specific shared memory limits test";
#endif
}

// Test shared memory persistence
TEST_F(SharedMemoryTest, Persistence) {
    const size_t shm_size = 4096;
    const char* test_data = "Persistent data test";
    
    // Create and write data
    {
        int fd = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
        ASSERT_NE(fd, -1);
        ASSERT_EQ(ftruncate(fd, shm_size), 0);
        
        void* ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ASSERT_NE(ptr, MAP_FAILED);
        
        strcpy(static_cast<char*>(ptr), test_data);
        
        munmap(ptr, shm_size);
        close(fd);
    }
    
    // Read data in new scope
    {
        int fd = shm_open(shm_name_.c_str(), O_RDWR, 0666);
        ASSERT_NE(fd, -1);
        
        void* ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        ASSERT_NE(ptr, MAP_FAILED);
        
        EXPECT_STREQ(static_cast<char*>(ptr), test_data);
        
        munmap(ptr, shm_size);
        close(fd);
    }
}

// Test error handling
TEST_F(SharedMemoryTest, ErrorHandling) {
    // Test opening non-existent shared memory
    int fd = shm_open("/nonexistent_shm_test", O_RDWR, 0666);
    EXPECT_EQ(fd, -1);
    EXPECT_EQ(errno, ENOENT);
    
    // Test invalid size
    fd = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
    ASSERT_NE(fd, -1);
    
    // Try to set negative size
    EXPECT_EQ(ftruncate(fd, -1), -1);
    
    close(fd);
}

// Test memory synchronization
TEST_F(SharedMemoryTest, MemorySynchronization) {
    const size_t shm_size = 4096;
    
    int fd = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
    ASSERT_NE(fd, -1);
    ASSERT_EQ(ftruncate(fd, shm_size), 0);
    
    void* ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ASSERT_NE(ptr, MAP_FAILED);
    
    // Write data
    volatile int* shared_counter = static_cast<volatile int*>(ptr);
    *shared_counter = 0;
    
    // Sync memory
    EXPECT_EQ(msync(ptr, sizeof(int), MS_SYNC), 0);
    
    // Launch writer thread
    std::thread writer([this, shm_size]() {
        int thread_fd = shm_open(shm_name_.c_str(), O_RDWR, 0666);
        ASSERT_NE(thread_fd, -1);
        
        void* thread_ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, 
                               MAP_SHARED, thread_fd, 0);
        ASSERT_NE(thread_ptr, MAP_FAILED);
        
        volatile int* counter = static_cast<volatile int*>(thread_ptr);
        
        for (int i = 0; i < 1000; ++i) {
            (*counter)++;
            msync(thread_ptr, sizeof(int), MS_SYNC);
        }
        
        munmap(thread_ptr, shm_size);
        close(thread_fd);
    });
    
    writer.join();
    
    // Verify final value
    EXPECT_EQ(*shared_counter, 1000);
    
    munmap(ptr, shm_size);
    close(fd);
}

}}} // namespace triton::backend::python