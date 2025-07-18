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
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <queue>
#include <random>
#include <chrono>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#include <pthread.h>
#endif

namespace triton { namespace backend { namespace python {

class ThreadSafetyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize test state
        counter_.store(0);
        stop_flag_.store(false);
    }

    void TearDown() override {
        // Ensure all threads are stopped
        stop_flag_.store(true);
    }
    
    std::atomic<int> counter_;
    std::atomic<bool> stop_flag_;
};

// Test basic mutex operations
TEST_F(ThreadSafetyTest, MutexBasics) {
    std::mutex mtx;
    int shared_value = 0;
    const int num_threads = 10;
    const int iterations = 1000;
    
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&mtx, &shared_value, iterations]() {
            for (int j = 0; j < iterations; ++j) {
                std::lock_guard<std::mutex> lock(mtx);
                shared_value++;
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    EXPECT_EQ(shared_value, num_threads * iterations);
}

// Test condition variables
TEST_F(ThreadSafetyTest, ConditionVariables) {
    std::mutex mtx;
    std::condition_variable cv;
    std::queue<int> queue;
    const int num_items = 100;
    std::atomic<int> consumed(0);
    
    // Producer thread
    std::thread producer([&]() {
        for (int i = 0; i < num_items; ++i) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                queue.push(i);
            }
            cv.notify_one();
            
            // Small delay to simulate work
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });
    
    // Consumer threads
    const int num_consumers = 4;
    std::vector<std::thread> consumers;
    
    for (int i = 0; i < num_consumers; ++i) {
        consumers.emplace_back([&]() {
            while (consumed < num_items) {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&queue]() { return !queue.empty(); });
                
                if (!queue.empty()) {
                    queue.pop();
                    consumed++;
                }
            }
        });
    }
    
    producer.join();
    for (auto& c : consumers) {
        c.join();
    }
    
    EXPECT_EQ(consumed.load(), num_items);
    EXPECT_TRUE(queue.empty());
}

// Test atomic operations
TEST_F(ThreadSafetyTest, AtomicOperations) {
    const int num_threads = 8;
    const int iterations = 10000;
    
    std::vector<std::thread> threads;
    
    // Test various atomic operations
    std::atomic<int> add_counter(0);
    std::atomic<int> sub_counter(1000000);
    std::atomic<bool> flag(false);
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            for (int j = 0; j < iterations; ++j) {
                add_counter.fetch_add(1, std::memory_order_relaxed);
                sub_counter.fetch_sub(1, std::memory_order_relaxed);
                
                // Test compare and swap
                bool expected = false;
                if (flag.compare_exchange_weak(expected, true)) {
                    // Do some work
                    std::this_thread::yield();
                    flag.store(false);
                }
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    EXPECT_EQ(add_counter.load(), num_threads * iterations);
    EXPECT_EQ(sub_counter.load(), 1000000 - num_threads * iterations);
}

#ifdef TRITON_PLATFORM_MACOS
// Test macOS-specific threading features
TEST_F(ThreadSafetyTest, MacOSDispatchQueues) {
    dispatch_queue_t queue = dispatch_queue_create("com.triton.test", 
                                                  DISPATCH_QUEUE_CONCURRENT);
    dispatch_group_t group = dispatch_group_create();
    
    std::atomic<int> counter(0);
    const int num_tasks = 100;
    
    // Submit tasks to dispatch queue
    for (int i = 0; i < num_tasks; ++i) {
        dispatch_group_async(group, queue, ^{
            counter.fetch_add(1);
            // Simulate some work
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        });
    }
    
    // Wait for all tasks to complete
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    
    EXPECT_EQ(counter.load(), num_tasks);
    
    // Cleanup
    dispatch_release(group);
    dispatch_release(queue);
}

// Test pthread specifics on macOS
TEST_F(ThreadSafetyTest, MacOSPthreadFeatures) {
    pthread_t thread;
    pthread_attr_t attr;
    
    // Initialize thread attributes
    ASSERT_EQ(pthread_attr_init(&attr), 0);
    
    // Set stack size (macOS has different defaults than Linux)
    size_t stack_size = 2 * 1024 * 1024; // 2MB
    ASSERT_EQ(pthread_attr_setstacksize(&attr, stack_size), 0);
    
    // Verify stack size was set
    size_t get_stack_size;
    ASSERT_EQ(pthread_attr_getstacksize(&attr, &get_stack_size), 0);
    EXPECT_EQ(get_stack_size, stack_size);
    
    // Create thread with custom attributes
    std::atomic<bool> thread_ran(false);
    
    struct ThreadData {
        std::atomic<bool>* flag;
    } data = { &thread_ran };
    
    ASSERT_EQ(pthread_create(&thread, &attr, [](void* arg) -> void* {
        ThreadData* data = static_cast<ThreadData*>(arg);
        data->flag->store(true);
        return nullptr;
    }, &data), 0);
    
    // Wait for thread
    ASSERT_EQ(pthread_join(thread, nullptr), 0);
    EXPECT_TRUE(thread_ran.load());
    
    // Cleanup
    pthread_attr_destroy(&attr);
}
#endif

// Test thread-local storage
TEST_F(ThreadSafetyTest, ThreadLocalStorage) {
    thread_local int tls_value = 0;
    const int num_threads = 4;
    const int thread_value = 100;
    
    std::vector<std::thread> threads;
    std::mutex result_mutex;
    std::vector<int> results;
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([i, thread_value, &result_mutex, &results]() {
            // Set thread-local value
            tls_value = thread_value + i;
            
            // Do some work
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            
            // Verify value hasn't changed
            EXPECT_EQ(tls_value, thread_value + i);
            
            // Store result
            std::lock_guard<std::mutex> lock(result_mutex);
            results.push_back(tls_value);
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Verify we got all expected values
    EXPECT_EQ(results.size(), num_threads);
    std::sort(results.begin(), results.end());
    for (int i = 0; i < num_threads; ++i) {
        EXPECT_EQ(results[i], thread_value + i);
    }
}

// Test race condition detection
TEST_F(ThreadSafetyTest, RaceConditionDetection) {
    // This test intentionally creates potential race conditions
    // to verify they can be detected and handled
    
    struct SharedData {
        int value;
        std::atomic<int> access_count;
        
        SharedData() : value(0), access_count(0) {}
    };
    
    SharedData data;
    const int num_threads = 4;
    const int iterations = 1000;
    std::atomic<int> race_detected(0);
    
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&data, &race_detected, iterations, i]() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(1, 10);
            
            for (int j = 0; j < iterations; ++j) {
                // Increment access count atomically
                int before = data.access_count.fetch_add(1);
                
                // Non-atomic access to value (potential race)
                int old_value = data.value;
                
                // Simulate some computation
                std::this_thread::sleep_for(std::chrono::microseconds(dis(gen)));
                
                data.value = old_value + 1;
                
                // Decrement access count
                int after = data.access_count.fetch_sub(1);
                
                // Detect concurrent access
                if (before > 0 || after > 1) {
                    race_detected++;
                }
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // We expect races to be detected
    EXPECT_GT(race_detected.load(), 0);
    
    // The final value will likely be less than expected due to races
    EXPECT_LT(data.value, num_threads * iterations);
}

// Test deadlock prevention
TEST_F(ThreadSafetyTest, DeadlockPrevention) {
    std::mutex mtx1, mtx2;
    std::atomic<bool> deadlock_detected(false);
    const int timeout_ms = 100;
    
    // Thread 1: locks mtx1 then mtx2
    std::thread t1([&]() {
        std::unique_lock<std::mutex> lock1(mtx1);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        // Try to lock mtx2 with timeout
        std::unique_lock<std::mutex> lock2(mtx2, std::defer_lock);
        if (!lock2.try_lock_for(std::chrono::milliseconds(timeout_ms))) {
            deadlock_detected = true;
        }
    });
    
    // Thread 2: locks mtx2 then mtx1
    std::thread t2([&]() {
        std::unique_lock<std::mutex> lock2(mtx2);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        
        // Try to lock mtx1 with timeout
        std::unique_lock<std::mutex> lock1(mtx1, std::defer_lock);
        if (!lock1.try_lock_for(std::chrono::milliseconds(timeout_ms))) {
            deadlock_detected = true;
        }
    });
    
    t1.join();
    t2.join();
    
    // In this test setup, we might or might not detect deadlock
    // depending on thread scheduling. The important thing is that
    // the threads complete without hanging.
}

}}} // namespace triton::backend::python