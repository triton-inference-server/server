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

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <dispatch/dispatch.h>
#include <CoreFoundation/CoreFoundation.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dlfcn.h>
#include <pthread.h>
#endif

#include <string>
#include <vector>
#include <thread>

namespace triton { namespace backend { namespace python {

#ifdef TRITON_PLATFORM_MACOS

class MacOSCompatibilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Get system information
        size_t len = sizeof(cpu_brand_);
        sysctlbyname("machdep.cpu.brand_string", cpu_brand_, &len, NULL, 0);
        
        len = sizeof(int);
        sysctlbyname("hw.ncpu", &num_cpus_, &len, NULL, 0);
        
        len = sizeof(int64_t);
        sysctlbyname("hw.memsize", &total_memory_, &len, NULL, 0);
    }
    
    char cpu_brand_[256];
    int num_cpus_;
    int64_t total_memory_;
};

// Test macOS system information
TEST_F(MacOSCompatibilityTest, SystemInformation) {
    std::cout << "CPU: " << cpu_brand_ << std::endl;
    std::cout << "CPU Cores: " << num_cpus_ << std::endl;
    std::cout << "Total Memory: " << (total_memory_ / (1024 * 1024 * 1024)) << " GB" << std::endl;
    
    // Verify we got valid information
    EXPECT_GT(num_cpus_, 0);
    EXPECT_GT(total_memory_, 0);
    
    // Check architecture
    struct utsname unameData;
    ASSERT_EQ(uname(&unameData), 0);
    
    std::string machine = unameData.machine;
    EXPECT_TRUE(machine == "x86_64" || machine == "arm64") 
        << "Unexpected architecture: " << machine;
    
    if (machine == "arm64") {
        std::cout << "Running on Apple Silicon (ARM64)" << std::endl;
    } else {
        std::cout << "Running on Intel (x86_64)" << std::endl;
    }
}

// Test Mach absolute time
TEST_F(MacOSCompatibilityTest, MachTimeAPI) {
    // Get time base info
    mach_timebase_info_data_t timebase;
    ASSERT_EQ(mach_timebase_info(&timebase), KERN_SUCCESS);
    
    // Measure time
    uint64_t start = mach_absolute_time();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    uint64_t end = mach_absolute_time();
    
    // Convert to nanoseconds
    uint64_t elapsed_ns = (end - start) * timebase.numer / timebase.denom;
    
    // Convert to milliseconds
    double elapsed_ms = elapsed_ns / 1000000.0;
    
    // Should be approximately 100ms (allow for some variance)
    EXPECT_GT(elapsed_ms, 90.0);
    EXPECT_LT(elapsed_ms, 150.0);
}

// Test Grand Central Dispatch (GCD)
TEST_F(MacOSCompatibilityTest, GrandCentralDispatch) {
    dispatch_queue_t queue = dispatch_queue_create("com.triton.test.serial", NULL);
    dispatch_group_t group = dispatch_group_create();
    
    __block int counter = 0;
    const int num_tasks = 100;
    
    // Submit tasks
    for (int i = 0; i < num_tasks; ++i) {
        dispatch_group_async(group, queue, ^{
            // Atomic increment using OSAtomic
            __sync_fetch_and_add(&counter, 1);
        });
    }
    
    // Wait for completion
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    
    EXPECT_EQ(counter, num_tasks);
    
    // Test concurrent queue
    dispatch_queue_t concurrent_queue = dispatch_queue_create("com.triton.test.concurrent",
                                                              DISPATCH_QUEUE_CONCURRENT);
    
    __block std::atomic<int> atomic_counter(0);
    
    dispatch_apply(num_tasks, concurrent_queue, ^(size_t index) {
        atomic_counter.fetch_add(1);
    });
    
    EXPECT_EQ(atomic_counter.load(), num_tasks);
    
    // Cleanup
    dispatch_release(group);
    dispatch_release(queue);
    dispatch_release(concurrent_queue);
}

// Test library validation and code signing
TEST_F(MacOSCompatibilityTest, LibraryValidation) {
    // Check if library validation is enabled
    int restricted = 0;
    size_t size = sizeof(restricted);
    
    if (sysctlbyname("kern.secure_kernel", &restricted, &size, NULL, 0) == 0) {
        if (restricted) {
            std::cout << "System Integrity Protection (SIP) is enabled" << std::endl;
        } else {
            std::cout << "System Integrity Protection (SIP) is disabled" << std::endl;
        }
    }
    
    // Test loading a system library
    void* handle = dlopen("/usr/lib/libSystem.B.dylib", RTLD_LAZY);
    EXPECT_NE(handle, nullptr) << "Failed to load system library: " << dlerror();
    
    if (handle) {
        dlclose(handle);
    }
}

// Test file system features
TEST_F(MacOSCompatibilityTest, FileSystemFeatures) {
    // Test case-insensitive file system (common on macOS)
    const char* test_dir = "/tmp/triton_macos_test";
    mkdir(test_dir, 0755);
    
    // Create a file
    std::string filename = std::string(test_dir) + "/TestFile.txt";
    FILE* fp = fopen(filename.c_str(), "w");
    ASSERT_NE(fp, nullptr);
    fprintf(fp, "test content");
    fclose(fp);
    
    // Try to access with different case
    std::string filename_lower = std::string(test_dir) + "/testfile.txt";
    struct stat st;
    
    // On case-insensitive file system, this should succeed
    int result = stat(filename_lower.c_str(), &st);
    bool case_insensitive = (result == 0);
    
    if (case_insensitive) {
        std::cout << "File system is case-insensitive" << std::endl;
    } else {
        std::cout << "File system is case-sensitive" << std::endl;
    }
    
    // Cleanup
    unlink(filename.c_str());
    rmdir(test_dir);
}

// Test pthread features specific to macOS
TEST_F(MacOSCompatibilityTest, PthreadFeatures) {
    pthread_t thread;
    pthread_attr_t attr;
    
    ASSERT_EQ(pthread_attr_init(&attr), 0);
    
    // Set thread name (macOS supports this)
    struct ThreadData {
        std::string name;
        bool name_set;
    } data = {"TritonTestThread", false};
    
    auto thread_func = [](void* arg) -> void* {
        ThreadData* data = static_cast<ThreadData*>(arg);
        
        // Set thread name
        pthread_setname_np(data->name.c_str());
        
        // Verify name was set
        char thread_name[256];
        if (pthread_getname_np(pthread_self(), thread_name, sizeof(thread_name)) == 0) {
            data->name_set = (strcmp(thread_name, data->name.c_str()) == 0);
        }
        
        return nullptr;
    };
    
    ASSERT_EQ(pthread_create(&thread, &attr, thread_func, &data), 0);
    ASSERT_EQ(pthread_join(thread, nullptr), 0);
    
    EXPECT_TRUE(data.name_set) << "Failed to set thread name";
    
    pthread_attr_destroy(&attr);
}

// Test process priority and scheduling
TEST_F(MacOSCompatibilityTest, ProcessScheduling) {
    // Get current process priority
    int priority = getpriority(PRIO_PROCESS, 0);
    EXPECT_GE(priority, -20);
    EXPECT_LE(priority, 20);
    
    // Test thread QoS (Quality of Service) classes
    pthread_t thread = pthread_self();
    qos_class_t qos_class;
    int relative_priority;
    
    if (pthread_get_qos_class_np(thread, &qos_class, &relative_priority) == 0) {
        const char* qos_name = nullptr;
        switch (qos_class) {
            case QOS_CLASS_USER_INTERACTIVE:
                qos_name = "USER_INTERACTIVE";
                break;
            case QOS_CLASS_USER_INITIATED:
                qos_name = "USER_INITIATED";
                break;
            case QOS_CLASS_DEFAULT:
                qos_name = "DEFAULT";
                break;
            case QOS_CLASS_UTILITY:
                qos_name = "UTILITY";
                break;
            case QOS_CLASS_BACKGROUND:
                qos_name = "BACKGROUND";
                break;
            case QOS_CLASS_UNSPECIFIED:
                qos_name = "UNSPECIFIED";
                break;
        }
        
        if (qos_name) {
            std::cout << "Thread QoS class: " << qos_name << std::endl;
        }
    }
}

// Test memory limits and resource constraints
TEST_F(MacOSCompatibilityTest, ResourceLimits) {
    struct rlimit rlim;
    
    // Check memory limits
    if (getrlimit(RLIMIT_AS, &rlim) == 0) {
        std::cout << "Virtual memory limit: ";
        if (rlim.rlim_cur == RLIM_INFINITY) {
            std::cout << "unlimited" << std::endl;
        } else {
            std::cout << rlim.rlim_cur / (1024 * 1024) << " MB" << std::endl;
        }
    }
    
    // Check file descriptor limits
    if (getrlimit(RLIMIT_NOFILE, &rlim) == 0) {
        std::cout << "File descriptor limit: " << rlim.rlim_cur << std::endl;
        EXPECT_GT(rlim.rlim_cur, 0);
    }
    
    // Check process limits
    if (getrlimit(RLIMIT_NPROC, &rlim) == 0) {
        std::cout << "Process limit: " << rlim.rlim_cur << std::endl;
        EXPECT_GT(rlim.rlim_cur, 0);
    }
}

// Test macOS vs Linux behavior differences
TEST_F(MacOSCompatibilityTest, PlatformDifferences) {
    // Test environment variable differences
    
    // DYLD_LIBRARY_PATH vs LD_LIBRARY_PATH
    const char* lib_path = getenv("DYLD_LIBRARY_PATH");
    std::cout << "DYLD_LIBRARY_PATH: " << (lib_path ? lib_path : "not set") << std::endl;
    
    // Test temp directory
    const char* tmpdir = getenv("TMPDIR");
    EXPECT_NE(tmpdir, nullptr) << "TMPDIR not set on macOS";
    if (tmpdir) {
        std::cout << "Temp directory: " << tmpdir << std::endl;
        
        // macOS typically uses /var/folders/... for temp
        EXPECT_TRUE(strstr(tmpdir, "/var/folders/") != nullptr ||
                   strstr(tmpdir, "/tmp") != nullptr);
    }
}

#else // !TRITON_PLATFORM_MACOS

class MacOSCompatibilityTest : public ::testing::Test {};

TEST_F(MacOSCompatibilityTest, SkipOnNonMacOS) {
    GTEST_SKIP() << "macOS compatibility tests only run on macOS";
}

#endif // TRITON_PLATFORM_MACOS

}}} // namespace triton::backend::python