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
#include <dlfcn.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <filesystem>

namespace triton { namespace backend { namespace python {

class DynamicLoadingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up library paths for macOS
#ifdef __APPLE__
        // Save original DYLD paths
        const char* dyld_library_path = getenv("DYLD_LIBRARY_PATH");
        if (dyld_library_path) {
            original_dyld_library_path_ = dyld_library_path;
        }
        
        const char* dyld_fallback_path = getenv("DYLD_FALLBACK_LIBRARY_PATH");
        if (dyld_fallback_path) {
            original_dyld_fallback_path_ = dyld_fallback_path;
        }
#endif
    }

    void TearDown() override {
        // Restore original paths
#ifdef __APPLE__
        if (!original_dyld_library_path_.empty()) {
            setenv("DYLD_LIBRARY_PATH", original_dyld_library_path_.c_str(), 1);
        }
        if (!original_dyld_fallback_path_.empty()) {
            setenv("DYLD_FALLBACK_LIBRARY_PATH", original_dyld_fallback_path_.c_str(), 1);
        }
#endif
        
        // Close any open handles
        for (void* handle : handles_) {
            if (handle) {
                dlclose(handle);
            }
        }
        handles_.clear();
    }
    
    std::string original_dyld_library_path_;
    std::string original_dyld_fallback_path_;
    std::vector<void*> handles_;
};

// Test basic dynamic library loading
TEST_F(DynamicLoadingTest, LoadSystemLibrary) {
    // Try to load a system library
#ifdef __APPLE__
    const char* lib_name = "libSystem.dylib";
#else
    const char* lib_name = "libc.so.6";
#endif
    
    void* handle = dlopen(lib_name, RTLD_LAZY);
    if (handle) {
        handles_.push_back(handle);
    }
    
    EXPECT_NE(handle, nullptr) << "Failed to load " << lib_name << ": " << dlerror();
    
    if (handle) {
        // Try to get a symbol
        void* symbol = dlsym(handle, "malloc");
        EXPECT_NE(symbol, nullptr) << "Failed to find malloc: " << dlerror();
    }
}

// Test library path resolution on macOS
TEST_F(DynamicLoadingTest, MacOSLibraryPaths) {
#ifdef TRITON_PLATFORM_MACOS
    // Test DYLD_LIBRARY_PATH
    setenv("DYLD_LIBRARY_PATH", "/usr/local/lib:/opt/homebrew/lib", 1);
    
    // Create a test scenario with @rpath
    std::string test_paths[] = {
        "@rpath/libtest.dylib",
        "@loader_path/libtest.dylib",
        "@executable_path/../lib/libtest.dylib"
    };
    
    for (const auto& path : test_paths) {
        void* handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_NOLOAD);
        // We expect these to fail since the libraries don't exist
        // This test verifies the path resolution doesn't crash
        EXPECT_EQ(handle, nullptr);
    }
    
    // Test fallback paths
    setenv("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib", 1);
    
    // Verify environment was set
    EXPECT_NE(getenv("DYLD_LIBRARY_PATH"), nullptr);
    EXPECT_NE(getenv("DYLD_FALLBACK_LIBRARY_PATH"), nullptr);
#else
    GTEST_SKIP() << "macOS specific library path test";
#endif
}

// Test RTLD flags behavior
TEST_F(DynamicLoadingTest, RTLDFlags) {
    const char* lib_name = nullptr;
#ifdef __APPLE__
    lib_name = "libm.dylib";
#else
    lib_name = "libm.so.6";
#endif
    
    // Test different loading flags
    struct {
        int flags;
        const char* description;
    } test_cases[] = {
        {RTLD_LAZY, "RTLD_LAZY"},
        {RTLD_NOW, "RTLD_NOW"},
        {RTLD_LOCAL, "RTLD_LOCAL"},
        {RTLD_GLOBAL, "RTLD_GLOBAL"},
        {RTLD_LAZY | RTLD_LOCAL, "RTLD_LAZY | RTLD_LOCAL"},
        {RTLD_NOW | RTLD_GLOBAL, "RTLD_NOW | RTLD_GLOBAL"}
    };
    
    for (const auto& test : test_cases) {
        void* handle = dlopen(lib_name, test.flags);
        EXPECT_NE(handle, nullptr) << "Failed with " << test.description 
                                   << ": " << dlerror();
        
        if (handle) {
            // Verify we can get symbols
            void* symbol = dlsym(handle, "sin");
            EXPECT_NE(symbol, nullptr) << "Failed to find sin with " 
                                      << test.description;
            dlclose(handle);
        }
    }
}

// Test error handling
TEST_F(DynamicLoadingTest, ErrorHandling) {
    // Clear any previous errors
    dlerror();
    
    // Try to load non-existent library
    void* handle = dlopen("libnonexistent_test_library.so", RTLD_LAZY);
    EXPECT_EQ(handle, nullptr);
    
    const char* error = dlerror();
    EXPECT_NE(error, nullptr);
    EXPECT_NE(strstr(error, "libnonexistent_test_library"), nullptr);
    
    // Try to get symbol from null handle
    void* symbol = dlsym(nullptr, "test_symbol");
    EXPECT_EQ(symbol, nullptr);
    
    error = dlerror();
    EXPECT_NE(error, nullptr);
}

// Test Python library loading
TEST_F(DynamicLoadingTest, LoadPythonLibrary) {
    // Try to find Python library
    std::vector<std::string> python_libs;
    
#ifdef __APPLE__
    python_libs = {
        "libpython3.dylib",
        "libpython3.10.dylib",
        "libpython3.11.dylib",
        "libpython3.12.dylib",
        "/usr/local/opt/python@3.10/lib/libpython3.10.dylib",
        "/opt/homebrew/opt/python@3.10/lib/libpython3.10.dylib"
    };
#else
    python_libs = {
        "libpython3.so",
        "libpython3.10.so",
        "libpython3.11.so",
        "libpython3.12.so"
    };
#endif
    
    void* python_handle = nullptr;
    std::string loaded_lib;
    
    for (const auto& lib : python_libs) {
        python_handle = dlopen(lib.c_str(), RTLD_LAZY | RTLD_GLOBAL);
        if (python_handle) {
            loaded_lib = lib;
            handles_.push_back(python_handle);
            break;
        }
    }
    
    if (python_handle) {
        // Verify we can get Python symbols
        void* py_init = dlsym(python_handle, "Py_Initialize");
        EXPECT_NE(py_init, nullptr) << "Failed to find Py_Initialize in " << loaded_lib;
        
        void* py_finalize = dlsym(python_handle, "Py_Finalize");
        EXPECT_NE(py_finalize, nullptr) << "Failed to find Py_Finalize in " << loaded_lib;
    } else {
        GTEST_SKIP() << "Python library not found - skipping Python loading test";
    }
}

// Test concurrent library loading
TEST_F(DynamicLoadingTest, ConcurrentLoading) {
    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::atomic<int> success_count(0);
    std::mutex handle_mutex;
    
#ifdef __APPLE__
    const char* lib_name = "libz.dylib";
#else
    const char* lib_name = "libz.so.1";
#endif
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([this, lib_name, &success_count, &handle_mutex]() {
            void* handle = dlopen(lib_name, RTLD_LAZY);
            if (handle) {
                success_count++;
                
                // Get a symbol to verify library is loaded
                void* symbol = dlsym(handle, "zlibVersion");
                if (symbol) {
                    // Call the function to verify it works
                    typedef const char* (*zlibVersion_t)();
                    zlibVersion_t zlibVersion = (zlibVersion_t)symbol;
                    const char* version = zlibVersion();
                    EXPECT_NE(version, nullptr);
                }
                
                std::lock_guard<std::mutex> lock(handle_mutex);
                handles_.push_back(handle);
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    EXPECT_EQ(success_count, num_threads);
}

// Test dladdr functionality
TEST_F(DynamicLoadingTest, DLAddrInfo) {
    // Get info about a known function
    Dl_info info;
    int result = dladdr((void*)dlopen, &info);
    
    EXPECT_NE(result, 0) << "dladdr failed";
    
    if (result != 0) {
        EXPECT_NE(info.dli_fname, nullptr);
        EXPECT_NE(info.dli_fbase, nullptr);
        EXPECT_NE(info.dli_sname, nullptr);
        EXPECT_NE(info.dli_saddr, nullptr);
        
        // Print info for debugging
        std::cout << "Library: " << (info.dli_fname ? info.dli_fname : "unknown") << std::endl;
        std::cout << "Symbol: " << (info.dli_sname ? info.dli_sname : "unknown") << std::endl;
    }
}

}}} // namespace triton::backend::python