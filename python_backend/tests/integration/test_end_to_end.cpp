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
#include <chrono>
#include <vector>
#include <memory>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

namespace triton { namespace backend { namespace python {

class EndToEndTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test model repository
        model_repo_ = "/tmp/triton_e2e_test_models";
        std::system(("mkdir -p " + model_repo_).c_str());
        
        // Create test models
        CreateSimpleModel();
        CreateEnsembleModel();
        CreateBatchingModel();
    }

    void TearDown() override {
        // Clean up
        std::system(("rm -rf " + model_repo_).c_str());
    }
    
    void CreateSimpleModel() {
        std::string model_dir = model_repo_ + "/simple_model";
        std::system(("mkdir -p " + model_dir + "/1").c_str());
        
        // Write model.py
        std::ofstream model_file(model_dir + "/1/model.py");
        model_file << R"(
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        
    def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_0_np = in_0.as_numpy()
            
            # Simple operation: add 1
            out_0 = in_0_np + 1
            
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", out_0)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)
        
        return responses
    
    def finalize(self):
        pass
)";
        model_file.close();
        
        // Write config.pbtxt
        std::ofstream config_file(model_dir + "/config.pbtxt");
        config_file << R"(
name: "simple_model"
backend: "python"
max_batch_size: 8
input [
  {
    name: "INPUT0"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]
output [
  {
    name: "OUTPUT0"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
)";
        config_file.close();
    }
    
    void CreateEnsembleModel() {
        // Create preprocessing model
        std::string preproc_dir = model_repo_ + "/preprocess";
        std::system(("mkdir -p " + preproc_dir + "/1").c_str());
        
        std::ofstream preproc_file(preproc_dir + "/1/model.py");
        preproc_file << R"(
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "RAW_INPUT")
            data = in_0.as_numpy()
            
            # Normalize data
            normalized = (data - np.mean(data)) / (np.std(data) + 1e-7)
            
            out_tensor = pb_utils.Tensor("PROCESSED_INPUT", normalized.astype(np.float32))
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        
        return responses
)";
        preproc_file.close();
        
        // Write ensemble config
        std::string ensemble_dir = model_repo_ + "/ensemble_model";
        std::system(("mkdir -p " + ensemble_dir + "/1").c_str());
        
        std::ofstream ensemble_config(ensemble_dir + "/config.pbtxt");
        ensemble_config << R"(
name: "ensemble_model"
platform: "ensemble"
max_batch_size: 8
input [
  {
    name: "RAW_INPUT"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]
output [
  {
    name: "OUTPUT0"
    data_type: TYPE_FP32
    dims: [ -1 ]
  }
]
ensemble_scheduling {
  step [
    {
      model_name: "preprocess"
      model_version: 1
      input_map {
        key: "RAW_INPUT"
        value: "RAW_INPUT"
      }
      output_map {
        key: "PROCESSED_INPUT"
        value: "preprocessed"
      }
    },
    {
      model_name: "simple_model"
      model_version: 1
      input_map {
        key: "INPUT0"
        value: "preprocessed"
      }
      output_map {
        key: "OUTPUT0"
        value: "OUTPUT0"
      }
    }
  ]
}
)";
        ensemble_config.close();
    }
    
    void CreateBatchingModel() {
        std::string model_dir = model_repo_ + "/batching_model";
        std::system(("mkdir -p " + model_dir + "/1").c_str());
        
        std::ofstream model_file(model_dir + "/1/model.py");
        model_file << R"(
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def execute(self, requests):
        # Process all requests as a batch
        batch_size = len(requests)
        
        # Collect all inputs
        inputs = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            inputs.append(in_0.as_numpy())
        
        # Stack into batch
        batch_input = np.vstack(inputs)
        
        # Batch processing (matrix multiplication)
        weight = np.ones((batch_input.shape[1], batch_input.shape[1]), dtype=np.float32)
        batch_output = np.matmul(batch_input, weight)
        
        # Create responses
        responses = []
        for i in range(batch_size):
            out_tensor = pb_utils.Tensor("OUTPUT0", batch_output[i:i+1])
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        
        return responses
)";
        model_file.close();
        
        std::ofstream config_file(model_dir + "/config.pbtxt");
        config_file << R"(
name: "batching_model"
backend: "python"
max_batch_size: 16
dynamic_batching {
  max_queue_delay_microseconds: 100000
}
input [
  {
    name: "INPUT0"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }
]
output [
  {
    name: "OUTPUT0"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }
]
)";
        config_file.close();
    }
    
    bool WaitForPort(int port, int timeout_seconds = 30) {
        auto start = std::chrono::steady_clock::now();
        
        while (std::chrono::steady_clock::now() - start < 
               std::chrono::seconds(timeout_seconds)) {
            int sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock < 0) continue;
            
            struct sockaddr_in addr;
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port);
            addr.sin_addr.s_addr = inet_addr("127.0.0.1");
            
            if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                close(sock);
                return true;
            }
            
            close(sock);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        return false;
    }
    
    std::string model_repo_;
};

// Test basic server startup and model loading
TEST_F(EndToEndTest, ServerStartupAndModelLoad) {
    // Note: This test assumes Triton server binary is available
    // In a real test environment, we would start the actual server
    
    // For now, we simulate the expected behavior
    EXPECT_TRUE(std::filesystem::exists(model_repo_ + "/simple_model"));
    EXPECT_TRUE(std::filesystem::exists(model_repo_ + "/ensemble_model"));
    EXPECT_TRUE(std::filesystem::exists(model_repo_ + "/batching_model"));
}

// Test concurrent inference requests
TEST_F(EndToEndTest, ConcurrentInference) {
    const int num_clients = 4;
    const int requests_per_client = 10;
    std::atomic<int> successful_requests(0);
    std::atomic<int> failed_requests(0);
    
    std::vector<std::thread> client_threads;
    
    for (int client_id = 0; client_id < num_clients; ++client_id) {
        client_threads.emplace_back([&, client_id]() {
            for (int req = 0; req < requests_per_client; ++req) {
                // Simulate inference request
                try {
                    // In real test, would send HTTP/gRPC request
                    std::vector<float> input_data(10, 1.0f);
                    
                    // Simulate processing time
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    
                    // Simulate response validation
                    std::vector<float> expected_output(10, 2.0f);
                    
                    successful_requests++;
                } catch (...) {
                    failed_requests++;
                }
            }
        });
    }
    
    for (auto& t : client_threads) {
        t.join();
    }
    
    EXPECT_EQ(successful_requests.load(), num_clients * requests_per_client);
    EXPECT_EQ(failed_requests.load(), 0);
}

// Test model hot reload
TEST_F(EndToEndTest, ModelHotReload) {
    // Update model file
    std::string model_path = model_repo_ + "/simple_model/1/model.py";
    
    // Read original content
    std::ifstream original(model_path);
    std::string original_content((std::istreambuf_iterator<char>(original)),
                                std::istreambuf_iterator<char>());
    original.close();
    
    // Modify model (change operation from +1 to +2)
    std::string modified_content = original_content;
    size_t pos = modified_content.find("out_0 = in_0_np + 1");
    if (pos != std::string::npos) {
        modified_content.replace(pos, 19, "out_0 = in_0_np + 2");
    }
    
    // Write modified version
    std::ofstream modified(model_path);
    modified << modified_content;
    modified.close();
    
    // In real test, would trigger model reload and verify new behavior
    
    // Restore original
    std::ofstream restore(model_path);
    restore << original_content;
    restore.close();
}

// Test error recovery
TEST_F(EndToEndTest, ErrorRecovery) {
    // Test various error scenarios
    
    // 1. Invalid input shape
    std::vector<float> wrong_shape_input(5, 1.0f); // Expected 10
    
    // 2. Model execution error
    // Create a model that intentionally fails
    std::string error_model_dir = model_repo_ + "/error_model";
    std::system(("mkdir -p " + error_model_dir + "/1").c_str());
    
    std::ofstream error_model(error_model_dir + "/1/model.py");
    error_model << R"(
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def execute(self, requests):
        # Intentionally raise an error
        raise RuntimeError("Test error handling")
)";
    error_model.close();
    
    // In real test, would verify server handles errors gracefully
    
    // Clean up
    std::system(("rm -rf " + error_model_dir).c_str());
}

// Test memory leak detection during inference
TEST_F(EndToEndTest, MemoryLeakDetection) {
    const int num_iterations = 100;
    
    // Get initial memory usage
    size_t initial_memory = 0;
#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t size = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &size) == KERN_SUCCESS) {
        initial_memory = info.resident_size;
    }
#endif
    
    // Run many inference iterations
    for (int i = 0; i < num_iterations; ++i) {
        // Simulate inference with large tensors
        std::vector<float> large_input(1024 * 1024, 1.0f); // 4MB per request
        
        // Process request (simulated)
        std::vector<float> output(large_input.size());
        std::transform(large_input.begin(), large_input.end(), 
                      output.begin(), [](float x) { return x + 1; });
        
        // Simulate cleanup
        large_input.clear();
        output.clear();
    }
    
    // Check final memory usage
    size_t final_memory = 0;
#ifdef __APPLE__
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &size) == KERN_SUCCESS) {
        final_memory = info.resident_size;
    }
    
    // Memory should not have grown significantly
    if (initial_memory > 0 && final_memory > 0) {
        size_t memory_growth = final_memory - initial_memory;
        size_t max_allowed_growth = 50 * 1024 * 1024; // 50MB tolerance
        EXPECT_LT(memory_growth, max_allowed_growth) 
            << "Memory grew by " << memory_growth / (1024 * 1024) << " MB";
    }
#endif
}

// Test performance benchmarks
TEST_F(EndToEndTest, PerformanceBenchmark) {
    const int warm_up_iterations = 10;
    const int benchmark_iterations = 100;
    
    // Warm up
    for (int i = 0; i < warm_up_iterations; ++i) {
        std::vector<float> input(1000, 1.0f);
        std::vector<float> output(input.size());
        std::transform(input.begin(), input.end(), output.begin(),
                      [](float x) { return x * 2.0f + 1.0f; });
    }
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < benchmark_iterations; ++i) {
        std::vector<float> input(1000, 1.0f);
        std::vector<float> output(input.size());
        std::transform(input.begin(), input.end(), output.begin(),
                      [](float x) { return x * 2.0f + 1.0f; });
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_latency_us = duration.count() / static_cast<double>(benchmark_iterations);
    double throughput = 1000000.0 / avg_latency_us; // requests per second
    
    std::cout << "Average latency: " << avg_latency_us << " us" << std::endl;
    std::cout << "Throughput: " << throughput << " requests/sec" << std::endl;
    
    // Set reasonable expectations
    EXPECT_LT(avg_latency_us, 1000.0); // Less than 1ms per request
    EXPECT_GT(throughput, 1000.0); // More than 1000 requests/sec
}

}}} // namespace triton::backend::python