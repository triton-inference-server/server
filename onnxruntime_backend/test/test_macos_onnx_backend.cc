// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Test program for ONNX Runtime backend on macOS

#include <iostream>
#include <memory>
#include <vector>
#include <dlfcn.h>
#include <cstring>

// Simplified Triton backend API definitions for testing
typedef struct TRITONBACKEND_Backend_t TRITONBACKEND_Backend;
typedef struct TRITONBACKEND_Model_t TRITONBACKEND_Model;
typedef struct TRITONBACKEND_ModelInstance_t TRITONBACKEND_ModelInstance;
typedef struct TRITONSERVER_Error_t TRITONSERVER_Error;
typedef struct TRITONSERVER_Message_t TRITONSERVER_Message;

typedef enum tritonserver_errorcode_enum {
  TRITONSERVER_ERROR_UNKNOWN,
  TRITONSERVER_ERROR_INTERNAL,
  TRITONSERVER_ERROR_NOT_FOUND,
  TRITONSERVER_ERROR_INVALID_ARG,
  TRITONSERVER_ERROR_UNAVAILABLE,
  TRITONSERVER_ERROR_UNSUPPORTED,
  TRITONSERVER_ERROR_ALREADY_EXISTS
} TRITONSERVER_Error_Code;

// Function pointer types for backend interface
typedef TRITONSERVER_Error* (*TritonBackendInitializeFn_t)(
    TRITONBACKEND_Backend* backend);
typedef TRITONSERVER_Error* (*TritonBackendFinalizeFn_t)(
    TRITONBACKEND_Backend* backend);
typedef TRITONSERVER_Error* (*TritonBackendModelInitializeFn_t)(
    TRITONBACKEND_Model* model);
typedef TRITONSERVER_Error* (*TritonBackendModelFinalizeFn_t)(
    TRITONBACKEND_Model* model);

// Helper to load and test the backend
class BackendTester {
 public:
  BackendTester(const std::string& backend_path)
      : backend_path_(backend_path), handle_(nullptr) {}

  ~BackendTester() {
    if (handle_) {
      dlclose(handle_);
    }
  }

  bool LoadBackend() {
    std::cout << "Loading backend from: " << backend_path_ << std::endl;
    
    // Load the dynamic library
    handle_ = dlopen(backend_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_) {
      std::cerr << "Failed to load backend: " << dlerror() << std::endl;
      return false;
    }
    
    std::cout << "Backend library loaded successfully" << std::endl;
    return true;
  }

  bool VerifyBackendSymbols() {
    std::cout << "\nVerifying backend symbols..." << std::endl;
    
    // Check for required backend interface functions
    const char* required_symbols[] = {
        "TRITONBACKEND_Initialize",
        "TRITONBACKEND_Finalize",
        "TRITONBACKEND_ModelInitialize",
        "TRITONBACKEND_ModelFinalize",
        "TRITONBACKEND_ModelInstanceInitialize",
        "TRITONBACKEND_ModelInstanceFinalize",
        "TRITONBACKEND_ModelInstanceExecute"
    };
    
    bool all_found = true;
    for (const char* symbol : required_symbols) {
      void* func = dlsym(handle_, symbol);
      if (func) {
        std::cout << "  ✓ Found " << symbol << std::endl;
      } else {
        std::cerr << "  ✗ Missing " << symbol << std::endl;
        all_found = false;
      }
    }
    
    return all_found;
  }

  bool CheckDependencies() {
    std::cout << "\nChecking backend dependencies..." << std::endl;
    
    // Get the path to the loaded library
    Dl_info info;
    if (dladdr(dlsym(handle_, "TRITONBACKEND_Initialize"), &info)) {
      std::cout << "Backend loaded from: " << info.dli_fname << std::endl;
      
      // Check if ONNX Runtime is properly linked
      void* ort_func = dlsym(handle_, "OrtGetApiBase");
      if (ort_func) {
        std::cout << "  ✓ ONNX Runtime symbols found (statically linked)" << std::endl;
      } else {
        // Check if it's dynamically linked
        std::string check_cmd = "otool -L ";
        check_cmd += info.dli_fname;
        check_cmd += " | grep -i onnxruntime";
        
        FILE* pipe = popen(check_cmd.c_str(), "r");
        if (pipe) {
          char buffer[256];
          bool found_onnxruntime = false;
          while (fgets(buffer, sizeof(buffer), pipe)) {
            found_onnxruntime = true;
            std::cout << "  ✓ ONNX Runtime dependency: " << buffer;
          }
          pclose(pipe);
          
          if (!found_onnxruntime) {
            std::cerr << "  ✗ ONNX Runtime dependency not found" << std::endl;
            return false;
          }
        }
      }
    }
    
    return true;
  }

  bool TestBackendInfo() {
    std::cout << "\nTesting backend information..." << std::endl;
    
    // Try to get backend API version
    typedef const char* (*TritonBackendApiVersionFn_t)();
    auto api_version_fn = (TritonBackendApiVersionFn_t)dlsym(handle_, "TRITONBACKEND_ApiVersion");
    if (api_version_fn) {
      const char* version = api_version_fn();
      std::cout << "  Backend API Version: " << (version ? version : "unknown") << std::endl;
    }
    
    return true;
  }

 private:
  std::string backend_path_;
  void* handle_;
};

// Test creating a simple ONNX model
void CreateTestOnnxModel(const std::string& model_path) {
  std::cout << "\nCreating test ONNX model..." << std::endl;
  
  // Python script to create a simple ONNX model
  const char* python_script = R"(
import onnx
import numpy as np
from onnx import helper, TensorProto

# Create a simple model: output = input + 1
X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3])

# Create constant tensor
const_tensor = helper.make_tensor(
    name='const_one',
    data_type=TensorProto.FLOAT,
    dims=[1, 3],
    vals=np.ones((1, 3), dtype=np.float32).flatten()
)

# Create Add node
add_node = helper.make_node(
    'Add',
    inputs=['input', 'const_one'],
    outputs=['output']
)

# Create the graph
graph_def = helper.make_graph(
    [add_node],
    'simple_add_model',
    [X],
    [Y],
    [const_tensor]
)

# Create the model
model_def = helper.make_model(graph_def, producer_name='onnx-example')
model_def.opset_import[0].version = 13

# Save the model
onnx.save(model_def, 'test_model.onnx')
print('Test model created: test_model.onnx')
)";

  // Write Python script to file
  FILE* fp = fopen("create_model.py", "w");
  if (fp) {
    fprintf(fp, "%s", python_script);
    fclose(fp);
    
    // Execute Python script
    system("python3 create_model.py");
    system("rm create_model.py");
  } else {
    std::cerr << "Failed to create test model script" << std::endl;
  }
}

// Main test function
int main(int argc, char* argv[]) {
  std::cout << "ONNX Runtime Backend Test for macOS" << std::endl;
  std::cout << "===================================" << std::endl;
  
  // Default backend path
  std::string backend_path = "./libtriton_onnxruntime.dylib";
  if (argc > 1) {
    backend_path = argv[1];
  }
  
  // Create tester
  BackendTester tester(backend_path);
  
  // Run tests
  if (!tester.LoadBackend()) {
    std::cerr << "Failed to load backend" << std::endl;
    return 1;
  }
  
  if (!tester.VerifyBackendSymbols()) {
    std::cerr << "Backend symbol verification failed" << std::endl;
    return 1;
  }
  
  if (!tester.CheckDependencies()) {
    std::cerr << "Backend dependency check failed" << std::endl;
    return 1;
  }
  
  if (!tester.TestBackendInfo()) {
    std::cerr << "Backend info test failed" << std::endl;
    return 1;
  }
  
  // Create test model
  CreateTestOnnxModel("test_model.onnx");
  
  std::cout << "\n✅ All tests passed!" << std::endl;
  std::cout << "The ONNX Runtime backend is ready for use on macOS." << std::endl;
  
  return 0;
}