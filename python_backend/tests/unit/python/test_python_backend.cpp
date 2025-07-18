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
#include <Python.h>
#include <string>
#include <vector>
#include <memory>
#include <fstream>

namespace triton { namespace backend { namespace python {

class PythonBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize Python interpreter if not already initialized
        if (!Py_IsInitialized()) {
            Py_Initialize();
            python_initialized_ = true;
        }
        
        // Set up test model directory
        test_model_dir_ = "/tmp/triton_python_test_models";
        std::system(("mkdir -p " + test_model_dir_).c_str());
    }

    void TearDown() override {
        // Clean up test directory
        std::system(("rm -rf " + test_model_dir_).c_str());
        
        // Don't finalize Python if we didn't initialize it
        if (python_initialized_ && Py_IsInitialized()) {
            // Note: Py_Finalize() can cause issues with subsequent tests
            // so we leave Python initialized
        }
    }
    
    void CreateTestModel(const std::string& model_name, const std::string& code) {
        std::string model_dir = test_model_dir_ + "/" + model_name;
        std::system(("mkdir -p " + model_dir + "/1").c_str());
        
        // Write model.py
        std::ofstream model_file(model_dir + "/1/model.py");
        model_file << code;
        model_file.close();
        
        // Write config.pbtxt
        std::ofstream config_file(model_dir + "/config.pbtxt");
        config_file << R"(
name: ")" << model_name << R"("
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
    
    bool python_initialized_ = false;
    std::string test_model_dir_;
};

// Test basic Python execution
TEST_F(PythonBackendTest, BasicPythonExecution) {
    // Test simple Python code execution
    PyObject* main_module = PyImport_AddModule("__main__");
    ASSERT_NE(main_module, nullptr);
    
    PyObject* global_dict = PyModule_GetDict(main_module);
    ASSERT_NE(global_dict, nullptr);
    
    // Execute simple Python code
    const char* code = "result = 2 + 2";
    PyObject* result = PyRun_String(code, Py_file_input, global_dict, global_dict);
    ASSERT_NE(result, nullptr);
    Py_DECREF(result);
    
    // Get the result
    PyObject* py_result = PyDict_GetItemString(global_dict, "result");
    ASSERT_NE(py_result, nullptr);
    EXPECT_EQ(PyLong_AsLong(py_result), 4);
}

// Test model loading
TEST_F(PythonBackendTest, ModelLoading) {
    const std::string model_code = R"(
import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = args['model_config']
        self.initialized = True
        
    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            input_array = input_tensor.as_numpy()
            
            # Simple operation: multiply by 2
            output_array = input_array * 2
            
            output_tensor = pb_utils.Tensor("OUTPUT0", output_array)
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)
        
        return responses
    
    def finalize(self):
        pass
)";
    
    CreateTestModel("test_model", model_code);
    
    // Simulate loading the model
    PyObject* sys_path = PySys_GetObject("path");
    ASSERT_NE(sys_path, nullptr);
    
    std::string model_path = test_model_dir_ + "/test_model/1";
    PyObject* path = PyUnicode_FromString(model_path.c_str());
    PyList_Append(sys_path, path);
    Py_DECREF(path);
    
    // Import the model module
    PyObject* model_module = PyImport_ImportModule("model");
    if (model_module == nullptr) {
        PyErr_Print();
        FAIL() << "Failed to import model module";
    }
    
    // Get the model class
    PyObject* model_class = PyObject_GetAttrString(model_module, "TritonPythonModel");
    ASSERT_NE(model_class, nullptr);
    
    // Instantiate the model
    PyObject* model_instance = PyObject_CallObject(model_class, nullptr);
    ASSERT_NE(model_instance, nullptr);
    
    // Clean up
    Py_DECREF(model_instance);
    Py_DECREF(model_class);
    Py_DECREF(model_module);
}

// Test error handling
TEST_F(PythonBackendTest, ErrorHandling) {
    // Test syntax error
    const char* bad_code = "def bad_function(\n    pass";
    PyObject* main_module = PyImport_AddModule("__main__");
    PyObject* global_dict = PyModule_GetDict(main_module);
    
    PyObject* result = PyRun_String(bad_code, Py_file_input, global_dict, global_dict);
    EXPECT_EQ(result, nullptr);
    
    // Check if error is set
    EXPECT_TRUE(PyErr_Occurred() != nullptr);
    
    // Clear the error
    PyErr_Clear();
    
    // Test runtime error
    const char* error_code = "1 / 0";
    result = PyRun_String(error_code, Py_eval_input, global_dict, global_dict);
    EXPECT_EQ(result, nullptr);
    EXPECT_TRUE(PyErr_Occurred() != nullptr);
    
    // Get error details
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    
    if (type) {
        PyObject* type_name = PyObject_GetAttrString(type, "__name__");
        if (type_name && PyUnicode_Check(type_name)) {
            const char* error_type = PyUnicode_AsUTF8(type_name);
            EXPECT_STREQ(error_type, "ZeroDivisionError");
            Py_DECREF(type_name);
        }
        Py_XDECREF(type);
    }
    
    Py_XDECREF(value);
    Py_XDECREF(traceback);
}

// Test memory management with Python objects
TEST_F(PythonBackendTest, PythonMemoryManagement) {
    // Create and destroy many Python objects
    const int num_objects = 10000;
    
    for (int i = 0; i < num_objects; ++i) {
        // Create a list
        PyObject* list = PyList_New(100);
        ASSERT_NE(list, nullptr);
        
        // Fill with integers
        for (int j = 0; j < 100; ++j) {
            PyObject* num = PyLong_FromLong(j);
            PyList_SET_ITEM(list, j, num); // Steals reference
        }
        
        // Create a dictionary
        PyObject* dict = PyDict_New();
        ASSERT_NE(dict, nullptr);
        
        // Add some items
        PyObject* key = PyUnicode_FromString("list");
        PyDict_SetItem(dict, key, list);
        Py_DECREF(key);
        
        // Clean up
        Py_DECREF(list);
        Py_DECREF(dict);
    }
}

// Test numpy integration
TEST_F(PythonBackendTest, NumpyIntegration) {
    // Import numpy
    PyObject* numpy = PyImport_ImportModule("numpy");
    if (numpy == nullptr) {
        PyErr_Clear();
        GTEST_SKIP() << "NumPy not available";
    }
    
    // Create numpy array
    PyObject* main_module = PyImport_AddModule("__main__");
    PyObject* global_dict = PyModule_GetDict(main_module);
    
    PyDict_SetItemString(global_dict, "np", numpy);
    
    const char* code = R"(
import numpy as np
arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
result = np.sum(arr)
)";
    
    PyObject* result = PyRun_String(code, Py_file_input, global_dict, global_dict);
    ASSERT_NE(result, nullptr);
    Py_DECREF(result);
    
    // Get the sum result
    PyObject* sum_result = PyDict_GetItemString(global_dict, "result");
    ASSERT_NE(sum_result, nullptr);
    
    double sum_value = PyFloat_AsDouble(sum_result);
    EXPECT_DOUBLE_EQ(sum_value, 15.0);
    
    Py_DECREF(numpy);
}

// Test concurrent Python execution
TEST_F(PythonBackendTest, ConcurrentExecution) {
    // Note: CPython has the GIL, so true parallelism is limited
    // This test verifies thread safety with GIL
    
    const int num_threads = 4;
    const int iterations = 100;
    std::vector<std::thread> threads;
    std::atomic<int> success_count(0);
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&success_count, iterations, t]() {
            for (int i = 0; i < iterations; ++i) {
                // Acquire GIL
                PyGILState_STATE gstate = PyGILState_Ensure();
                
                // Execute Python code
                PyObject* main_module = PyImport_AddModule("__main__");
                PyObject* global_dict = PyModule_GetDict(main_module);
                
                std::string code = "thread_" + std::to_string(t) + 
                                  "_iter_" + std::to_string(i) + " = " + 
                                  std::to_string(t * 1000 + i);
                
                PyObject* result = PyRun_String(code.c_str(), Py_file_input, 
                                              global_dict, global_dict);
                
                if (result != nullptr) {
                    success_count++;
                    Py_DECREF(result);
                }
                
                // Release GIL
                PyGILState_Release(gstate);
                
                // Small delay
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    EXPECT_EQ(success_count.load(), num_threads * iterations);
}

#ifdef TRITON_PLATFORM_MACOS
// Test macOS-specific Python features
TEST_F(PythonBackendTest, MacOSPythonFeatures) {
    // Test framework Python on macOS
    PyObject* sys_module = PyImport_ImportModule("sys");
    ASSERT_NE(sys_module, nullptr);
    
    // Get Python version
    PyObject* version = PyObject_GetAttrString(sys_module, "version");
    ASSERT_NE(version, nullptr);
    
    if (PyUnicode_Check(version)) {
        const char* version_str = PyUnicode_AsUTF8(version);
        std::cout << "Python version: " << version_str << std::endl;
    }
    
    // Check platform
    PyObject* platform = PyImport_ImportModule("platform");
    ASSERT_NE(platform, nullptr);
    
    PyObject* system = PyObject_CallMethod(platform, "system", nullptr);
    ASSERT_NE(system, nullptr);
    
    if (PyUnicode_Check(system)) {
        const char* system_str = PyUnicode_AsUTF8(system);
        EXPECT_STREQ(system_str, "Darwin");
    }
    
    // Test macOS-specific modules
    PyObject* objc = PyImport_ImportModule("objc");
    if (objc == nullptr) {
        PyErr_Clear();
        std::cout << "PyObjC not available" << std::endl;
    } else {
        Py_DECREF(objc);
    }
    
    Py_DECREF(system);
    Py_DECREF(platform);
    Py_DECREF(version);
    Py_DECREF(sys_module);
}
#endif

}}} // namespace triton::backend::python