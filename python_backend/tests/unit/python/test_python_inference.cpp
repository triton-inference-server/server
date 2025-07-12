// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include <gtest/gtest.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <chrono>

namespace triton { namespace backend { namespace python {

class PythonInferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!Py_IsInitialized()) {
            Py_Initialize();
        }
        
        // Initialize NumPy
        import_array();
    }
    
    PyObject* CreateNumpyArray(const std::vector<float>& data) {
        npy_intp dims[1] = {static_cast<npy_intp>(data.size())};
        PyObject* array = PyArray_SimpleNewFromData(
            1, dims, NPY_FLOAT32, (void*)data.data());
        return array;
    }
};

TEST_F(PythonInferenceTest, SimpleInference) {
    // Create test data
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    PyObject* input_array = CreateNumpyArray(input_data);
    ASSERT_NE(input_array, nullptr);
    
    // Simulate inference operation
    PyObject* main_module = PyImport_AddModule("__main__");
    PyObject* global_dict = PyModule_GetDict(main_module);
    
    PyDict_SetItemString(global_dict, "input_data", input_array);
    
    const char* inference_code = R"(
import numpy as np
output_data = input_data * 2.0 + 1.0
)";
    
    PyObject* result = PyRun_String(inference_code, Py_file_input, 
                                   global_dict, global_dict);
    ASSERT_NE(result, nullptr);
    Py_DECREF(result);
    
    // Get output
    PyObject* output_array = PyDict_GetItemString(global_dict, "output_data");
    ASSERT_NE(output_array, nullptr);
    
    // Verify results
    ASSERT_TRUE(PyArray_Check(output_array));
    float* output_ptr = (float*)PyArray_DATA((PyArrayObject*)output_array);
    
    for (size_t i = 0; i < input_data.size(); ++i) {
        EXPECT_FLOAT_EQ(output_ptr[i], input_data[i] * 2.0f + 1.0f);
    }
    
    Py_DECREF(input_array);
}

TEST_F(PythonInferenceTest, BatchedInference) {
    const int batch_size = 4;
    const int feature_size = 10;
    
    // Create batch data
    std::vector<float> batch_data(batch_size * feature_size);
    for (int i = 0; i < batch_size * feature_size; ++i) {
        batch_data[i] = static_cast<float>(i);
    }
    
    npy_intp dims[2] = {batch_size, feature_size};
    PyObject* batch_array = PyArray_SimpleNewFromData(
        2, dims, NPY_FLOAT32, (void*)batch_data.data());
    ASSERT_NE(batch_array, nullptr);
    
    PyObject* main_module = PyImport_AddModule("__main__");
    PyObject* global_dict = PyModule_GetDict(main_module);
    
    PyDict_SetItemString(global_dict, "batch_input", batch_array);
    
    const char* batch_code = R"(
import numpy as np

# Simulate batch processing
batch_mean = np.mean(batch_input, axis=1, keepdims=True)
batch_std = np.std(batch_input, axis=1, keepdims=True) + 1e-7
batch_normalized = (batch_input - batch_mean) / batch_std
)";
    
    PyObject* result = PyRun_String(batch_code, Py_file_input, 
                                   global_dict, global_dict);
    ASSERT_NE(result, nullptr);
    Py_DECREF(result);
    
    PyObject* output = PyDict_GetItemString(global_dict, "batch_normalized");
    ASSERT_NE(output, nullptr);
    ASSERT_TRUE(PyArray_Check(output));
    
    // Verify shape
    PyArrayObject* output_array = (PyArrayObject*)output;
    EXPECT_EQ(PyArray_NDIM(output_array), 2);
    EXPECT_EQ(PyArray_DIM(output_array, 0), batch_size);
    EXPECT_EQ(PyArray_DIM(output_array, 1), feature_size);
    
    Py_DECREF(batch_array);
}

TEST_F(PythonInferenceTest, InferencePerformance) {
    const int num_iterations = 100;
    const int data_size = 1000;
    
    std::vector<float> data(data_size, 1.0f);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < num_iterations; ++iter) {
        npy_intp dims[1] = {data_size};
        PyObject* array = PyArray_SimpleNewFromData(
            1, dims, NPY_FLOAT32, (void*)data.data());
        
        PyObject* main_module = PyImport_AddModule("__main__");
        PyObject* global_dict = PyModule_GetDict(main_module);
        
        PyDict_SetItemString(global_dict, "x", array);
        
        PyObject* result = PyRun_String(
            "y = x * 2.0 + 3.0", Py_file_input, global_dict, global_dict);
        
        if (result) Py_DECREF(result);
        Py_DECREF(array);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_latency_us = duration.count() / static_cast<double>(num_iterations);
    std::cout << "Average inference latency: " << avg_latency_us << " us" << std::endl;
    
    // Performance should be reasonable
    EXPECT_LT(avg_latency_us, 10000.0); // Less than 10ms per inference
}

TEST_F(PythonInferenceTest, ErrorHandlingDuringInference) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    PyObject* array = CreateNumpyArray(data);
    
    PyObject* main_module = PyImport_AddModule("__main__");
    PyObject* global_dict = PyModule_GetDict(main_module);
    
    PyDict_SetItemString(global_dict, "data", array);
    
    // Code that will cause an error
    const char* error_code = R"(
import numpy as np
# This will cause a shape mismatch error
result = np.matmul(data, np.ones((5, 5)))
)";
    
    PyObject* result = PyRun_String(error_code, Py_file_input, 
                                   global_dict, global_dict);
    
    EXPECT_EQ(result, nullptr);
    EXPECT_TRUE(PyErr_Occurred() != nullptr);
    
    // Clear error
    PyErr_Clear();
    
    Py_DECREF(array);
}

#ifdef TRITON_PLATFORM_MACOS
TEST_F(PythonInferenceTest, MacOSAcceleratedCompute) {
    // Test using Accelerate framework through NumPy
    const int size = 1000;
    std::vector<float> a(size), b(size);
    
    for (int i = 0; i < size; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(size - i);
    }
    
    npy_intp dims[1] = {size};
    PyObject* array_a = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, a.data());
    PyObject* array_b = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT32, b.data());
    
    PyObject* main_module = PyImport_AddModule("__main__");
    PyObject* global_dict = PyModule_GetDict(main_module);
    
    PyDict_SetItemString(global_dict, "a", array_a);
    PyDict_SetItemString(global_dict, "b", array_b);
    
    // NumPy on macOS can use Accelerate framework
    const char* accelerate_code = R"(
import numpy as np
# These operations may use Accelerate framework on macOS
dot_product = np.dot(a, b)
fft_result = np.fft.fft(a[:128])  # Power of 2 for FFT
)";
    
    PyObject* result = PyRun_String(accelerate_code, Py_file_input,
                                   global_dict, global_dict);
    ASSERT_NE(result, nullptr);
    
    Py_DECREF(result);
    Py_DECREF(array_a);
    Py_DECREF(array_b);
}
#endif

}}} // namespace triton::backend::python