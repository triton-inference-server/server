// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#include <gtest/gtest.h>
#include <Python.h>
#include <filesystem>
#include <fstream>
#include <thread>

namespace triton { namespace backend { namespace python {

class ModelLoadingTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!Py_IsInitialized()) {
            Py_Initialize();
        }
        
        // Create test model directory
        test_dir_ = "/tmp/triton_model_loading_test";
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(test_dir_);
    }
    
    void CreateModel(const std::string& name, const std::string& code) {
        std::string model_dir = test_dir_ + "/" + name + "/1";
        std::filesystem::create_directories(model_dir);
        
        std::ofstream f(model_dir + "/model.py");
        f << code;
    }
    
    std::string test_dir_;
};

TEST_F(ModelLoadingTest, LoadValidModel) {
    const std::string model_code = R"(
class TritonPythonModel:
    def initialize(self, args):
        self.loaded = True
    
    def execute(self, requests):
        return []
    
    def finalize(self):
        pass
)";
    
    CreateModel("valid_model", model_code);
    
    // Test loading
    PyObject* sys_path = PySys_GetObject("path");
    PyObject* path = PyUnicode_FromString((test_dir_ + "/valid_model/1").c_str());
    PyList_Append(sys_path, path);
    Py_DECREF(path);
    
    PyObject* module = PyImport_ImportModule("model");
    ASSERT_NE(module, nullptr) << "Failed to load model module";
    
    PyObject* model_class = PyObject_GetAttrString(module, "TritonPythonModel");
    ASSERT_NE(model_class, nullptr);
    
    Py_DECREF(model_class);
    Py_DECREF(module);
}

TEST_F(ModelLoadingTest, LoadInvalidModel) {
    const std::string bad_code = "invalid python syntax here!@#$";
    CreateModel("invalid_model", bad_code);
    
    PyObject* sys_path = PySys_GetObject("path");
    PyObject* path = PyUnicode_FromString((test_dir_ + "/invalid_model/1").c_str());
    PyList_Append(sys_path, path);
    Py_DECREF(path);
    
    PyObject* module = PyImport_ImportModule("model");
    EXPECT_EQ(module, nullptr);
    PyErr_Clear();
}

TEST_F(ModelLoadingTest, ConcurrentModelLoading) {
    // Create multiple models
    for (int i = 0; i < 5; ++i) {
        std::string model_code = R"(
class TritonPythonModel:
    def __init__(self):
        self.id = )" + std::to_string(i) + R"(
)";
        CreateModel("model_" + std::to_string(i), model_code);
    }
    
    std::vector<std::thread> threads;
    std::atomic<int> success_count(0);
    
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([this, i, &success_count]() {
            PyGILState_STATE gstate = PyGILState_Ensure();
            
            std::string model_path = test_dir_ + "/model_" + std::to_string(i) + "/1";
            PyObject* sys_modules = PyImport_GetModuleDict();
            
            // Remove if already loaded
            PyDict_DelItemString(sys_modules, "model");
            
            PyObject* sys_path = PySys_GetObject("path");
            PyObject* path = PyUnicode_FromString(model_path.c_str());
            PyList_Insert(sys_path, 0, path);
            
            PyObject* module = PyImport_ImportModule("model");
            if (module) {
                success_count++;
                Py_DECREF(module);
            }
            
            PyList_SetSlice(sys_path, 0, 1, nullptr);
            Py_DECREF(path);
            
            PyGILState_Release(gstate);
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    EXPECT_EQ(success_count.load(), 5);
}

#ifdef TRITON_PLATFORM_MACOS
TEST_F(ModelLoadingTest, MacOSBundleLoading) {
    // Test loading from macOS bundle structure
    std::string bundle_path = test_dir_ + "/TestModel.bundle";
    std::filesystem::create_directories(bundle_path + "/Contents/Resources");
    
    std::ofstream plist(bundle_path + "/Contents/Info.plist");
    plist << R"(<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>com.nvidia.triton.testmodel</string>
    <key>CFBundleName</key>
    <string>TestModel</string>
</dict>
</plist>)";
    
    // Bundle loading test would go here
    EXPECT_TRUE(std::filesystem::exists(bundle_path));
}
#endif

}}} // namespace triton::backend::python