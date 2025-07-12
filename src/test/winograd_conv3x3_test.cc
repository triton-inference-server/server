// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Winograd Convolution Tests

#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <cmath>

#include "../apple/winograd_conv3x3.h"
#include "../apple/amx_provider.h"

using namespace triton::apple;

class WinogradConv3x3Test : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize AMX if available
        AMXProvider::Instance().Initialize();
    }
    
    // Helper to generate random data
    void GenerateRandomData(float* data, size_t size, float min_val = -1.0f, float max_val = 1.0f) {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(min_val, max_val);
        
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
    }
    
    // Direct convolution reference implementation
    void DirectConv3x3Reference(
        const float* input,
        const float* kernel,
        float* output,
        size_t batch_size,
        size_t height,
        size_t width,
        size_t in_channels,
        size_t out_channels,
        size_t stride = 1,
        size_t padding = 1,
        const float* bias = nullptr) {
        
        // Clear output
        size_t output_size = batch_size * height * width * out_channels;
        std::fill(output, output + output_size, 0.0f);
        
        // Convolution
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t oc = 0; oc < out_channels; ++oc) {
                for (size_t y = 0; y < height; ++y) {
                    for (size_t x = 0; x < width; ++x) {
                        float sum = 0;
                        
                        for (size_t ic = 0; ic < in_channels; ++ic) {
                            for (size_t ky = 0; ky < 3; ++ky) {
                                for (size_t kx = 0; kx < 3; ++kx) {
                                    int in_y = y * stride + ky - padding;
                                    int in_x = x * stride + kx - padding;
                                    
                                    if (in_y >= 0 && in_y < height && 
                                        in_x >= 0 && in_x < width) {
                                        
                                        size_t in_idx = ((b * height + in_y) * width + in_x) * 
                                                       in_channels + ic;
                                        size_t k_idx = (oc * in_channels + ic) * 9 + ky * 3 + kx;
                                        
                                        sum += input[in_idx] * kernel[k_idx];
                                    }
                                }
                            }
                        }
                        
                        size_t out_idx = ((b * height + y) * width + x) * out_channels + oc;
                        output[out_idx] = sum;
                        
                        if (bias) {
                            output[out_idx] += bias[oc];
                        }
                    }
                }
            }
        }
    }
    
    // Compare outputs with tolerance
    bool CompareOutputs(const float* ref, const float* test, size_t size, 
                       float rel_tol = 1e-3f, float abs_tol = 1e-5f) {
        for (size_t i = 0; i < size; ++i) {
            float diff = std::abs(ref[i] - test[i]);
            float tol = abs_tol + rel_tol * std::abs(ref[i]);
            
            if (diff > tol) {
                std::cerr << "Mismatch at index " << i << ": ref=" << ref[i] 
                         << ", test=" << test[i] << ", diff=" << diff 
                         << ", tol=" << tol << std::endl;
                return false;
            }
        }
        return true;
    }
};

TEST_F(WinogradConv3x3Test, BasicConvolution) {
    // Test configuration
    WinogradConv3x3::Config config;
    config.batch_size = 1;
    config.height = 8;
    config.width = 8;
    config.in_channels = 3;
    config.out_channels = 16;
    config.stride = 1;
    config.padding = 1;
    
    // Create convolution
    WinogradConv3x3 winograd;
    ASSERT_EQ(winograd.Initialize(config), nullptr);
    
    // Allocate data
    size_t input_size = config.batch_size * config.height * config.width * config.in_channels;
    size_t kernel_size = config.out_channels * config.in_channels * 9;
    size_t output_size = config.batch_size * config.height * config.width * config.out_channels;
    
    std::vector<float> input(input_size);
    std::vector<float> kernel(kernel_size);
    std::vector<float> output_winograd(output_size);
    std::vector<float> output_ref(output_size);
    
    // Generate random data
    GenerateRandomData(input.data(), input_size);
    GenerateRandomData(kernel.data(), kernel_size);
    
    // Execute Winograd
    ASSERT_EQ(winograd.Execute(input.data(), kernel.data(), output_winograd.data()), nullptr);
    
    // Execute reference
    DirectConv3x3Reference(input.data(), kernel.data(), output_ref.data(),
                          config.batch_size, config.height, config.width,
                          config.in_channels, config.out_channels);
    
    // Compare results
    EXPECT_TRUE(CompareOutputs(output_ref.data(), output_winograd.data(), output_size));
}

TEST_F(WinogradConv3x3Test, WithBias) {
    WinogradConv3x3::Config config;
    config.batch_size = 2;
    config.height = 16;
    config.width = 16;
    config.in_channels = 32;
    config.out_channels = 64;
    
    WinogradConv3x3 winograd;
    ASSERT_EQ(winograd.Initialize(config), nullptr);
    
    // Allocate data
    size_t input_size = config.batch_size * config.height * config.width * config.in_channels;
    size_t kernel_size = config.out_channels * config.in_channels * 9;
    size_t output_size = config.batch_size * config.height * config.width * config.out_channels;
    
    std::vector<float> input(input_size);
    std::vector<float> kernel(kernel_size);
    std::vector<float> bias(config.out_channels);
    std::vector<float> output_winograd(output_size);
    std::vector<float> output_ref(output_size);
    
    GenerateRandomData(input.data(), input_size);
    GenerateRandomData(kernel.data(), kernel_size, -0.1f, 0.1f);  // Smaller kernel values
    GenerateRandomData(bias.data(), config.out_channels, -1.0f, 1.0f);
    
    // Execute with bias
    ASSERT_EQ(winograd.Execute(input.data(), kernel.data(), output_winograd.data(), 
                              bias.data()), nullptr);
    
    // Reference with bias
    DirectConv3x3Reference(input.data(), kernel.data(), output_ref.data(),
                          config.batch_size, config.height, config.width,
                          config.in_channels, config.out_channels,
                          1, 1, bias.data());
    
    // Compare - may need slightly higher tolerance due to accumulation
    EXPECT_TRUE(CompareOutputs(output_ref.data(), output_winograd.data(), output_size, 
                              1e-2f, 1e-4f));
}

TEST_F(WinogradConv3x3Test, LargeProblem) {
    WinogradConv3x3::Config config;
    config.batch_size = 8;
    config.height = 224;
    config.width = 224;
    config.in_channels = 64;
    config.out_channels = 128;
    config.use_amx = true;
    config.use_metal = false;
    
    WinogradConv3x3 winograd;
    ASSERT_EQ(winograd.Initialize(config), nullptr);
    
    // Just test that it runs without crashing for large sizes
    size_t input_size = config.batch_size * config.height * config.width * config.in_channels;
    size_t kernel_size = config.out_channels * config.in_channels * 9;
    size_t output_size = config.batch_size * config.height * config.width * config.out_channels;
    
    std::vector<float> input(input_size);
    std::vector<float> kernel(kernel_size);
    std::vector<float> output(output_size);
    
    GenerateRandomData(input.data(), input_size);
    GenerateRandomData(kernel.data(), kernel_size, -0.05f, 0.05f);
    
    ASSERT_EQ(winograd.Execute(input.data(), kernel.data(), output.data()), nullptr);
    
    // Basic sanity check - output shouldn't be all zeros
    float sum = 0;
    for (size_t i = 0; i < output_size; ++i) {
        sum += std::abs(output[i]);
    }
    EXPECT_GT(sum, 0);
}

TEST_F(WinogradConv3x3Test, TransformedKernelReuse) {
    WinogradConv3x3::Config config;
    config.batch_size = 1;
    config.height = 32;
    config.width = 32;
    config.in_channels = 16;
    config.out_channels = 32;
    
    WinogradConv3x3 winograd;
    ASSERT_EQ(winograd.Initialize(config), nullptr);
    
    // Allocate data
    size_t input_size = config.batch_size * config.height * config.width * config.in_channels;
    size_t kernel_size = config.out_channels * config.in_channels * 9;
    size_t output_size = config.batch_size * config.height * config.width * config.out_channels;
    
    std::vector<float> input1(input_size);
    std::vector<float> input2(input_size);
    std::vector<float> kernel(kernel_size);
    std::vector<float> output1(output_size);
    std::vector<float> output2(output_size);
    
    GenerateRandomData(input1.data(), input_size);
    GenerateRandomData(input2.data(), input_size);
    GenerateRandomData(kernel.data(), kernel_size);
    
    // Transform kernel once
    size_t transformed_size = winograd.GetTransformedKernelSize() / sizeof(float);
    std::vector<float> transformed_kernel(transformed_size);
    ASSERT_EQ(winograd.TransformKernel(kernel.data(), transformed_kernel.data()), nullptr);
    
    // Execute with pre-transformed kernel twice
    ASSERT_EQ(winograd.ExecuteWithTransformedKernel(input1.data(), transformed_kernel.data(), 
                                                   output1.data()), nullptr);
    ASSERT_EQ(winograd.ExecuteWithTransformedKernel(input2.data(), transformed_kernel.data(), 
                                                   output2.data()), nullptr);
    
    // Both outputs should be valid
    float sum1 = 0, sum2 = 0;
    for (size_t i = 0; i < output_size; ++i) {
        sum1 += std::abs(output1[i]);
        sum2 += std::abs(output2[i]);
    }
    EXPECT_GT(sum1, 0);
    EXPECT_GT(sum2, 0);
    EXPECT_NE(sum1, sum2);  // Different inputs should give different outputs
}

TEST_F(WinogradConv3x3Test, AutoSelector) {
    // Test small problem - should select direct
    auto result = WinogradAutoSelector::SelectOptimal(1, 4, 4, 8, 8);
    EXPECT_EQ(result.type, WinogradAutoSelector::WinogradType::NONE);
    
    // Test medium problem - should select F(2x2)
    result = WinogradAutoSelector::SelectOptimal(4, 56, 56, 64, 64);
    EXPECT_EQ(result.type, WinogradAutoSelector::WinogradType::F2x2_3x3);
    
    // Test stride > 1 - should select direct
    result = WinogradAutoSelector::SelectOptimal(1, 224, 224, 3, 64, 2);
    EXPECT_EQ(result.type, WinogradAutoSelector::WinogradType::NONE);
    
    // Test large problem - might select F(4x4) when implemented
    result = WinogradAutoSelector::SelectOptimal(32, 224, 224, 128, 256);
    EXPECT_NE(result.type, WinogradAutoSelector::WinogradType::NONE);
}

TEST_F(WinogradConv3x3Test, Performance) {
    WinogradConv3x3::Config config;
    config.batch_size = 8;
    config.height = 56;
    config.width = 56;
    config.in_channels = 64;
    config.out_channels = 64;
    config.use_amx = AMXProvider::Instance().IsEnabled();
    
    WinogradConv3x3 winograd;
    ASSERT_EQ(winograd.Initialize(config), nullptr);
    
    // Profile and check speedup
    auto profile_result = winograd.Profile(20);
    
    std::cout << "\nPerformance Test Results:\n";
    std::cout << "  Problem size: " << config.batch_size << "x" << config.height 
              << "x" << config.width << "x" << config.in_channels 
              << "->" << config.out_channels << "\n";
    std::cout << "  Winograd time: " << profile_result.winograd_time_ms << " ms\n";
    std::cout << "  Direct time: " << profile_result.direct_time_ms << " ms\n";
    std::cout << "  Speedup: " << profile_result.speedup << "x\n";
    std::cout << "  Memory usage: " << profile_result.memory_usage_mb << " MB\n";
    
    // Expect at least some speedup for this problem size
    EXPECT_GT(profile_result.speedup, 1.2f);
}

TEST_F(WinogradConv3x3Test, WorkspaceSize) {
    WinogradConv3x3::Config config;
    config.batch_size = 4;
    config.height = 28;
    config.width = 28;
    config.in_channels = 32;
    config.out_channels = 64;
    
    WinogradConv3x3 winograd;
    ASSERT_EQ(winograd.Initialize(config), nullptr);
    
    size_t workspace = winograd.GetWorkspaceSize();
    EXPECT_GT(workspace, 0);
    
    // Workspace should scale with problem size
    size_t tiles = ((config.height - 2) / 2) * ((config.width - 2) / 2);
    size_t expected_min = config.batch_size * tiles * 
                         (config.in_channels + config.out_channels) * 16 * sizeof(float);
    EXPECT_GE(workspace, expected_min);
}

// Benchmark different configurations
TEST_F(WinogradConv3x3Test, BenchmarkConfigurations) {
    struct TestConfig {
        size_t batch_size;
        size_t height;
        size_t width;
        size_t in_channels;
        size_t out_channels;
        const char* name;
    };
    
    std::vector<TestConfig> configs = {
        {1, 224, 224, 3, 64, "ResNet-FirstLayer"},
        {8, 56, 56, 64, 64, "ResNet-Conv3x3"},
        {16, 28, 28, 128, 128, "ResNet-Downsample"},
        {32, 14, 14, 256, 256, "ResNet-Deep"},
        {1, 224, 224, 32, 32, "MobileNet-Depthwise"},
    };
    
    std::cout << "\n=== Winograd Benchmark Results ===\n";
    
    for (const auto& test_config : configs) {
        std::cout << "\n" << test_config.name << ":\n";
        WinogradAutoSelector::BenchmarkAll(
            test_config.batch_size,
            test_config.height,
            test_config.width,
            test_config.in_channels,
            test_config.out_channels
        );
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}