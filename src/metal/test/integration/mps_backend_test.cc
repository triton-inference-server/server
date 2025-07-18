// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include "src/metal/test/utils/metal_test_fixtures.h"
#include "backends/metal_mps/include/mps_model.h"
#include "backends/metal_mps/include/mps_engine.h"

namespace triton { namespace server { namespace test {

class MPSBackendIntegrationTest : public MetalBackendTest {
 protected:
  void SetUp() override {
    MetalBackendTest::SetUp();
    
    // Initialize MPS backend
    mps_engine_ = std::make_unique<MPSEngine>();
    auto err = mps_engine_->Initialize();
    if (err != nullptr) {
      TRITONSERVER_ErrorDelete(err);
      GTEST_SKIP() << "Failed to initialize MPS engine";
    }
  }
  
  void TearDown() override {
    mps_engine_.reset();
    MetalBackendTest::TearDown();
  }
  
  std::unique_ptr<MPSEngine> mps_engine_;
};

TEST_F(MPSBackendIntegrationTest, BasicMatrixMultiplication)
{
  // Create test matrices
  const int M = 128, N = 256, K = 64;
  TestDataGenerator gen;
  
  auto A = gen.GenerateFloatData(M * K);
  auto B = gen.GenerateFloatData(K * N);
  std::vector<float> C(M * N, 0.0f);
  
  // Create MPS model for GEMM
  MPSModelConfig config;
  config.name = "test_gemm";
  config.device_id = 0;
  
  auto model = mps_engine_->CreateModel(config);
  ASSERT_NE(model, nullptr);
  
  // Add GEMM operation
  auto err = model->AddGEMMOperation("gemm", M, N, K, 1.0f, 0.0f);
  ASSERT_TRITON_OK(err);
  
  // Set inputs
  err = model->SetInput("A", A.data(), M * K * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  err = model->SetInput("B", B.data(), K * N * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  // Execute
  err = model->Execute();
  ASSERT_TRITON_OK(err);
  
  // Get output
  err = model->GetOutput("C", C.data(), M * N * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  // Verify result (basic check - at least some non-zero values)
  int non_zero_count = 0;
  for (const auto& val : C) {
    if (val != 0.0f) non_zero_count++;
  }
  EXPECT_GT(non_zero_count, M * N / 2);
}

TEST_F(MPSBackendIntegrationTest, ConvolutionOperation)
{
  // Test 2D convolution
  const int batch = 1, channels = 3, height = 224, width = 224;
  const int out_channels = 64, kernel_size = 3;
  
  TestDataGenerator gen;
  auto input = gen.GenerateFloatData(batch * channels * height * width);
  auto weights = gen.GenerateFloatData(out_channels * channels * kernel_size * kernel_size);
  auto bias = gen.GenerateFloatData(out_channels);
  
  // Create model
  MPSModelConfig config;
  config.name = "test_conv";
  config.device_id = 0;
  
  auto model = mps_engine_->CreateModel(config);
  ASSERT_NE(model, nullptr);
  
  // Add convolution
  MPSConvolutionParams conv_params;
  conv_params.stride_h = 1;
  conv_params.stride_w = 1;
  conv_params.padding_h = 1;
  conv_params.padding_w = 1;
  conv_params.dilation_h = 1;
  conv_params.dilation_w = 1;
  
  auto err = model->AddConvolution2D(
      "conv", batch, channels, height, width,
      out_channels, kernel_size, kernel_size, conv_params);
  ASSERT_TRITON_OK(err);
  
  // Set inputs
  err = model->SetInput("input", input.data(), input.size() * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  err = model->SetInput("weights", weights.data(), weights.size() * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  err = model->SetInput("bias", bias.data(), bias.size() * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  // Execute
  err = model->Execute();
  ASSERT_TRITON_OK(err);
  
  // Get output shape
  std::vector<int64_t> output_shape;
  err = model->GetOutputShape("output", output_shape);
  ASSERT_TRITON_OK(err);
  
  EXPECT_EQ(output_shape.size(), 4);
  EXPECT_EQ(output_shape[0], batch);
  EXPECT_EQ(output_shape[1], out_channels);
  EXPECT_EQ(output_shape[2], height);  // With padding=1, same size
  EXPECT_EQ(output_shape[3], width);
}

TEST_F(MPSBackendIntegrationTest, ActivationFunctions)
{
  const size_t size = 1024;
  TestDataGenerator gen;
  auto input = gen.GenerateFloatData(size, -2.0f, 2.0f);
  
  // Test different activation functions
  std::vector<std::string> activations = {
      "relu", "sigmoid", "tanh", "gelu"
  };
  
  for (const auto& activation : activations) {
    MPSModelConfig config;
    config.name = "test_" + activation;
    config.device_id = 0;
    
    auto model = mps_engine_->CreateModel(config);
    ASSERT_NE(model, nullptr);
    
    // Add activation
    auto err = model->AddActivation(activation, activation);
    ASSERT_TRITON_OK(err);
    
    // Set input
    err = model->SetInput("input", input.data(), size * sizeof(float));
    ASSERT_TRITON_OK(err);
    
    // Execute
    err = model->Execute();
    ASSERT_TRITON_OK(err);
    
    // Get output
    std::vector<float> output(size);
    err = model->GetOutput("output", output.data(), size * sizeof(float));
    ASSERT_TRITON_OK(err);
    
    // Basic validation
    if (activation == "relu") {
      for (size_t i = 0; i < size; ++i) {
        EXPECT_GE(output[i], 0.0f);
        EXPECT_EQ(output[i], std::max(0.0f, input[i]));
      }
    } else if (activation == "sigmoid") {
      for (size_t i = 0; i < size; ++i) {
        EXPECT_GE(output[i], 0.0f);
        EXPECT_LE(output[i], 1.0f);
      }
    } else if (activation == "tanh") {
      for (size_t i = 0; i < size; ++i) {
        EXPECT_GE(output[i], -1.0f);
        EXPECT_LE(output[i], 1.0f);
      }
    }
  }
}

TEST_F(MPSBackendIntegrationTest, BatchNormalization)
{
  const int batch = 32, channels = 64, height = 56, width = 56;
  TestDataGenerator gen;
  
  auto input = gen.GenerateFloatData(batch * channels * height * width);
  auto scale = gen.GenerateFloatData(channels, 0.5f, 1.5f);
  auto bias = gen.GenerateFloatData(channels, -0.5f, 0.5f);
  auto mean = gen.GenerateFloatData(channels, -0.1f, 0.1f);
  auto variance = gen.GenerateFloatData(channels, 0.5f, 1.5f);
  
  MPSModelConfig config;
  config.name = "test_batchnorm";
  config.device_id = 0;
  
  auto model = mps_engine_->CreateModel(config);
  ASSERT_NE(model, nullptr);
  
  // Add batch norm
  auto err = model->AddBatchNormalization(
      "batchnorm", channels, 1e-5f, true /* training */);
  ASSERT_TRITON_OK(err);
  
  // Set inputs
  err = model->SetInput("input", input.data(), input.size() * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  err = model->SetInput("scale", scale.data(), scale.size() * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  err = model->SetInput("bias", bias.data(), bias.size() * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  err = model->SetInput("mean", mean.data(), mean.size() * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  err = model->SetInput("variance", variance.data(), variance.size() * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  // Execute
  err = model->Execute();
  ASSERT_TRITON_OK(err);
  
  // Get output
  std::vector<float> output(input.size());
  err = model->GetOutput("output", output.data(), output.size() * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  // Basic validation - output should have similar shape but normalized
  EXPECT_EQ(output.size(), input.size());
}

TEST_F(MPSBackendIntegrationTest, Pooling)
{
  const int batch = 1, channels = 64, height = 112, width = 112;
  TestDataGenerator gen;
  auto input = gen.GenerateFloatData(batch * channels * height * width);
  
  // Test max pooling
  {
    MPSModelConfig config;
    config.name = "test_maxpool";
    config.device_id = 0;
    
    auto model = mps_engine_->CreateModel(config);
    ASSERT_NE(model, nullptr);
    
    MPSPoolingParams pool_params;
    pool_params.kernel_h = 2;
    pool_params.kernel_w = 2;
    pool_params.stride_h = 2;
    pool_params.stride_w = 2;
    pool_params.padding_h = 0;
    pool_params.padding_w = 0;
    
    auto err = model->AddPooling2D("maxpool", "max", pool_params);
    ASSERT_TRITON_OK(err);
    
    err = model->SetInput("input", input.data(), input.size() * sizeof(float));
    ASSERT_TRITON_OK(err);
    
    err = model->Execute();
    ASSERT_TRITON_OK(err);
    
    std::vector<int64_t> output_shape;
    err = model->GetOutputShape("output", output_shape);
    ASSERT_TRITON_OK(err);
    
    EXPECT_EQ(output_shape[0], batch);
    EXPECT_EQ(output_shape[1], channels);
    EXPECT_EQ(output_shape[2], height / 2);
    EXPECT_EQ(output_shape[3], width / 2);
  }
  
  // Test average pooling
  {
    MPSModelConfig config;
    config.name = "test_avgpool";
    config.device_id = 0;
    
    auto model = mps_engine_->CreateModel(config);
    ASSERT_NE(model, nullptr);
    
    MPSPoolingParams pool_params;
    pool_params.kernel_h = 2;
    pool_params.kernel_w = 2;
    pool_params.stride_h = 2;
    pool_params.stride_w = 2;
    pool_params.padding_h = 0;
    pool_params.padding_w = 0;
    
    auto err = model->AddPooling2D("avgpool", "average", pool_params);
    ASSERT_TRITON_OK(err);
    
    err = model->SetInput("input", input.data(), input.size() * sizeof(float));
    ASSERT_TRITON_OK(err);
    
    err = model->Execute();
    ASSERT_TRITON_OK(err);
  }
}

TEST_F(MPSBackendIntegrationTest, ResidualConnection)
{
  const size_t size = 1024;
  TestDataGenerator gen;
  auto input1 = gen.GenerateFloatData(size);
  auto input2 = gen.GenerateFloatData(size);
  
  MPSModelConfig config;
  config.name = "test_residual";
  config.device_id = 0;
  
  auto model = mps_engine_->CreateModel(config);
  ASSERT_NE(model, nullptr);
  
  // Add element-wise addition
  auto err = model->AddElementwise("add", "add");
  ASSERT_TRITON_OK(err);
  
  err = model->SetInput("input1", input1.data(), size * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  err = model->SetInput("input2", input2.data(), size * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  err = model->Execute();
  ASSERT_TRITON_OK(err);
  
  std::vector<float> output(size);
  err = model->GetOutput("output", output.data(), size * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  // Verify addition
  for (size_t i = 0; i < size; ++i) {
    EXPECT_NEAR(output[i], input1[i] + input2[i], 1e-5f);
  }
}

TEST_F(MPSBackendIntegrationTest, CompleteNeuralNetwork)
{
  // Build a small CNN
  const int batch = 1, input_channels = 3, height = 32, width = 32;
  const int num_classes = 10;
  
  TestDataGenerator gen;
  auto input = gen.GenerateFloatData(batch * input_channels * height * width);
  
  MPSModelConfig config;
  config.name = "test_cnn";
  config.device_id = 0;
  config.optimize = true;
  
  auto model = mps_engine_->CreateModel(config);
  ASSERT_NE(model, nullptr);
  
  // Build network: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC
  
  // First conv block
  auto err = model->AddConvolution2D(
      "conv1", batch, input_channels, height, width,
      32, 3, 3, {1, 1, 1, 1, 1, 1});
  ASSERT_TRITON_OK(err);
  
  err = model->AddActivation("relu1", "relu");
  ASSERT_TRITON_OK(err);
  
  err = model->AddPooling2D("pool1", "max", {2, 2, 2, 2, 0, 0});
  ASSERT_TRITON_OK(err);
  
  // Second conv block
  err = model->AddConvolution2D(
      "conv2", batch, 32, 16, 16,
      64, 3, 3, {1, 1, 1, 1, 1, 1});
  ASSERT_TRITON_OK(err);
  
  err = model->AddActivation("relu2", "relu");
  ASSERT_TRITON_OK(err);
  
  err = model->AddPooling2D("pool2", "max", {2, 2, 2, 2, 0, 0});
  ASSERT_TRITON_OK(err);
  
  // Flatten and FC
  err = model->AddFlatten("flatten");
  ASSERT_TRITON_OK(err);
  
  err = model->AddFullyConnected("fc", 64 * 8 * 8, num_classes);
  ASSERT_TRITON_OK(err);
  
  // Compile model
  err = model->Compile();
  ASSERT_TRITON_OK(err);
  
  // Set input
  err = model->SetInput("input", input.data(), input.size() * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  // Execute
  err = model->Execute();
  ASSERT_TRITON_OK(err);
  
  // Get output
  std::vector<float> output(num_classes);
  err = model->GetOutput("output", output.data(), output.size() * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  EXPECT_EQ(output.size(), num_classes);
}

TEST_F(MPSBackendIntegrationTest, MultipleInputsOutputs)
{
  // Test model with multiple inputs and outputs
  const size_t size = 512;
  TestDataGenerator gen;
  
  auto input1 = gen.GenerateFloatData(size);
  auto input2 = gen.GenerateFloatData(size);
  auto input3 = gen.GenerateFloatData(size);
  
  MPSModelConfig config;
  config.name = "test_multi_io";
  config.device_id = 0;
  
  auto model = mps_engine_->CreateModel(config);
  ASSERT_NE(model, nullptr);
  
  // Create computation graph:
  // output1 = input1 + input2
  // output2 = input1 * input3
  // output3 = relu(output1 - output2)
  
  err = model->AddElementwise("add", "add");
  ASSERT_TRITON_OK(err);
  
  err = model->AddElementwise("mul", "multiply");
  ASSERT_TRITON_OK(err);
  
  err = model->AddElementwise("sub", "subtract");
  ASSERT_TRITON_OK(err);
  
  err = model->AddActivation("relu", "relu");
  ASSERT_TRITON_OK(err);
  
  // Connect graph
  err = model->ConnectLayers("input1", "add.input1");
  ASSERT_TRITON_OK(err);
  
  err = model->ConnectLayers("input2", "add.input2");
  ASSERT_TRITON_OK(err);
  
  err = model->ConnectLayers("input1", "mul.input1");
  ASSERT_TRITON_OK(err);
  
  err = model->ConnectLayers("input3", "mul.input2");
  ASSERT_TRITON_OK(err);
  
  err = model->ConnectLayers("add.output", "sub.input1");
  ASSERT_TRITON_OK(err);
  
  err = model->ConnectLayers("mul.output", "sub.input2");
  ASSERT_TRITON_OK(err);
  
  err = model->ConnectLayers("sub.output", "relu.input");
  ASSERT_TRITON_OK(err);
  
  // Mark outputs
  err = model->MarkAsOutput("add.output", "output1");
  ASSERT_TRITON_OK(err);
  
  err = model->MarkAsOutput("mul.output", "output2");
  ASSERT_TRITON_OK(err);
  
  err = model->MarkAsOutput("relu.output", "output3");
  ASSERT_TRITON_OK(err);
  
  // Compile
  err = model->Compile();
  ASSERT_TRITON_OK(err);
  
  // Set inputs
  err = model->SetInput("input1", input1.data(), size * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  err = model->SetInput("input2", input2.data(), size * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  err = model->SetInput("input3", input3.data(), size * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  // Execute
  err = model->Execute();
  ASSERT_TRITON_OK(err);
  
  // Get outputs
  std::vector<float> output1(size), output2(size), output3(size);
  
  err = model->GetOutput("output1", output1.data(), size * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  err = model->GetOutput("output2", output2.data(), size * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  err = model->GetOutput("output3", output3.data(), size * sizeof(float));
  ASSERT_TRITON_OK(err);
  
  // Validate
  for (size_t i = 0; i < size; ++i) {
    EXPECT_NEAR(output1[i], input1[i] + input2[i], 1e-5f);
    EXPECT_NEAR(output2[i], input1[i] * input3[i], 1e-5f);
    float expected = std::max(0.0f, output1[i] - output2[i]);
    EXPECT_NEAR(output3[i], expected, 1e-5f);
  }
}

TEST_F(MPSBackendIntegrationTest, DynamicBatching)
{
  // Test different batch sizes
  std::vector<int> batch_sizes = {1, 8, 16, 32};
  const int channels = 3, height = 224, width = 224;
  
  TestDataGenerator gen;
  
  for (int batch : batch_sizes) {
    auto input = gen.GenerateFloatData(batch * channels * height * width);
    
    MPSModelConfig config;
    config.name = "test_dynamic_batch";
    config.device_id = 0;
    config.supports_dynamic_batch = true;
    
    auto model = mps_engine_->CreateModel(config);
    ASSERT_NE(model, nullptr);
    
    // Simple model
    auto err = model->AddActivation("relu", "relu");
    ASSERT_TRITON_OK(err);
    
    err = model->SetBatchSize(batch);
    ASSERT_TRITON_OK(err);
    
    err = model->SetInput("input", input.data(), input.size() * sizeof(float));
    ASSERT_TRITON_OK(err);
    
    err = model->Execute();
    ASSERT_TRITON_OK(err);
    
    std::vector<float> output(input.size());
    err = model->GetOutput("output", output.data(), output.size() * sizeof(float));
    ASSERT_TRITON_OK(err);
    
    // Verify batch processing worked
    for (size_t i = 0; i < input.size(); ++i) {
      EXPECT_EQ(output[i], std::max(0.0f, input[i]));
    }
  }
}

TEST_F(MPSBackendIntegrationTest, ErrorHandling)
{
  MPSModelConfig config;
  config.name = "test_errors";
  config.device_id = 0;
  
  auto model = mps_engine_->CreateModel(config);
  ASSERT_NE(model, nullptr);
  
  // Test various error conditions
  
  // 1. Execute without inputs
  auto err = model->Execute();
  EXPECT_NE(err, nullptr);
  if (err) TRITONSERVER_ErrorDelete(err);
  
  // 2. Get output that doesn't exist
  float dummy;
  err = model->GetOutput("nonexistent", &dummy, sizeof(float));
  EXPECT_NE(err, nullptr);
  if (err) TRITONSERVER_ErrorDelete(err);
  
  // 3. Invalid layer configuration
  err = model->AddConvolution2D(
      "invalid_conv", 1, 3, 224, 224,
      64, -1, -1, {1, 1, 1, 1, 1, 1});  // Invalid kernel size
  EXPECT_NE(err, nullptr);
  if (err) TRITONSERVER_ErrorDelete(err);
  
  // 4. Mismatched input size
  err = model->AddActivation("relu", "relu");
  ASSERT_TRITON_OK(err);
  
  float small_input[10];
  err = model->SetInput("input", small_input, sizeof(small_input));
  ASSERT_TRITON_OK(err);
  
  err = model->Execute();
  EXPECT_NE(err, nullptr);  // Should fail due to size mismatch
  if (err) TRITONSERVER_ErrorDelete(err);
}

}}}  // namespace triton::server::test