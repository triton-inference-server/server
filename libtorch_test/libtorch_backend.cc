#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include <torch/data.h>
#include <torch/types.h>

#include <algorithm>
#include <future>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#include <memory>
#include <c10/util/ArrayRef.h>

using namespace torch::data; // NOLINT

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: libtorch_backend <path-to-exported-script-module>" << std::endl;
    return -1;
  }

  // Load model
  torch::Device device(torch::kCUDA, /*device id=*/0); // load on GPU
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1], device);
  assert(module != nullptr);
  std::cout << "Model loaded successfully" << std::endl;

  // Write vector to binary (file)
  std::vector<float> input_vector = {1.0, 2.0, 3.0, 4.0};
  std::ofstream writeFile;
  writeFile.open("Mesh.bin", std::ios::out | std::ios::binary);
  writeFile.write((char*)&input_vector[0], input_vector.size() * sizeof(float));
  writeFile.close();
  // Read file from binary and create tensor
  std::ifstream stream("Mesh.bin", std::ios::in | std::ios::binary);
  std::vector<char> contents((std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
  // std::cout<<contents.size()<<"\n";
  // Shape of tensor (since it is stored sequentially)
  std::vector<int64_t> tensor_shape{2,2};
  torch::Tensor test_tensor = torch::from_blob(contents.data(), tensor_shape,
      torch::kFloat32);
  std::cout << test_tensor << std::endl;

  // Create a vector of inputs
  std::vector<torch::jit::IValue> inputs;
  int32_t batch_size = 2;
  torch::Tensor input_tensor = torch::rand({1*batch_size, 3, 224, 224}, device);
  inputs.push_back(input_tensor);
  // Execute the model and store its outputs into a tensor.
  torch::Tensor outputs = module->forward(inputs).toTensor();
  outputs.to(torch::kCPU);
  // torch::Tensor output_mini = outputs.slice(/*dim=*/1, /*start=*/0, /*end=*/5);
  // std::cout << output_mini << "\n";
  auto shape = outputs.sizes();
  std::cout<< shape << "\n";

  int64_t output_size = 1;
  for (auto itr = shape.begin(); itr != shape.end(); itr++){
    output_size *= *itr;
  }
  torch::Tensor output_flat = outputs.flatten();
  std::vector<float> outputs_vector;//[output_size];
  char* content[output_flat.nbytes()];
  for(int i=0;i<output_flat.sizes()[0];i++){
    outputs_vector.push_back(output_flat[i].item().to<float>());
  }
  std::cout<< outputs_vector.size() << std::endl;
  // Copy output into buffer
  memcpy(content, &outputs_vector[0], output_flat.nbytes()); // outputs_vector.size() * sizeof(float)
  // Test for Success
  torch::Tensor reconst_output = torch::from_blob(content, {2, 1000}, torch::kFloat32);
  if (reconst_output[0][0].item().to<float>() == output_flat[0].item().to<float>()){
      std::cout<< "Works!" << std::endl;
  }

  // std::vector<double> v = {1.0, 2.0, 3.0};
  // torch::Tensor test_tensor = torch::from_blob(v.data(), v.size(),
  //   torch::dtype(torch::kFloat64).requires_grad(false));
  //
  // // OR
  // std::vector<double> w = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0};
  // test_tensor = torch::Tensor(w);
  //
  // // 2D (N-D)
  // std::vector<int32_t> v2d = {
  //   1, 2, 3,
  //   4, 5, 6,
  //   7, 8, 9
  // };
  //
  // torch::Tensor test_tensor2 = torch::from_blob(v2d.data(),/*sizes=*/{3, 3},
  //     /*strides=*/{1, 3}, torch::kInt32);
  //
  // std::cout << "Datatype of Tensor: "<< test_tensor2.dtype(); // torch::kInt32
  // std::cout << "\nNo. of values: "<< test_tensor2.numel(); // 9
  //
  // torch::Tensor test_tensor3 = torch::tensor(3.14, torch::kCUDA);
  // torch::Scalar scalar = test_tensor3.item();
  // std::cout << "\nTensor to Scalar: " << scalar.to<float>() << "\n";
  //
  // // std::cout << test_tensor2.slice(1, 0, 9); //data<int>();
  // std::cout << test_tensor2.toType(at::kInt)[1][1].item().to<int>();
  //
  // std::cout << "\n";
  // test_tensor = test_tensor.to(torch::TensorOptions(at::kDouble));
  // test_tensor = test_tensor.to(at::kDouble);
}
