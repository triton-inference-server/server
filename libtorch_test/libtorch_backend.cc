#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include <torch/data.h>
#include <torch/types.h>
#include <c10/util/ArrayRef.h>
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

using namespace torch::data; // NOLINT

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: libtorch_backend <path-to-exported-script-module>\n";
    return -1;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);

  assert(module != nullptr);
  std::cout << "Model loaded successfully\n";

  torch::Tensor input_tensor = torch::rand({1, 3, 224, 224});
  // std::cout << input_tensor << std::endl;
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input_tensor);

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module->forward(inputs).toTensor();

  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  std::vector<double> v = {1.0, 2.0, 3.0};
  auto test_tensor = torch::from_blob(v.data(), v.size(),
    torch::dtype(torch::kFloat64).requires_grad(true));

  // OR
  std::vector<double> w = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0};
  test_tensor = at::tensor(w);

  // 2D (N-D)
  std::vector<int32_t> v = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  };

  auto test_tensor2 = torch::from_blob(v.data(),/*sizes=*/{3, 3},
      /*strides=*/{1, 3}, torch::kInt32);

  std::cout << "datatype of Tensor: "<< test_tensor2.dtype(); // torch::kInt32
  std::cout << "No. of values: "<< test_tensor2.numel(); // 9

  torch::Tensor test_tensor3 = torch::tensor(3.14, torch::kCUDA);
  torch::Scalar scalar = test_tensor3.item()
  std::cout << "Tensor to Scalar" << scalar.to<float>();

  test_tensor = test_tensor.to(at::TensorOptions(at::kDouble));
  test_tensor = test_tensor.to(at::kDouble);

}
