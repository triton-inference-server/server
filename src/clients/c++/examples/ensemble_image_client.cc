// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include "src/clients/c++/examples/json_utils.h"
#include "src/clients/c++/library/grpc_client.h"
#include "src/clients/c++/library/http_client.h"

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    nic::Error err = (X);                                          \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

namespace {

void
Postprocess(
    const std::unique_ptr<nic::InferResult> result,
    const std::vector<std::string>& filenames, const size_t batch_size,
    const size_t topk)
{
  std::string output_name("OUTPUT");
  if (!result->RequestStatus().IsOk()) {
    std::cerr << "inference  failed with error: " << result->RequestStatus()
              << std::endl;
    exit(1);
  }
  if (filenames.size() != batch_size) {
    std::cerr << "expected " << batch_size << " filenames, got "
              << filenames.size() << std::endl;
    exit(1);
  }

  // Get and validate the shape and datatype
  std::vector<int64_t> shape;
  nic::Error err = result->Shape(output_name, &shape);
  if (!err.IsOk()) {
    std::cerr << "unable to get shape for " << output_name << std::endl;
    exit(1);
  }
  // Validate shape
  if ((shape.size() != 2) || (shape[0] != (int)batch_size) ||
      (shape[1] != (int)topk)) {
    std::cerr << "received incorrect shapes for " << output_name << std::endl;
    exit(1);
  }
  std::string datatype;
  err = result->Datatype(output_name, &datatype);
  if (!err.IsOk()) {
    std::cerr << "unable to get datatype for " << output_name << std::endl;
    exit(1);
  }
  // Validate datatype
  if (datatype.compare("BYTES") != 0) {
    std::cerr << "received incorrect datatype for " << output_name << ": "
              << datatype << std::endl;
    exit(1);
  }

  std::vector<std::string> result_data;
  err = result->StringData(output_name, &result_data);
  if (!err.IsOk()) {
    std::cerr << "unable to get data for " << output_name << std::endl;
    exit(1);
  }

  if (result_data.size() != (topk * batch_size)) {
    std::cerr << "unexpected number of strings in the result, expected "
              << (topk * batch_size) << ", got " << result_data.size()
              << std::endl;
    exit(1);
  }
  size_t index = 0;
  for (size_t b = 0; b < batch_size; ++b) {
    std::cout << "Image '" << filenames[b] << "':" << std::endl;
    for (size_t c = 0; c < topk; ++c) {
      std::istringstream is(result_data[index]);
      int count = 0;
      std::string token;
      while (getline(is, token, ':')) {
        if (count == 0) {
          std::cout << "    " << token;
        } else if (count == 1) {
          std::cout << " (" << token << ")";
        } else if (count == 2) {
          std::cout << " = " << token;
        }
        count++;
      }
      std::cout << std::endl;
      index++;
    }
  }
}

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0]
            << " [options] <image filename / image folder>" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-c <topk>" << std::endl;
  std::cerr << "\t-i <Protocol used to communicate with inference service>"
            << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "For -c, the <topk> classes will be returned, default is 1."
            << std::endl;
  std::cerr
      << "For -i, available protocols are 'grpc' and 'http'. Default is 'http."
      << std::endl;

  exit(1);
}

union TritonClient {
  TritonClient()
  {
    new (&http_client_) std::unique_ptr<nic::InferenceServerHttpClient>{};
  }
  ~TritonClient() {}

  std::unique_ptr<nic::InferenceServerHttpClient> http_client_;
  std::unique_ptr<nic::InferenceServerGrpcClient> grpc_client_;
};

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  std::string url("localhost:8000");
  std::string protocol = "http";
  size_t topk = 1;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vi:u:p:c:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'i':
        protocol = optarg;
        break;
      case 'u':
        url = optarg;
        break;
      case 'c':
        topk = std::atoi(optarg);
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  if (topk <= 0) {
    Usage(argv, "topk must be > 0");
  }

  // The ensemble model takes 1 input tensor with shape [ 1 ] and STRING
  // data type and returns 1 output tensor as top k (see '-c' flag)
  // classification result of the input.
  std::string model_name = "preprocess_resnet50_ensemble";

  // Create the inference client for the model.
  TritonClient triton_client;
  nic::Error err;
  if (protocol == "http") {
    err = nic::InferenceServerHttpClient::Create(
        &triton_client.http_client_, url, verbose);
  } else {
    err = nic::InferenceServerGrpcClient::Create(
        &triton_client.grpc_client_, url, verbose);
  }
  if (!err.IsOk()) {
    std::cerr << "error: unable to create client for inference: " << err
              << std::endl;
    exit(1);
  }

  if (optind >= argc) {
    Usage(argv, "image file or image folder must be specified");
  }

  if (!err.IsOk()) {
    std::cerr << "error: unable to create inference context: " << err
              << std::endl;
    exit(1);
  }

  // Obtain a list of the image names to be processed
  std::vector<std::string> image_filenames;

  struct stat name_stat;
  if (stat(argv[optind], &name_stat) != 0) {
    std::cerr << "Failed to find '" << std::string(argv[optind])
              << "': " << strerror(errno) << std::endl;
    exit(1);
  }

  if (name_stat.st_mode & S_IFDIR) {
    const std::string dirname = argv[optind];
    DIR* dir_ptr = opendir(dirname.c_str());
    struct dirent* d_ptr;
    while ((d_ptr = readdir(dir_ptr)) != NULL) {
      const std::string filename = d_ptr->d_name;
      if ((filename != ".") && (filename != "..")) {
        image_filenames.push_back(dirname + "/" + filename);
      }
    }
    closedir(dir_ptr);
  } else {
    image_filenames.push_back(argv[optind]);
  }

  // Sort the filenames so that we always visit them in the same order
  // (readdir does not guarantee any particular order).
  std::sort(image_filenames.begin(), image_filenames.end());

  // Read the raw image as string
  std::vector<std::vector<std::string>> images;
  for (const auto& fn : image_filenames) {
    images.emplace_back();
    auto& image_str = images.back();
    std::ifstream file(fn);
    file >> std::noskipws;
    image_str.emplace_back(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    if (image_str.back().empty()) {
      std::cerr << "error: unable to read image file " << fn << std::endl;
      exit(1);
    }
  }

  // this client only send one request for simplicity. So the maximum number
  // of the images to be processed is limited by the maximum batch size
  size_t batch_size = 0;
  if (protocol == "http") {
    std::string model_config;
    err = triton_client.http_client_->ModelConfig(&model_config, model_name);
    if (!err.IsOk()) {
      std::cerr << "error: failed to get model config: " << err << std::endl;
    }

    rapidjson::Document model_config_json;
    err = nic::ParseJson(&model_config_json, model_config);
    if (!err.IsOk()) {
      std::cerr << "error: failed to parse model config: " << err << std::endl;
    }

    const auto bs_itr = model_config_json.FindMember("max_batch_size");
    if (bs_itr != model_config_json.MemberEnd()) {
      batch_size = bs_itr->value.GetInt();
    }
  } else {
    inference::ModelConfigResponse model_config;
    err = triton_client.grpc_client_->ModelConfig(&model_config, model_name);
    if (!err.IsOk()) {
      std::cerr << "error: failed to get model config: " << err << std::endl;
    }
    batch_size = model_config.config().max_batch_size();
  }

  if (images.size() > batch_size) {
    std::cerr << "The number of images exceeds maximum batch size, only the"
              << " first " << batch_size << " images, sorted by name"
              << " alphabetically, will be processed" << std::endl;
  }
  batch_size = (images.size() < batch_size) ? images.size() : batch_size;

  // Initialize the inputs with the data.
  nic::InferInput* input;
  std::vector<int64_t> shape{(int64_t)batch_size, 1};
  err = nic::InferInput::Create(&input, "INPUT", shape, "BYTES");
  if (!err.IsOk()) {
    std::cerr << "unable to get input: " << err << std::endl;
    exit(1);
  }
  std::shared_ptr<nic::InferInput> input_ptr(input);


  nic::InferRequestedOutput* output;
  // Set the number of classification expected
  err = nic::InferRequestedOutput::Create(&output, "OUTPUT", topk);
  if (!err.IsOk()) {
    std::cerr << "unable to get output: " << err << std::endl;
    exit(1);
  }
  std::shared_ptr<nic::InferRequestedOutput> output_ptr(output);

  std::vector<nic::InferInput*> inputs = {input_ptr.get()};
  std::vector<const nic::InferRequestedOutput*> outputs = {output_ptr.get()};

  nic::InferOptions options(model_name);

  FAIL_IF_ERR(input_ptr->Reset(), "unable to reset INPUT");

  for (size_t i = 0; i < batch_size; i++) {
    FAIL_IF_ERR(
        input_ptr->AppendFromString(images[i]), "unable to set data for INPUT");
  }

  // Send inference request to the inference server.
  nic::InferResult* results;
  if (protocol == "http") {
    FAIL_IF_ERR(
        triton_client.http_client_->Infer(&results, options, inputs, outputs),
        "unable to run model");
  } else {
    FAIL_IF_ERR(
        triton_client.grpc_client_->Infer(&results, options, inputs, outputs),
        "unable to run model");
  }
  std::unique_ptr<nic::InferResult> results_ptr;
  results_ptr.reset(results);

  // Print classification results
  Postprocess(std::move(results_ptr), image_filenames, batch_size, topk);

  return 0;
}
