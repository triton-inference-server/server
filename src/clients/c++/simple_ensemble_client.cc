// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/clients/c++/request.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

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
    const std::map<std::string, std::unique_ptr<nic::InferContext::Result>>&
        results,
    const std::vector<std::string>& filenames, const size_t batch_size)
{
  if (results.size() != 1) {
    std::cerr << "expected 1 result, got " << results.size() << std::endl;
    exit(1);
  }

  const std::unique_ptr<nic::InferContext::Result>& result =
      results.begin()->second;

  for (size_t b = 0; b < batch_size; ++b) {
    size_t cnt = 0;
    nic::Error err = result->GetClassCount(b, &cnt);
    if (!err.IsOk()) {
      std::cerr << "failed reading class count for batch " << b << ": " << err
                << std::endl;
      exit(1);
    }

    std::cout << "Image '" << filenames[b] << "':" << std::endl;

    for (size_t c = 0; c < cnt; ++c) {
      nic::InferContext::Result::ClassResult cls;
      nic::Error err = result->GetClassAtCursor(b, &cls);
      if (!err.IsOk()) {
        std::cerr << "failed reading class for batch " << b << ": " << err
                  << std::endl;
        exit(1);
      }

      std::cout << "    " << cls.idx << " (" << cls.label << ") = " << cls.value
                << std::endl;
    }
  }
}

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options] <image filename / image folder>" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-i <Protocol used to communicate with inference service>"
            << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-p <proprocessed output filename prefix>" << std::endl;
  std::cerr << std::endl;
  std::cerr
      << "For -i, available protocols are 'grpc' and 'http'. Default is 'http."
      << std::endl;
  std::cerr
      << "For -p, if specified, the client will only send the raw images to the"
      << " preprocess custom backend, and write the response to files as output"
      << " in <prfix><batch index> format."
      << std::endl;

  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  std::string url("localhost:8000");
  std::string protocol = "http";
  std::string prefix = "";

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vi:u:p:")) != -1) {
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
      case 'p':
        prefix = optarg;
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  nic::Error err;

  // We use a simple model that takes 2 input tensors of 16 strings
  // each and returns 2 output tensors of 16 strings each. The input
  // strings must represent integers. One output tensor is the
  // element-wise sum of the inputs and one output is the element-wise
  // difference.
  std::string model_name = "image_preprocess";
  if (prefix.empty()) {
    // [TODO] set model name to ensemble model once ensemble is implemented
    Usage(argv, "error: attempt to use unimplemented ensemble feature");
  }

  // Create the inference context for the model.
  std::unique_ptr<nic::InferContext> ctx;
  if (protocol == "http") {
    err = nic::InferHttpContext::Create(
        &ctx, url, model_name, -1 /* model_version */, verbose);
  } else if (protocol == "grpc") {
    err = nic::InferGrpcContext::Create(
        &ctx, url, model_name, -1 /* model_version */, verbose);
  } else {
    Usage(argv, "unknown protocol '" + protocol + "'");
  }
  if (optind >= argc) {
    Usage(argv, "image file or image folder must be specified");
  }

  if (!err.IsOk()) {
    std::cerr << "error: unable to create inference context: " << err
              << std::endl;
    exit(1);
  }

  // Create the data for the two input tensors. Initialize the first
  // to unique integers and the second to all ones. The input tensors
  // are the string representation of these values.
  // Collect the names of the image(s).
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
    image_str.emplace_back((std::istreambuf_iterator<char>(file)),
                  std::istreambuf_iterator<char>());
    if (image_str.back().empty()) {
      std::cerr << "error: unable to read image file " << fn << std::endl;
      exit(1);
    }
  }

  // only send one request for simplicity
  uint64_t batch_size = ctx->MaxBatchSize();
  batch_size = (images.size() < batch_size) ? images.size() : batch_size;

  // Set the context options to do batch-size 1 requests. Also request
  // that all output tensors be returned.
  std::unique_ptr<nic::InferContext::Options> options;
  FAIL_IF_ERR(
      nic::InferContext::Options::Create(&options),
      "unable to create inference options");

  options->SetBatchSize(batch_size);
  for (const auto& output : ctx->Outputs()) {
    if (prefix.empty()) {
      options->AddClassResult(output, 1);
    } else {
      options->AddRawResult(output);
    }
  }

  FAIL_IF_ERR(ctx->SetRunOptions(*options), "unable to set inference options");

  // Initialize the inputs with the data.
  std::shared_ptr<nic::InferContext::Input> input;
  FAIL_IF_ERR(ctx->GetInput("INPUT", &input), "unable to get INPUT");

  FAIL_IF_ERR(input->Reset(), "unable to reset INPUT");

  for (size_t i = 0; i < batch_size; i++) {
    FAIL_IF_ERR(
        input->SetFromString(images[i]), "unable to set data for INPUT");
  }

  // Send inference request to the inference server.
  std::map<std::string, std::unique_ptr<nic::InferContext::Result>> results;
  FAIL_IF_ERR(ctx->Run(&results), "unable to run model");

  // We expect there to be 2 results. Walk over all 16 result elements
  // and print the sum and difference calculated by the model.
  if (results.size() != 1) {
    std::cerr << "error: expected 1 results, got " << results.size()
              << std::endl;
  }

  // Read results
  if (prefix.empty()) {
    for (size_t i = 0; i < batch_size; i++) {
      // Read the output values (they are strings).
      Postprocess(results, image_filenames, batch_size);
    }
  } else {
    for (size_t i = 0; i < batch_size; i++) {
      const std::vector<uint8_t>* buf;
      FAIL_IF_ERR(
          results["OUTPUT"]->GetRaw(i /* batch idx */, &buf),
          "unable to get OUTPUT result at idx " + std::to_string(i));
      std::cout << "batch idx " << i << ": " << image_filenames[i] << std::endl;
      std::ofstream output_file(prefix + std::to_string(i));
      std::ostream_iterator<uint8_t> output_iterator(output_file);
      std::copy((*buf).begin(), (*buf).end(), output_iterator);
    }
  }

  return 0;
}
