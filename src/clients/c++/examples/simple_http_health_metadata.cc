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

#include <unistd.h>
#include <iostream>
#include <string>
#include "src/clients/c++/examples/json_utils.h"
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
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-H <HTTP header>" << std::endl;
  std::cerr << std::endl;
  std::cerr
      << "For -H, header must be 'Header:Value'. May be given multiple times."
      << std::endl;

  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  std::string url("localhost:8000");
  nic::Headers http_headers;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vu:H:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'u':
        url = optarg;
        break;
      case 'H': {
        std::string arg = optarg;
        std::string header = arg.substr(0, arg.find(":"));
        http_headers[header] = arg.substr(header.size() + 1);
        break;
      }
      case '?':
        Usage(argv);
        break;
    }
  }

  // We use a simple model that takes 2 input tensors of 16 integers
  // each and returns 2 output tensors of 16 integers each. One output
  // tensor is the element-wise sum of the inputs and one output is
  // the element-wise difference.
  std::string model_name = "simple";
  std::string model_version = "";

  // Create a InferenceServerHttpClient instance to communicate with the
  // server using http protocol.
  std::unique_ptr<nic::InferenceServerHttpClient> client;
  FAIL_IF_ERR(
      nic::InferenceServerHttpClient::Create(&client, url, verbose),
      "unable to create http client");

  bool live;
  FAIL_IF_ERR(
      client->IsServerLive(&live, http_headers),
      "unable to get server liveness");
  if (!live) {
    std::cerr << "error: server is not live" << std::endl;
    exit(1);
  }

  bool ready;
  FAIL_IF_ERR(
      client->IsServerReady(&ready, http_headers), "server is not live");

  bool model_ready;
  FAIL_IF_ERR(
      client->IsModelReady(
          &model_ready, model_name, model_version, http_headers),
      "unable to get model readiness");
  if (!model_ready) {
    std::cerr << "error: model " << model_name << " is not live" << std::endl;
    exit(1);
  }

  {
    std::string server_metadata;
    FAIL_IF_ERR(
        client->ServerMetadata(&server_metadata, http_headers),
        "unable to get server metadata");

    rapidjson::Document server_metadata_json;
    FAIL_IF_ERR(
        nic::ParseJson(&server_metadata_json, server_metadata),
        "failed to parse server metadata");
    if ((std::string(server_metadata_json["name"].GetString()))
            .compare("triton") != 0) {
      std::cerr << "error: unexpected server metadata: " << server_metadata
                << std::endl;
      exit(1);
    }
  }

  {
    std::string model_metadata;
    FAIL_IF_ERR(
        client->ModelMetadata(
            &model_metadata, model_name, model_version, http_headers),
        "unable to get model metadata");

    rapidjson::Document model_metadata_json;
    FAIL_IF_ERR(
        nic::ParseJson(&model_metadata_json, model_metadata),
        "failed to parse model metadata");
    if ((std::string(model_metadata_json["name"].GetString()))
            .compare(model_name) != 0) {
      std::cerr << "error: unexpected model metadata: " << model_metadata
                << std::endl;
      exit(1);
    }
  }

  {
    std::string model_config;
    FAIL_IF_ERR(
        client->ModelConfig(
            &model_config, model_name, model_version, http_headers),
        "unable to get model config");

    rapidjson::Document model_config_json;
    FAIL_IF_ERR(
        nic::ParseJson(&model_config_json, model_config),
        "failed to parse model config");
    if ((std::string(model_config_json["name"].GetString()))
            .compare(model_name) != 0) {
      std::cerr << "error: unexpected model config: " << model_config
                << std::endl;
      exit(1);
    }
  }

  {
    std::string model_metadata;
    nic::Error err = client->ModelMetadata(
        &model_metadata, "wrong_model_name", model_version, http_headers);
    if (err.IsOk()) {
      std::cerr << "error: expected an error but got: " << err << std::endl;
      exit(1);
    }
  }

  std::cout << "SUCCESS" << std::endl;

  return 0;
}
