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
#include <condition_variable>
#include <iostream>
#include <string>
#include <vector>

#include "src/clients/c++/library/grpc_client.h"

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

using ResultList = std::vector<std::shared_ptr<nic::InferResult>>;

// Global mutex to synchronize the threads
std::mutex mutex_;
std::condition_variable cv_;

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
  std::cerr << "\t-u <URL for inference service and its gRPC port>"
            << std::endl;
  std::cerr
      << "For -H, header must be 'Header:Value'. May be given multiple times."
      << std::endl;
  std::cerr << "\t-t <stream timeout in microseconds>" << std::endl;
  std::cerr << "\t-o <offset for sequence ID>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "For -o, the client will use sequence ID <1 + 2 * offset> "
            << "and <2 + 2 * offset>. Default offset is 0." << std::endl;

  exit(1);
}

void
StreamSend(
    const std::unique_ptr<nic::InferenceServerGrpcClient>& client,
    const std::string& model_name, int32_t value, const uint64_t sequence_id,
    bool start_of_sequence, bool end_of_sequence, const int32_t index)
{
  nic::InferOptions options(model_name);
  options.sequence_id_ = sequence_id;
  options.sequence_start_ = start_of_sequence;
  options.sequence_end_ = end_of_sequence;
  options.request_id_ =
      std::to_string(sequence_id) + "_" + std::to_string(index);

  // Initialize the inputs with the data.
  nic::InferInput* input;
  std::vector<int64_t> shape{1, 1};
  FAIL_IF_ERR(
      nic::InferInput::Create(&input, "INPUT", shape, "INT32"),
      "unable to create 'INPUT'");
  std::shared_ptr<nic::InferInput> ivalue(input);
  FAIL_IF_ERR(ivalue->Reset(), "unable to reset 'INPUT'");
  FAIL_IF_ERR(
      ivalue->AppendRaw(reinterpret_cast<uint8_t*>(&value), sizeof(int32_t)),
      "unable to set data for 'INPUT'");

  std::vector<nic::InferInput*> inputs = {ivalue.get()};

  // Send inference request to the inference server.
  FAIL_IF_ERR(client->AsyncStreamInfer(options, inputs), "unable to run model");
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  bool dyna_sequence = false;
  std::string url("localhost:8001");
  nic::Headers http_headers;
  int sequence_id_offset = 0;
  uint32_t stream_timeout = 0;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vdu:H:t:o:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'd':
        dyna_sequence = true;
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
      case 't':
        stream_timeout = std::stoi(optarg);
        break;
      case 'o':
        sequence_id_offset = std::stoi(optarg);
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  nic::Error err;

  // We use the custom "sequence" model which takes 1 input value. The
  // output is the accumulated value of the inputs. See
  // src/custom/sequence.
  std::string model_name = "simple_sequence";


  const uint64_t sequence_id0 = 1 + sequence_id_offset * 2;
  const uint64_t sequence_id1 = 2 + sequence_id_offset * 2;
  std::cout << "sequence ID " << sequence_id0 << " : "
            << "sequence ID " << sequence_id1 << std::endl;

  // Create a InferenceServerGrpcClient instance to communicate with the
  // server using gRPC protocol.
  std::unique_ptr<nic::InferenceServerGrpcClient> client;
  FAIL_IF_ERR(
      nic::InferenceServerGrpcClient::Create(&client, url, verbose),
      "unable to create grpc client");

  // Now send the inference sequences..
  //
  std::vector<int32_t> values{11, 7, 5, 3, 2, 0, 1};
  ResultList result_list;

  FAIL_IF_ERR(
      client->StartStream(
          [&](nic::InferResult* result) {
            {
              std::shared_ptr<nic::InferResult> result_ptr(result);
              std::lock_guard<std::mutex> lk(mutex_);
              result_list.push_back(result_ptr);
            }
            cv_.notify_all();
          },
          false /*ship_stats*/, stream_timeout, http_headers),
      "unable to establish a streaming connection to server");

  // Send requests, first reset accumulator for the sequence.
  int32_t index = 0;
  StreamSend(
      client, model_name, 0, sequence_id0, true /* start-of-sequence */,
      false /* end-of-sequence */, index++);
  StreamSend(
      client, model_name, 100, sequence_id1, true /* start-of-sequence */,
      false /* end-of-sequence */, index++);

  // Now send a sequence of values...
  for (int32_t v : values) {
    StreamSend(
        client, model_name, v, sequence_id0, false /* start-of-sequence */,
        (v == 1) /* end-of-sequence */, index++);
    StreamSend(
        client, model_name, -v, sequence_id1, false /* start-of-sequence */,
        (v == 1) /* end-of-sequence */, index++);
  }

  // Wait until all callbacks are invoked
  {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&]() {
      if (result_list.size() > (2 * values.size() + 1)) {
        return true;
      } else {
        return false;
      }
    });
  }

  // Extract data from the result
  std::vector<int32_t> result0_data;
  std::vector<int32_t> result1_data;
  for (const auto& this_result : result_list) {
    auto err = this_result->RequestStatus();
    if (!err.IsOk()) {
      std::cerr << "The inference failed: " << err << std::endl;
      exit(1);
    }
    // Get pointers to the result returned...
    int32_t* output_data;
    size_t output_byte_size;
    FAIL_IF_ERR(
        this_result->RawData(
            "OUTPUT", (const uint8_t**)&output_data, &output_byte_size),
        "unable to get result data for 'OUTPUT'");
    if (output_byte_size != 4) {
      std::cerr << "error: received incorrect byte size for 'OUTPUT': "
                << output_byte_size << std::endl;
      exit(1);
    }

    std::string request_id;
    FAIL_IF_ERR(
        this_result->Id(&request_id), "unable to get request id for response");
    uint64_t this_sequence_id =
        std::stoi(std::string(request_id, 0, request_id.find("_")));
    if (this_sequence_id == sequence_id0) {
      result0_data.push_back(*output_data);
    } else if (this_sequence_id == sequence_id1) {
      result1_data.push_back(*output_data);
    } else {
      std::cerr << "error: received incorrect sequence id in response: "
                << this_sequence_id << std::endl;
      exit(1);
    }
  }


  int32_t seq0_expected = 0;
  int32_t seq1_expected = 100;

  for (size_t i = 0; i < result0_data.size(); i++) {
    std::cout << "[" << i << "] " << result0_data[i] << " : " << result1_data[i]
              << std::endl;

    if ((seq0_expected != result0_data[i]) ||
        (seq1_expected != result1_data[i])) {
      std::cout << "[ expected ] " << seq0_expected << " : " << seq1_expected
                << std::endl;
      return 1;
    }

    if (i < values.size()) {
      seq0_expected += values[i];
      seq1_expected -= values[i];

      // The dyna_sequence custom backend adds the sequence ID to
      // the last request in a sequence.
      if (dyna_sequence && (values[i] == 1)) {
        seq0_expected += sequence_id0;
        seq1_expected += sequence_id1;
      }
    }
  }

  return 0;
}
