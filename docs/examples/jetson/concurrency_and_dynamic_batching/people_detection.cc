// Copyright (c) 2021, NVIDIA CORPORATION& AFFILIATES.All rights reserved.
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

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "triton/core/tritonserver.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace {

bool enforce_memory_type = false;
TRITONSERVER_MemoryType requested_memory_type;

#ifdef TRITON_ENABLE_GPU
static auto cuda_data_deleter = [](void* data) {
  if (data != nullptr) {
    cudaPointerAttributes attr;
    auto cuerr = cudaPointerGetAttributes(&attr, data);
    if (cuerr != cudaSuccess) {
      std::cerr << "error: failed to get CUDA pointer attribute of " << data
                << ": " << cudaGetErrorString(cuerr) << std::endl;
    }
    if (attr.type == cudaMemoryTypeDevice) {
      cuerr = cudaFree(data);
    } else if (attr.type == cudaMemoryTypeHost) {
      cuerr = cudaFreeHost(data);
    }
    if (cuerr != cudaSuccess) {
      std::cerr << "error: failed to release CUDA pointer " << data << ": "
                << cudaGetErrorString(cuerr) << std::endl;
    }
  }
};
#endif  // TRITON_ENABLE_GPU

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-m <\"system\"|\"pinned\"|gpu>"
            << " Enforce the memory type for input and output tensors."
            << " If not specified, inputs will be in system memory and outputs"
            << " will be based on the model's preferred type." << std::endl;
  std::cerr << "\t-v Enable verbose logging." << std::endl;
  std::cerr
      << "\t-t Thread count to simulate the number of concurrent requests."
      << std::endl;
  std::cerr << "\t-r [model repository absolute path]." << std::endl;
  std::cerr << "\t-p [tritonserver path]." << std::endl;
  std::cerr << "\t-s <true|false>."
            << " Specify whether output visualizations will be saved to the "
               "project folder."
            << " If not specified, no outputs will be saved." << std::endl;

  exit(1);
}

TRITONSERVER_Error*
ResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  // Initially attempt to make the actual memory type and id that we
  // allocate be the same as preferred memory type
  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;

  // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
  // need to do any other book-keeping.
  if (byte_size == 0) {
    *buffer = nullptr;
    *buffer_userp = nullptr;
    std::cout << "allocated " << byte_size << " bytes for result tensor "
              << tensor_name << std::endl;
  } else {
    void* allocated_ptr = nullptr;
    if (enforce_memory_type) {
      *actual_memory_type = requested_memory_type;
    }

    switch (*actual_memory_type) {
#ifdef TRITON_ENABLE_GPU
      case TRITONSERVER_MEMORY_CPU_PINNED: {
        auto err = cudaSetDevice(*actual_memory_type_id);
        if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
            (err != cudaErrorInsufficientDriver)) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "unable to recover current CUDA device: " +
                  std::string(cudaGetErrorString(err)))
                  .c_str());
        }

        err = cudaHostAlloc(&allocated_ptr, byte_size, cudaHostAllocPortable);
        if (err != cudaSuccess) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "cudaHostAlloc failed: " +
                  std::string(cudaGetErrorString(err)))
                  .c_str());
        }
        break;
      }

      case TRITONSERVER_MEMORY_GPU: {
        auto err = cudaSetDevice(*actual_memory_type_id);
        if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
            (err != cudaErrorInsufficientDriver)) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "unable to recover current CUDA device: " +
                  std::string(cudaGetErrorString(err)))
                  .c_str());
        }

        err = cudaMalloc(&allocated_ptr, byte_size);
        if (err != cudaSuccess) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "cudaMalloc failed: " + std::string(cudaGetErrorString(err)))
                  .c_str());
        }
        break;
      }
#endif  // TRITON_ENABLE_GPU

      // Use CPU memory if the requested memory type is unknown
      // (default case).
      case TRITONSERVER_MEMORY_CPU:
      default: {
        *actual_memory_type = TRITONSERVER_MEMORY_CPU;
        allocated_ptr = malloc(byte_size);
        break;
      }
    }

    // Pass the tensor name with buffer_userp so we can show it when
    // releasing the buffer.
    if (allocated_ptr != nullptr) {
      *buffer = allocated_ptr;
      *buffer_userp = new std::string(tensor_name);
      std::cout << "allocated " << byte_size << " bytes in "
                << TRITONSERVER_MemoryTypeString(*actual_memory_type)
                << " for result tensor " << tensor_name << std::endl;
    }
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
ResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  std::string* name = nullptr;
  if (buffer_userp != nullptr) {
    name = reinterpret_cast<std::string*>(buffer_userp);
  } else {
    name = new std::string("<unknown>");
  }

  std::cout << "Releasing buffer " << buffer << " of size " << byte_size
            << " in " << TRITONSERVER_MemoryTypeString(memory_type)
            << " for result '" << *name << "'" << std::endl;
  switch (memory_type) {
    case TRITONSERVER_MEMORY_CPU:
      free(buffer);
      break;
#ifdef TRITON_ENABLE_GPU
    case TRITONSERVER_MEMORY_CPU_PINNED: {
      auto err = cudaSetDevice(memory_type_id);
      if (err == cudaSuccess) {
        err = cudaFreeHost(buffer);
      }
      if (err != cudaSuccess) {
        std::cerr << "error: failed to cudaFree " << buffer << ": "
                  << cudaGetErrorString(err) << std::endl;
      }
      break;
    }
    case TRITONSERVER_MEMORY_GPU: {
      auto err = cudaSetDevice(memory_type_id);
      if (err == cudaSuccess) {
        err = cudaFree(buffer);
      }
      if (err != cudaSuccess) {
        std::cerr << "error: failed to cudaFree " << buffer << ": "
                  << cudaGetErrorString(err) << std::endl;
      }
      break;
    }
#endif  // TRITON_ENABLE_GPU
    default:
      std::cerr << "error: unexpected buffer allocated in CUDA managed memory"
                << std::endl;
      break;
  }

  delete name;

  return nullptr;  // Success
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  // We reuse the request so we don't delete it here.
}

void
InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  if (response != nullptr) {
    // Send 'response' to the future.
    std::promise<TRITONSERVER_InferenceResponse*>* p =
        reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
    p->set_value(response);
    delete p;
  }
}


TRITONSERVER_Error*
ParseModelMetadata(const rapidjson::Document& model_metadata)
{
  std::string seen_data_type;
  for (const auto& input : model_metadata["inputs"].GetArray()) {
    if (strcmp(input["datatype"].GetString(), "FP32")) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "this example only supports model with data type FP32");
    }
    if (seen_data_type.empty()) {
      seen_data_type = input["datatype"].GetString();
    } else if (strcmp(seen_data_type.c_str(), input["datatype"].GetString())) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "the inputs and outputs of this model must have the data type");
    }
  }
  for (const auto& output : model_metadata["outputs"].GetArray()) {
    if (strcmp(output["datatype"].GetString(), "FP32")) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "this example only supports model with data type FP32");
    } else if (strcmp(seen_data_type.c_str(), output["datatype"].GetString())) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "the inputs and outputs of this model must have the data type");
    }
  }

  return nullptr;
}


cv::Mat
ResizeKeepAspectRatio(
    const cv::Mat& input, const cv::Size& dstSize, const cv::Scalar& bgcolor,
    bool& fixHeight, float& ratio, int& sideCache)
{
  cv::Mat output;

  double h1 = dstSize.width * (input.rows / (double)input.cols);
  double w2 = dstSize.height * (input.cols / (double)input.rows);
  if (h1 <= dstSize.height) {
    cv::resize(input, output, cv::Size(dstSize.width, h1));
    ratio = (float)dstSize.width / input.cols;
    fixHeight = false;
    sideCache = (int)(ratio * input.rows);
    std::cout << "Resizing to fixed width. Ratio " << ratio << std::endl;
    std::cout << "Height cache " << sideCache << std::endl;
  } else {
    cv::resize(input, output, cv::Size(w2, dstSize.height));
    ratio = (float)dstSize.height / input.rows;
    fixHeight = true;
    sideCache = (int)(ratio * input.cols);
    std::cout << "Resizing to fixed height. Ratio " << ratio << std::endl;
    std::cout << "Width cache " << sideCache << std::endl;
  }

  int top = (dstSize.height - output.rows) / 2;
  int down = (dstSize.height - output.rows + 1) / 2;
  int left = (dstSize.width - output.cols) / 2;
  int right = (dstSize.width - output.cols + 1) / 2;

  cv::copyMakeBorder(
      output, output, top, down, left, right, cv::BORDER_CONSTANT, bgcolor);

  return output;
}


void
SaveOverlay(
    std::vector<cv::Rect>& bboxes_list, std::vector<int>& indexes,
    std::vector<int64_t>& input0_shape, bool& fixHeight, float& ratio,
    int& sideCache, std::string imageName, size_t& thread_id)
{
  const int inputC = input0_shape[1];
  const int inputH = input0_shape[2];
  const int inputW = input0_shape[3];

  cv::Mat image = cv::imread(imageName);

  cv::Scalar color = cv::Scalar(0, 255, 0);

  int xmin, ymin, xmax, ymax;

  for (auto i : indexes) {
    xmin = bboxes_list[i].x;
    ymin = bboxes_list[i].y;
    xmax = bboxes_list[i].x + bboxes_list[i].width;
    ymax = bboxes_list[i].y + bboxes_list[i].height;

    if (fixHeight) {
      xmin = int((xmin - (inputW - sideCache) / 2) / ratio);
      xmax = int((xmax - (inputW - sideCache) / 2) / ratio);
      ymin = int(ymin / ratio);
      ymax = int(ymax / ratio);
    } else {
      ymin = int((ymin - (inputH - sideCache) / 2) / ratio);
      ymax = int((ymax - (inputH - sideCache) / 2) / ratio);
      xmin = int(xmin / ratio);
      xmax = int(xmax / ratio);
    }
    cv::Point p1(xmin, ymin);
    cv::Point p2(xmax, ymax);
    cv::rectangle(image, p1, p2, color, 4);
  }

  std::string outName = "capture_overlay_" + std::to_string(thread_id) + ".jpg";
  imwrite(outName, image);
}


void
Normalize(cv::Mat img, std::vector<float>*& data, int inputC)
{
  for (int c = 0; c < inputC; ++c) {
    for (int i = 0; i < img.rows; ++i) {
      cv::Vec3b* p1 = img.ptr<cv::Vec3b>(i);
      for (int j = 0; j < img.cols; ++j) {
        ((float*)data->data())[c * img.cols * img.rows + i * img.cols + j] =
            p1[j][c] / 255.f;
      }
    }
  }
}


void
RecoverBoundingBoxes(
    std::unordered_map<std::string, std::vector<float>>& output_data,
    std::unordered_map<std::string, const int64_t*>& shapes,
    std::vector<int64_t>& input0_shape, std::vector<cv::Rect>& bboxes_list,
    std::vector<float>& scores_list, std::vector<int>& indexes)
{
  const float box_scale = 35.f;
  const float box_offset = 0.5f;
  const float score_threshold = 0.5f;
  const float nms_threshold = 0.5f;

  int gridH = shapes["output_cov/Sigmoid"][2];
  int gridW = shapes["output_cov/Sigmoid"][3];

  std::cout << "gridH: " << gridH << std::endl;
  std::cout << "gridW: " << gridW << std::endl;

  int modelH = input0_shape[2];
  int modelW = input0_shape[3];
  int batch = input0_shape[0];

  std::cout << "batch: " << batch << std::endl;
  std::cout << "modelH: " << modelH << std::endl;
  std::cout << "modelW: " << modelW << std::endl;

  int cellH = modelH / gridH;
  int cellW = modelW / gridW;

  for (int b = 0; b < batch; b++) {
    for (int h = 0; h < gridH; h++) {
      for (int w = 0; w < gridW; w++) {
        // value(n, c, h, w) = n * CHW + c * HW + h * W + w
        int idx = b * gridH * gridW + h * gridW + w;
        float val = output_data["output_cov/Sigmoid"][idx];
        if (val > score_threshold) {
          scores_list.push_back(val);

          // location of the w, h coordinate in the original image
          int mx = w * cellW;
          int my = h * cellH;

          // scale the detected coordinates to original and return their
          // location in the image
          int idxX1 = b * 3 * gridH * gridW + 0 * gridH * gridW + h * gridW + w;
          int idxY1 = b * 3 * gridH * gridW + 1 * gridH * gridW + h * gridW + w;
          int idxX2 = b * 3 * gridH * gridW + 2 * gridH * gridW + h * gridW + w;
          int idxY2 = b * 3 * gridH * gridW + 3 * gridH * gridW + h * gridW + w;

          int rectX1 =
              -(output_data["output_bbox/BiasAdd"][idxX1] + box_offset) *
                  box_scale +
              mx;
          int rectY1 =
              -(output_data["output_bbox/BiasAdd"][idxY1] + box_offset) *
                  box_scale +
              my;
          int rectX2 =
              (output_data["output_bbox/BiasAdd"][idxX2] + box_offset) *
                  box_scale +
              mx;
          int rectY2 =
              (output_data["output_bbox/BiasAdd"][idxY2] + box_offset) *
                  box_scale +
              my;

          // Rect ROI (x, y, width, height);
          cv::Rect bbox(rectX1, rectY1, rectX2 - rectX1, rectY2 - rectY1);
          bboxes_list.push_back(bbox);
        }
      }
    }
  }

  // Execute non-maximum suppression
  cv::dnn::NMSBoxes(
      bboxes_list, scores_list, score_threshold, nms_threshold, indexes);
}

void
ParseDetections(
    TRITONSERVER_InferenceResponse* response, const std::string& output0,
    const std::string& output1,
    std::unordered_map<std::string, std::vector<float>>& output_data,
    std::unordered_map<std::string, const int64_t*>& shapes)
{
  uint32_t output_count;
  FAIL_IF_ERR(
      TRITONSERVER_InferenceResponseOutputCount(response, &output_count),
      "getting number of response outputs");
  if (output_count != 2) {
    FAIL("expecting 2 response outputs, got " + std::to_string(output_count));
  }

  for (uint32_t idx = 0; idx < output_count; ++idx) {
    const char* cname;
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint64_t dim_count;
    const void* base;
    size_t byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    void* userp;

    FAIL_IF_ERR(
        TRITONSERVER_InferenceResponseOutput(
            response, idx, &cname, &datatype, &shape, &dim_count, &base,
            &byte_size, &memory_type, &memory_type_id, &userp),
        "getting output info");

    if (cname == nullptr) {
      FAIL("unable to get output name");
    }

    std::string name(cname);
    if ((name != output0) && (name != output1)) {
      FAIL("unexpected output '" + name + "'");
    }

    shapes[name] = shape;

    std::vector<float>& odata = output_data[name];

    switch (memory_type) {
      case TRITONSERVER_MEMORY_CPU: {
        std::cout << std::endl
                  << name << " is stored in system memory" << std::endl;
        const float* cbase = reinterpret_cast<const float*>(base);
        odata.assign(cbase, cbase + byte_size / sizeof(float));
        break;
      }

      case TRITONSERVER_MEMORY_CPU_PINNED: {
        std::cout << std::endl
                  << name << " is stored in pinned memory" << std::endl;
        const float* cbase = reinterpret_cast<const float*>(base);
        odata.assign(cbase, cbase + byte_size / sizeof(float));
        break;
      }

#ifdef TRITON_ENABLE_GPU
      case TRITONSERVER_MEMORY_GPU: {
        std::cout << std::endl
                  << name << " is stored in GPU memory" << std::endl;
        odata.reserve(byte_size);
        FAIL_IF_CUDA_ERR(
            cudaMemcpy(&odata[0], base, byte_size, cudaMemcpyDeviceToHost),
            "getting " + name + " data from GPU memory");
        break;
      }
#endif

      default:
        FAIL("unexpected memory type");
    }
  }
}

void
DetectionInferenceOutput(
    std::vector<int>& result_indexes, std::vector<cv::Rect>& bboxes_list,
    TRITONSERVER_InferenceResponse* completed_response,
    const std::string& output0, const std::string& output1,
    std::vector<int64_t>& input0_shape, bool& fixHeight, float& ratio,
    int& sideCache, size_t& thread_id, bool visualize = false,
    std::string imageName = "capture.jpg")
{
  // Parse outputs
  std::unordered_map<std::string, std::vector<float>> output_data;
  std::unordered_map<std::string, const int64_t*> shapes;
  ParseDetections(completed_response, output0, output1, output_data, shapes);

  std::vector<float> scores_list;
  RecoverBoundingBoxes(
      output_data, shapes, input0_shape, bboxes_list, scores_list,
      result_indexes);

  std::cout << "Detection finished. Indexes of detected objects: " << std::endl;
  for (auto idx : result_indexes) {
    std::cout << idx << std::endl;
    std::cout << bboxes_list[idx] << std::endl;
  }

  if (visualize)
    SaveOverlay(
        bboxes_list, result_indexes, input0_shape, fixHeight, ratio, sideCache,
        imageName, thread_id);
}


}  // namespace


void
SetServerOptions(
    TRITONSERVER_ServerOptions** server_options, bool verbose_level,
    std::string model_repository_path, std::string tritonserver_path)
{
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsNew(server_options), "creating server options");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetModelRepositoryPath(
          *server_options, model_repository_path.c_str()),
      "setting model repository path");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetLogVerbose(*server_options, verbose_level),
      "setting verbose logging level");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetMetrics(*server_options, true),
      "failed to enable metrics");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetStrictReadiness(*server_options, true),
      "failed to set strict readiness");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetStrictModelConfig(*server_options, true),
      "failed to set strict model config");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetModelControlMode(
          *server_options, TRITONSERVER_MODEL_CONTROL_EXPLICIT),
      "failed to set model control mode to explicit");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetBackendDirectory(
          *server_options, (tritonserver_path + "/backends").c_str()),
      "setting backend directory");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
          *server_options, (tritonserver_path + "/repoagents").c_str()),
      "setting repository agent directory");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetStrictModelConfig(*server_options, true),
      "setting strict model configuration");
#ifdef TRITON_ENABLE_GPU
  double min_compute_capability = TRITON_MIN_COMPUTE_CAPABILITY;
#else
  double min_compute_capability = 0;
#endif  // TRITON_ENABLE_GPU
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
          *server_options, min_compute_capability),
      "setting minimum supported CUDA compute capability");
}


void
CheckServerLiveAndReady(std::shared_ptr<TRITONSERVER_Server> server)
{
  size_t wait_seconds = 0;
  while (true) {
    bool live, ready;
    FAIL_IF_ERR(
        TRITONSERVER_ServerIsLive(server.get(), &live),
        "unable to get server liveness");
    FAIL_IF_ERR(
        TRITONSERVER_ServerIsReady(server.get(), &ready),
        "unable to get server readiness");
    std::cout << "Server Health: live " << live << ", ready " << ready
              << std::endl;
    if (live && ready) {
      break;
    }

    if (++wait_seconds >= 10) {
      FAIL("failed to find healthy inference server");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
}


void
PrintServerStatus(std::shared_ptr<TRITONSERVER_Server> server)
{
  TRITONSERVER_Message* server_metadata_message;
  FAIL_IF_ERR(
      TRITONSERVER_ServerMetadata(server.get(), &server_metadata_message),
      "unable to get server metadata message");
  const char* buffer;
  size_t byte_size;
  FAIL_IF_ERR(
      TRITONSERVER_MessageSerializeToJson(
          server_metadata_message, &buffer, &byte_size),
      "unable to serialize server metadata message");

  std::cout << "Server Status:" << std::endl;
  std::cout << std::string(buffer, byte_size) << std::endl;

  FAIL_IF_ERR(
      TRITONSERVER_MessageDelete(server_metadata_message),
      "deleting status metadata");
}


void
AwaitModelReady(
    std::shared_ptr<TRITONSERVER_Server> server, const std::string model_name)
{
  bool is_ready = false;
  size_t wait_seconds = 0;
  while (!is_ready) {
    FAIL_IF_ERR(
        TRITONSERVER_ServerModelIsReady(
            server.get(), model_name.c_str(), 1, &is_ready),
        "unable to get model readiness");
    if (!is_ready) {
      if (++wait_seconds >= 5) {
        FAIL("model failed to be ready in 5 seconds");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      continue;
    }

    TRITONSERVER_Message* model_metadata_message;
    FAIL_IF_ERR(
        TRITONSERVER_ServerModelMetadata(
            server.get(), model_name.c_str(), 1, &model_metadata_message),
        "unable to get model metadata message");
    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(
        TRITONSERVER_MessageSerializeToJson(
            model_metadata_message, &buffer, &byte_size),
        "unable to serialize model status protobuf");

    rapidjson::Document model_metadata;
    model_metadata.Parse(buffer, byte_size);
    if (model_metadata.HasParseError()) {
      FAIL(
          "error: failed to parse model metadata from JSON: " +
          std::string(GetParseError_En(model_metadata.GetParseError())) +
          " at " + std::to_string(model_metadata.GetErrorOffset()));
    }

    FAIL_IF_ERR(
        TRITONSERVER_MessageDelete(model_metadata_message),
        "deleting status protobuf");

    if (strcmp(model_metadata["name"].GetString(), model_name.c_str())) {
      FAIL("unable to find metadata for model");
    }

    bool found_version = false;
    if (model_metadata.HasMember("versions")) {
      for (const auto& version : model_metadata["versions"].GetArray()) {
        if (strcmp(version.GetString(), "1") == 0) {
          found_version = true;
          break;
        }
      }
    }
    if (!found_version) {
      FAIL("unable to find version 1 status for model");
    }

    FAIL_IF_ERR(ParseModelMetadata(model_metadata), "parsing model metadata");
  }
}


void
LoadInputImageFromFile(
    cv::Mat& dst, std::vector<int64_t>& input0_shape, bool& fixHeight,
    float& ratio, int& sideCache, std::string imageName = "capture.jpg")
{
  const int inputC = input0_shape[1];
  const int inputH = input0_shape[2];
  const int inputW = input0_shape[3];
  const int batchSize = input0_shape[0];

  cv::Mat image = cv::imread(imageName);

  if (image.empty()) {
    std::cout << "Cannot open image " << imageName << std::endl;
    exit(0);
  }

  // resize keeping aspect ratio and pad
  dst = ResizeKeepAspectRatio(
      image, cv::Size(inputW, inputH), cv::Scalar(0, 0, 0), fixHeight, ratio,
      sideCache);

  cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
}


void
LoadInputData(
    cv::Mat& dst, std::vector<float>* input0_data,
    std::vector<int64_t>& input0_shape)
{
  const int inputC = input0_shape[1];
  const int inputH = input0_shape[2];
  const int inputW = input0_shape[3];

  input0_data->resize(inputC * inputH * inputW * sizeof(float));

  // normalize
  Normalize(dst, input0_data, inputC);
}

static std::mutex mutex;

void
RunInferenceAndValidate(
    std::shared_ptr<TRITONSERVER_Server> server,
    TRITONSERVER_ResponseAllocator* allocator, cv::Mat scaled_input_image,
    bool fixHeight, float ratio, int sideCache, std::string model_name,
    size_t thread_id, bool visualize)
{
  TRITONSERVER_InferenceRequest* irequest = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestNew(
          &irequest, server.get(), model_name.c_str(), -1),
      "creating inference request");

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestSetId(irequest, "my_request_id"),
      "setting ID for the request");

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestSetReleaseCallback(
          irequest, InferRequestComplete, nullptr),
      "setting request release callback");

  // Inputs
  auto input0 = "input_1";
  std::vector<int64_t> input0_shape({1, 3, 544, 960});

  const TRITONSERVER_DataType datatype = TRITONSERVER_TYPE_FP32;

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddInput(
          irequest, input0, datatype, &input0_shape[0], input0_shape.size()),
      "setting input 0 meta-data for the request");

  // Outputs
  auto output0 = "output_bbox/BiasAdd";
  auto output1 = "output_cov/Sigmoid";

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output0),
      "requesting output 0 for the request");
  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output1),
      "requesting output 1 for the request");

  // Load the input data
  std::vector<float> input0_data;
  std::vector<int> result_indexes;
  std::vector<cv::Rect> bboxes_list;

  LoadInputData(scaled_input_image, &input0_data, input0_shape);

  size_t input0_size = input0_data.size();

  const void* input0_base = &input0_data[0];

#ifdef TRITON_ENABLE_GPU
  std::unique_ptr<void, decltype(cuda_data_deleter)> input0_gpu(
      nullptr, cuda_data_deleter);
  bool use_cuda_memory =
      (enforce_memory_type &&
       (requested_memory_type != TRITONSERVER_MEMORY_CPU));
  if (use_cuda_memory) {
    FAIL_IF_CUDA_ERR(cudaSetDevice(0), "setting CUDA device to device 0");
    if (requested_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
      void* dst;
      FAIL_IF_CUDA_ERR(
          cudaMalloc(&dst, input0_size),
          "allocating GPU memory for INPUT0 data");
      input0_gpu.reset(dst);
      FAIL_IF_CUDA_ERR(
          cudaMemcpy(dst, &input0_data[0], input0_size, cudaMemcpyHostToDevice),
          "setting INPUT0 data in GPU memory");
    } else {
      void* dst;
      FAIL_IF_CUDA_ERR(
          cudaHostAlloc(&dst, input0_size, cudaHostAllocPortable),
          "allocating pinned memory for INPUT0 data");
      input0_gpu.reset(dst);
      FAIL_IF_CUDA_ERR(
          cudaMemcpy(dst, &input0_data[0], input0_size, cudaMemcpyHostToHost),
          "setting INPUT0 data in pinned memory");
    }
  }

  input0_base = use_cuda_memory ? input0_gpu.get() : &input0_data[0];
#endif  // TRITON_ENABLE_GPU

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAppendInputData(
          irequest, input0, input0_base, input0_size, requested_memory_type, 0),
      "assigning INPUT0 data");

  // Perform inference...
  {
    auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
    std::future<TRITONSERVER_InferenceResponse*> completed = p->get_future();

    FAIL_IF_ERR(
        TRITONSERVER_InferenceRequestSetResponseCallback(
            irequest, allocator, nullptr, InferResponseComplete,
            reinterpret_cast<void*>(p)),
        "setting response callback");

    FAIL_IF_ERR(
        TRITONSERVER_ServerInferAsync(server.get(), irequest, nullptr),
        "running inference");

    // Wait for the inference to complete.
    TRITONSERVER_InferenceResponse* completed_response = completed.get();

    FAIL_IF_ERR(
        TRITONSERVER_InferenceResponseError(completed_response),
        "response status");

    std::unique_lock<std::mutex> lock(mutex);

    // Process output
    DetectionInferenceOutput(
        result_indexes, bboxes_list, completed_response, output0, output1,
        input0_shape, fixHeight, ratio, sideCache, thread_id, visualize);

    FAIL_IF_ERR(
        TRITONSERVER_InferenceResponseDelete(completed_response),
        "deleting inference response");
  }

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestDelete(irequest),
      "deleting inference request");
}


void
PrintModelStats(
    std::shared_ptr<TRITONSERVER_Server> server, const std::string model_name)
{
  TRITONSERVER_Message* model_stats_message = nullptr;

  FAIL_IF_ERR(
      TRITONSERVER_ServerModelStatistics(
          server.get(), model_name.c_str(), -1 /* model_version */,
          &model_stats_message),
      "unable to get model stats message");
  const char* buffer;
  size_t byte_size;
  FAIL_IF_ERR(
      TRITONSERVER_MessageSerializeToJson(
          model_stats_message, &buffer, &byte_size),
      "unable to serialize server metadata message");

  std::cout << "Model '" << model_name << "' Stats:" << std::endl;
  std::cout << std::string(buffer, byte_size) << std::endl;

  FAIL_IF_ERR(
      TRITONSERVER_MessageDelete(model_stats_message),
      "deleting model stats message");
}


void
CreateAndRunTritonserverInstance(
    std::string model_repository_path, std::string tritonserver_path,
    bool verbose_level, int thread_count, bool visualize)
{
  TRITONSERVER_ServerOptions* server_options = nullptr;

  SetServerOptions(
      &server_options, verbose_level, model_repository_path, tritonserver_path);

  TRITONSERVER_Server* server_ptr = nullptr;

  FAIL_IF_ERR(
      TRITONSERVER_ServerNew(&server_ptr, server_options),
      "creating server instance. ");

  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsDelete(server_options),
      "deleting server options");

  std::shared_ptr<TRITONSERVER_Server> server(
      server_ptr, TRITONSERVER_ServerDelete);

  // Wait and until the server is both live and ready.
  CheckServerLiveAndReady(server);

  // Print status of the servers.
  PrintServerStatus(server);
  std::string model = "peoplenet";

  // Load models in server.
  FAIL_IF_ERR(
      TRITONSERVER_ServerLoadModel(server.get(), model.c_str()),
      "failed to load model peoplenet");

  // Wait for the models to become available.
  AwaitModelReady(server, model.c_str());

  // Create the allocator that will be used to allocate buffers for
  // the result tensors.
  TRITONSERVER_ResponseAllocator* allocator = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_ResponseAllocatorNew(
          &allocator, ResponseAlloc, ResponseRelease, nullptr /* start_fn */),
      "creating response allocator");


  // Measure total execution time
  using std::chrono::duration;
  using std::chrono::duration_cast;
  using std::chrono::high_resolution_clock;
  using std::chrono::milliseconds;

  cv::Mat scaled_input_image;
  bool fixHeight;
  float ratio;
  int sideCache;
  std::vector<int64_t> input0_shape({1, 3, 544, 960});

  // the input image is loaded only once and used for all inferences
  LoadInputImageFromFile(
      scaled_input_image, input0_shape, fixHeight, ratio, sideCache);

  auto t1 = high_resolution_clock::now();

  // Multi-thread inference
  std::thread inferences[thread_count];
  for (size_t i = 0; i < thread_count; i++) {
    inferences[i] = std::thread(
        &RunInferenceAndValidate, server, allocator, scaled_input_image,
        fixHeight, ratio, sideCache, model.c_str(), i, visualize);
  }

  for (int i = 0; i < thread_count; ++i) {
    inferences[i].join();
  }

  // Second time point to measure elapsed time
  auto t2 = high_resolution_clock::now();

  FAIL_IF_ERR(
      TRITONSERVER_ResponseAllocatorDelete(allocator),
      "deleting response allocator");

  // Print Model Statistics for all models
  PrintModelStats(server, model.c_str());

  // Unload models in the servers.
  FAIL_IF_ERR(
      TRITONSERVER_ServerUnloadModel(server.get(), model.c_str()),
      "failed to unload model");

  /* Getting number of milliseconds as an integer. */
  auto ms_int = duration_cast<milliseconds>(t2 - t1);

  std::cout << "\n TOTAL INFERENCE TIME: " << ms_int.count() << "ms\n";
}


int
main(int argc, char** argv)
{
  std::string model_repository_path;
  std::string tritonserver_path;
  int verbose_level = 0;
  int thread_count = 2;
  bool visualize = false;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vm:r:p:t:s:")) != -1) {
    switch (opt) {
      case 'm': {
        enforce_memory_type = true;
        if (!strcmp(optarg, "system")) {
          requested_memory_type = TRITONSERVER_MEMORY_CPU;
        } else if (!strcmp(optarg, "pinned")) {
          requested_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
        } else if (!strcmp(optarg, "gpu")) {
          requested_memory_type = TRITONSERVER_MEMORY_GPU;
        } else {
          Usage(
              argv,
              "-m must be used to specify one of the following types:"
              " <\"system\"|\"pinned\"|gpu>");
        }
        break;
      }
      case 'r':
        model_repository_path = optarg;
        break;
      case 'p':
        tritonserver_path = optarg;
        break;
      case 'v':
        verbose_level = 1;
        break;
      case 't':
        thread_count = std::stoi(optarg);
        break;
      case 's':
        if (!strcmp(optarg, "true")) {
          visualize = true;
        } else if (!strcmp(optarg, "false")) {
          visualize = false;
        } else {
          Usage(
              argv,
              "-s must be:"
              " <true|false>");
        }
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  if (thread_count < 1) {
    Usage(argv, "thread_count must be >= 1");
  }

  if (model_repository_path.empty()) {
    Usage(argv, "-r must be used to specify model repository path");
  }
#ifndef TRITON_ENABLE_GPU
  if (enforce_memory_type && requested_memory_type != TRITONSERVER_MEMORY_CPU) {
    Usage(argv, "-m can only be set to \"system\" without enabling GPU");
  }
#endif  // TRITON_ENABLE_GPU

  // Check API version.
  uint32_t api_version_major, api_version_minor;
  FAIL_IF_ERR(
      TRITONSERVER_ApiVersion(&api_version_major, &api_version_minor),
      "getting Triton API version");
  if ((TRITONSERVER_API_VERSION_MAJOR != api_version_major) ||
      (TRITONSERVER_API_VERSION_MINOR > api_version_minor)) {
    FAIL("triton server API version mismatch");
  }

  CreateAndRunTritonserverInstance(
      model_repository_path, tritonserver_path, verbose_level, thread_count,
      visualize);

  return 0;
}
