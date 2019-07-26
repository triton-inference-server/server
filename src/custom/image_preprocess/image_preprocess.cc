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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/custom/sdk/custom_instance.h"

#define LOG_ERROR std::cerr
#define LOG_INFO std::cout

// This custom backend takes a byte string of original image as input and
// returns preprocessed image in the shape and format specified in model
// configuration.
//

namespace nvidia { namespace inferenceserver { namespace custom {
namespace image_preprocess {

enum ScaleType { NONE = 0, VGG = 1, INCEPTION = 2 };

// Context object. All state must be kept in this object.
class Context : public CustomInstance {
 public:
  Context(
      const std::string& instance_name, const ModelConfig& config,
      const int gpu_device);

  // Initialize the context. Validate that the model configuration,
  // etc. is something that we can handle.
  int Init();

  // Perform custom execution on the payloads.
  int Execute(
      const uint32_t payload_cnt, CustomPayload* payloads,
      CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn);

 private:
  // Obtain the input tensor as contiguous chunk. If batched, decompose it.
  int GetInputTensor(
      CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
      const uint32_t batch_size, std::vector<std::vector<char>>& input);

  int Preprocess(const cv::Mat& img, char* data, size_t* image_byte_size);

  bool ParseType(const DataType& dtype, int* type1, int* type3);

  // The format of the preprocessed image
  ModelInput::Format format_;

  // The scaling used to normalize the image for corresponding models
  ScaleType scaling_;

  // The shape of preprocessed image
  std::vector<int64_t> output_shape_;

  // The data type of preprocessed image
  DataType output_type_;

  // Local error codes
  const int kBatching = RegisterError("batching not supported");
  const int kOutput = RegisterError(
      "expected single output with 3 dimensions, each dimension >= 1");
  const int kOutputBuffer =
      RegisterError("unable to get buffer for output tensor values");
  const int kInput = RegisterError("expected single input, 1 STRING element");
  const int kInputBuffer =
      RegisterError("unable to get buffer for input tensor values");
  const int kInputSize =
      RegisterError("input obtained does not match batch size");
  const int kOpenCV = RegisterError("unable to preprocess image");
};

Context::Context(
    const std::string& instance_name, const ModelConfig& model_config,
    const int gpu_device)
    : CustomInstance(instance_name, model_config, gpu_device)
{
}

int
Context::Init()
{
  // There must be one input, and batch is allowed to preprocess multiple
  // images
  if (model_config_.input_size() != 1) {
    return kInput;
  }
  // Using STRING type such that input only needs shape [1]
  if (model_config_.input(0).dims_size() != 1) {
    return kInput;
  }
  if (model_config_.input(0).dims(0) != 1) {
    return kInput;
  }
  if (model_config_.input(0).data_type() != DataType::TYPE_STRING) {
    return kInput;
  }

  // There must be an output
  if (model_config_.output_size() != 1) {
    return kOutput;
  }

  // The output shape must have three dims for width, height and color channel
  if (model_config_.output(0).dims_size() != 3) {
    return kOutput;
  } else {
    for (const auto& dim : model_config_.output(0).dims()) {
      if (dim < 1) {
        return kOutput;
      }
      output_shape_.push_back(dim);
    }
  }
  output_type_ = model_config_.output(0).data_type();

  for (const auto& pr : model_config_.parameters()) {
    if (pr.first == "format") {
      if (pr.second.string_value() == "NHWC") {
        format_ = ModelInput::FORMAT_NHWC;
      } else {
        format_ = ModelInput::FORMAT_NCHW;
      }
    } else if (pr.first == "scaling") {
      if (pr.second.string_value() == "VGG") {
        scaling_ = ScaleType::VGG;
      } else if (pr.second.string_value() == "INCEPTION") {
        scaling_ = ScaleType::INCEPTION;
      } else {
        scaling_ = NONE;
      }
    }
  }

  return ErrorCodes::Success;
}

int
Context::Execute(
    const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
  for (size_t idx = 0; idx < payload_cnt; idx++) {
    // If output wasn't requested just do nothing.
    if (payloads[idx].output_cnt == 0) {
      continue;
    }

    // Reads input
    uint32_t batch_size =
        (payloads[idx].batch_size == 0) ? 1 : payloads[idx].batch_size;
    std::vector<std::vector<char>> input;
    int err = GetInputTensor(
        input_fn, payloads[idx].input_context, "INPUT", batch_size, input);
    if (err != ErrorCodes::Success) {
      payloads[idx].error_code = err;
      continue;
    }

    // Obtain the output buffer for the whole batch
    std::vector<int64_t> output_shape = output_shape_;
    output_shape.insert(output_shape.begin(), payloads[idx].batch_size);

    void* obuffer;
    if (!output_fn(
            payloads[idx].output_context,
            payloads[idx].required_output_names[0], output_shape.size(),
            &output_shape[0], GetByteSize(output_type_, output_shape),
            &obuffer)) {
      payloads[idx].error_code = kOutputBuffer;
      continue;
    }

    // If no error but the 'obuffer' is returned as nullptr, then
    // skip writing this output.
    if (obuffer != nullptr) {
      size_t byte_used = 0;
      for (const auto& data : input) {
        cv::Mat img = imdecode(cv::Mat(data), 1);
        if (img.empty()) {
          payloads[idx].error_code = kOpenCV;
          break;
        }

        size_t image_byte_size;

        err = Preprocess(img, (char*)obuffer + byte_used, &image_byte_size);
        if (err != ErrorCodes::Success) {
          payloads[idx].error_code = err;
          break;
        }
        byte_used += image_byte_size;
      }
    }
  }

  return ErrorCodes::Success;
}

int
Context::GetInputTensor(
    CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
    const uint32_t batch_size, std::vector<std::vector<char>>& input)
{
  // The values for an input tensor are not necessarily in one
  // contiguous chunk, so we copy the chunks into 'input' vector. A
  // more performant solution would attempt to use the input tensors
  // in-place instead of having this copy.

  // The size of a STRING data type can only be obtained from the data
  // (convention: first 4 bytes stores the size of the actual data)
  uint32_t image_size = 0;
  uint32_t byte_read = 0;
  std::vector<char> size_buffer;
  while (true) {
    const void* content;
    // Get all content out
    uint64_t content_byte_size = -1;
    if (!input_fn(input_context, name, &content, &content_byte_size)) {
      return kInputBuffer;
    }

    // If 'content' returns nullptr we have all the input.
    if (content == nullptr) {
      break;
    }

    // Keep consuming the content because we want to decompose the batched input
    while (content_byte_size > 0) {
      // If there are input and 'image_size' is not set, try to read image_size
      if (image_size == 0) {
        // Make sure we have enought bytes to read as 'image_size'
        uint64_t byte_to_append = 4 - size_buffer.size();
        byte_to_append = (byte_to_append < content_byte_size)
                             ? byte_to_append
                             : content_byte_size;
        size_buffer.insert(
            size_buffer.end(), static_cast<const char*>(content),
            static_cast<const char*>(content) + byte_to_append);

        // modify position to unread content
        content = static_cast<const char*>(content) + byte_to_append;
        content_byte_size -= byte_to_append;
        if (size_buffer.size() == 4) {
          image_size = *(uint32_t*)(&size_buffer[0]);
          byte_read = 0;
          size_buffer.clear();
          input.emplace_back();
        } else {
          break;
        }
      }

      uint32_t byte_to_read = image_size - byte_read;
      byte_to_read =
          (byte_to_read < content_byte_size) ? byte_to_read : content_byte_size;

      input.back().insert(
          input.back().end(), static_cast<const char*>(content),
          static_cast<const char*>(content) + byte_to_read);

      content = static_cast<const char*>(content) + byte_to_read;
      content_byte_size -= byte_to_read;
      byte_read += byte_to_read;
      if (byte_read == image_size) {
        image_size = 0;
      }
    }
  }

  // Make sure we end up with exactly the amount of input we expect.
  if (batch_size != input.size()) {
    return kInputSize;
  }

  return ErrorCodes::Success;
}

int
Context::Preprocess(const cv::Mat& img, char* data, size_t* image_byte_size)
{
  // Image channels are in BGR order. Currently model configuration
  // data doesn't provide any information as to the expected channel
  // orderings (like RGB, BGR). We are going to assume that RGB is the
  // most likely ordering and so change the channels to that ordering.
  size_t c, h, w;
  if (format_ == ModelInput::FORMAT_NHWC) {
    h = output_shape_[0];
    w = output_shape_[1];
    c = output_shape_[2];
  } else {
    c = output_shape_[0];
    h = output_shape_[1];
    w = output_shape_[2];
  }
  auto img_size = cv::Size(w, h);

  cv::Mat sample;
  if ((img.channels() == 3) && (c == 1)) {
    cv::cvtColor(img, sample, CV_BGR2GRAY);
  } else if ((img.channels() == 4) && (c == 1)) {
    cv::cvtColor(img, sample, CV_BGRA2GRAY);
  } else if ((img.channels() == 3) && (c == 3)) {
    cv::cvtColor(img, sample, CV_BGR2RGB);
  } else if ((img.channels() == 4) && (c == 3)) {
    cv::cvtColor(img, sample, CV_BGRA2RGB);
  } else if ((img.channels() == 1) && (c == 3)) {
    cv::cvtColor(img, sample, CV_GRAY2RGB);
  } else {
    return kOpenCV;
  }

  int img_type1, img_type3;
  if (!ParseType(output_type_, &img_type1, &img_type3)) {
    return kOpenCV;
  }

  cv::Mat sample_resized;
  if (sample.size() != img_size) {
    cv::resize(sample, sample_resized, img_size);
  } else {
    sample_resized = sample;
  }

  cv::Mat sample_type;
  sample_resized.convertTo(sample_type, (c == 3) ? img_type3 : img_type1);

  cv::Mat sample_final;
  if (scaling_ == ScaleType::INCEPTION) {
    if (c == 1) {
      sample_final = sample_type.mul(cv::Scalar(1 / 128.0));
      sample_final = sample_final - cv::Scalar(1.0);
    } else {
      sample_final =
          sample_type.mul(cv::Scalar(1 / 128.0, 1 / 128.0, 1 / 128.0));
      sample_final = sample_final - cv::Scalar(1.0, 1.0, 1.0);
    }
  } else if (scaling_ == ScaleType::VGG) {
    if (c == 1) {
      sample_final = sample_type - cv::Scalar(128);
    } else {
      sample_final = sample_type - cv::Scalar(104, 117, 123);
    }
  } else {
    sample_final = sample_type;
  }

  // Allocate a buffer to hold all image elements.
  *image_byte_size = sample_final.total() * sample_final.elemSize();
  size_t pos = 0;

  // For NHWC format Mat is already in the correct order but need to
  // handle both cases of data being contigious or not.
  if (format_ == ModelInput::FORMAT_NHWC) {
    if (sample_final.isContinuous()) {
      memcpy(data, sample_final.datastart, *image_byte_size);
      pos = *image_byte_size;
    } else {
      size_t row_byte_size = sample_final.cols * sample_final.elemSize();
      for (int r = 0; r < sample_final.rows; ++r) {
        memcpy(&(data[pos]), sample_final.ptr<uint8_t>(r), row_byte_size);
        pos += row_byte_size;
      }
    }
  } else {
    // (format_ == ModelInput::FORMAT_NCHW)
    //
    // For CHW formats must split out each channel from the matrix and
    // order them as BBBB...GGGG...RRRR. To do this split the channels
    // of the image directly into 'input_data'. The BGR channels are
    // backed by the 'input_data' vector so that ends up with CHW
    // order of the data.
    std::vector<cv::Mat> input_bgr_channels;
    for (size_t i = 0; i < c; ++i) {
      input_bgr_channels.emplace_back(
          img_size.height, img_size.width, img_type1, &(data[pos]));
      pos += input_bgr_channels.back().total() *
             input_bgr_channels.back().elemSize();
    }

    cv::split(sample_final, input_bgr_channels);
  }

  if (pos != *image_byte_size) {
    return kOpenCV;
  }
  return ErrorCodes::Success;
}

bool
Context::ParseType(const DataType& dtype, int* type1, int* type3)
{
  if (dtype == DataType::TYPE_UINT8) {
    *type1 = CV_8UC1;
    *type3 = CV_8UC3;
  } else if (dtype == DataType::TYPE_INT8) {
    *type1 = CV_8SC1;
    *type3 = CV_8SC3;
  } else if (dtype == DataType::TYPE_UINT16) {
    *type1 = CV_16UC1;
    *type3 = CV_16UC3;
  } else if (dtype == DataType::TYPE_INT16) {
    *type1 = CV_16SC1;
    *type3 = CV_16SC3;
  } else if (dtype == DataType::TYPE_INT32) {
    *type1 = CV_32SC1;
    *type3 = CV_32SC3;
  } else if (dtype == DataType::TYPE_FP32) {
    *type1 = CV_32FC1;
    *type3 = CV_32FC3;
  } else if (dtype == DataType::TYPE_FP64) {
    *type1 = CV_64FC1;
    *type3 = CV_64FC3;
  } else {
    return false;
  }

  return true;
}


}  // namespace image_preprocess

// Creates a new image_preprocess instance
int
CustomInstance::Create(
    CustomInstance** instance, const std::string& name,
    const ModelConfig& model_config, int gpu_device,
    const CustomInitializeData* data)
{
  // Create the context and validate that the model configuration is
  // something that we can handle.
  image_preprocess::Context* context =
      new image_preprocess::Context(name, model_config, gpu_device);
  *instance = context;

  if (context == nullptr) {
    return ErrorCodes::CreationFailure;
  }

  return context->Init();
}


}}}  // namespace nvidia::inferenceserver::custom
