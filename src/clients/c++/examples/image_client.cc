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
#include <getopt.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <iterator>
#include <mutex>
#include <queue>
#include <string>
#include "src/clients/c++/examples/json_utils.h"
#include "src/clients/c++/library/grpc_client.h"
#include "src/clients/c++/library/http_client.h"

#include <opencv2/core/version.hpp>
#if CV_MAJOR_VERSION == 2
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#elif CV_MAJOR_VERSION >= 3
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#endif

#if CV_MAJOR_VERSION == 4
#define GET_TRANSFORMATION_CODE(x) cv::COLOR_##x
#else
#define GET_TRANSFORMATION_CODE(x) CV_##x
#endif

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;

namespace {

enum ScaleType { NONE = 0, VGG = 1, INCEPTION = 2 };

enum ProtocolType { HTTP = 0, GRPC = 1 };

struct ModelInfo {
  std::string output_name_;
  std::string input_name_;
  std::string input_datatype_;
  // The shape of the input
  int input_c_;
  int input_h_;
  int input_w_;
  // The format of the input
  std::string input_format_;
  int type1_;
  int type3_;
  int max_batch_size_;
};

void
Preprocess(
    const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
    size_t img_channels, const cv::Size& img_size, const ScaleType scale,
    std::vector<uint8_t>* input_data)
{
  // Image channels are in BGR order. Currently model configuration
  // data doesn't provide any information as to the expected channel
  // orderings (like RGB, BGR). We are going to assume that RGB is the
  // most likely ordering and so change the channels to that ordering.

  cv::Mat sample;
  if ((img.channels() == 3) && (img_channels == 1)) {
    cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(BGR2GRAY));
  } else if ((img.channels() == 4) && (img_channels == 1)) {
    cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(BGRA2GRAY));
  } else if ((img.channels() == 3) && (img_channels == 3)) {
    cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(BGR2RGB));
  } else if ((img.channels() == 4) && (img_channels == 3)) {
    cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(BGRA2RGB));
  } else if ((img.channels() == 1) && (img_channels == 3)) {
    cv::cvtColor(img, sample, GET_TRANSFORMATION_CODE(GRAY2RGB));
  } else {
    std::cerr << "unexpected number of channels " << img.channels()
              << " in input image, model expects " << img_channels << "."
              << std::endl;
    exit(1);
  }

  cv::Mat sample_resized;
  if (sample.size() != img_size) {
    cv::resize(sample, sample_resized, img_size);
  } else {
    sample_resized = sample;
  }

  cv::Mat sample_type;
  sample_resized.convertTo(
      sample_type, (img_channels == 3) ? img_type3 : img_type1);

  cv::Mat sample_final;
  if (scale == ScaleType::INCEPTION) {
    if (img_channels == 1) {
      sample_final = sample_type.mul(cv::Scalar(1 / 128.0));
      sample_final = sample_final - cv::Scalar(1.0);
    } else {
      sample_final =
          sample_type.mul(cv::Scalar(1 / 128.0, 1 / 128.0, 1 / 128.0));
      sample_final = sample_final - cv::Scalar(1.0, 1.0, 1.0);
    }
  } else if (scale == ScaleType::VGG) {
    if (img_channels == 1) {
      sample_final = sample_type - cv::Scalar(128);
    } else {
      sample_final = sample_type - cv::Scalar(104, 117, 123);
    }
  } else {
    sample_final = sample_type;
  }

  // Allocate a buffer to hold all image elements.
  size_t img_byte_size = sample_final.total() * sample_final.elemSize();
  size_t pos = 0;
  input_data->resize(img_byte_size);

  // For NHWC format Mat is already in the correct order but need to
  // handle both cases of data being contigious or not.
  if (format.compare("FORMAT_NHWC") == 0) {
    if (sample_final.isContinuous()) {
      memcpy(&((*input_data)[0]), sample_final.datastart, img_byte_size);
      pos = img_byte_size;
    } else {
      size_t row_byte_size = sample_final.cols * sample_final.elemSize();
      for (int r = 0; r < sample_final.rows; ++r) {
        memcpy(
            &((*input_data)[pos]), sample_final.ptr<uint8_t>(r), row_byte_size);
        pos += row_byte_size;
      }
    }
  } else {
    // (format.compare("FORMAT_NCHW") == 0)
    //
    // For CHW formats must split out each channel from the matrix and
    // order them as BBBB...GGGG...RRRR. To do this split the channels
    // of the image directly into 'input_data'. The BGR channels are
    // backed by the 'input_data' vector so that ends up with CHW
    // order of the data.
    std::vector<cv::Mat> input_bgr_channels;
    for (size_t i = 0; i < img_channels; ++i) {
      input_bgr_channels.emplace_back(
          img_size.height, img_size.width, img_type1, &((*input_data)[pos]));
      pos += input_bgr_channels.back().total() *
             input_bgr_channels.back().elemSize();
    }

    cv::split(sample_final, input_bgr_channels);
  }

  if (pos != img_byte_size) {
    std::cerr << "unexpected total size of channels " << pos << ", expecting "
              << img_byte_size << std::endl;
    exit(1);
  }
}


void
Postprocess(
    const std::unique_ptr<nic::InferResult> result,
    const std::vector<std::string>& filenames, const size_t batch_size,
    const std::string& output_name, const size_t topk, const bool batching)
{
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

  // Validate shape. Special handling for non-batch model
  if (!batching) {
    if ((shape.size() != 1) || (shape[0] != (int)topk)) {
      std::cerr << "received incorrect shape for " << output_name << std::endl;
      exit(1);
    }
  } else {
    if ((shape.size() != 2) || (shape[0] != (int)batch_size) ||
        (shape[1] != (int)topk)) {
      std::cerr << "received incorrect shape for " << output_name << std::endl;
      exit(1);
    }
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
  std::cerr << "    Note that image folder should only contain image files."
            << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-a" << std::endl;
  std::cerr << "\t--streaming" << std::endl;
  std::cerr << "\t-b <batch size>" << std::endl;
  std::cerr << "\t-c <topk>" << std::endl;
  std::cerr << "\t-s <NONE|INCEPTION|VGG>" << std::endl;
  std::cerr << "\t-p <proprocessed output filename>" << std::endl;
  std::cerr << "\t-m <model name>" << std::endl;
  std::cerr << "\t-x <model version>" << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-i <Protocol used to communicate with inference service>"
            << std::endl;
  std::cerr << "\t-H <HTTP header>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "If -a is specified then asynchronous client API will be used. "
            << "Default is to use the synchronous API." << std::endl;
  std::cerr << "The --streaming flag is only valid with gRPC protocol."
            << std::endl;
  std::cerr
      << "For -b, a single image will be replicated and sent in a batch"
      << std::endl
      << "        of the specified size. A directory of images will be grouped"
      << std::endl
      << "        into batches. Default is 1." << std::endl;
  std::cerr << "For -c, the <topk> classes will be returned, default is 1."
            << std::endl;
  std::cerr << "For -s, specify the type of pre-processing scaling that"
            << std::endl
            << "        should be performed on the image, default is NONE."
            << std::endl
            << "    INCEPTION: scale each pixel RGB value to [-1.0, 1.0)."
            << std::endl
            << "    VGG: subtract mean BGR value (104, 117, 123) from"
            << std::endl
            << "         each pixel." << std::endl;
  std::cerr
      << "If -x is not specified the most recent version (that is, the highest "
      << "numbered version) of the model will be used." << std::endl;
  std::cerr << "For -p, it generates file only if image file is specified."
            << std::endl;
  std::cerr << "For -u, the default server URL is localhost:8000." << std::endl;
  std::cerr << "For -i, available protocols are gRPC and HTTP. Default is HTTP."
            << std::endl;
  std::cerr
      << "For -H, the header will be added to HTTP requests (ignored for GRPC "
         "requests). The header must be specified as 'Header:Value'. -H may be "
         "specified multiple times to add multiple headers."
      << std::endl;
  std::cerr << std::endl;

  exit(1);
}

ScaleType
ParseScale(const std::string& str)
{
  if (str == "NONE") {
    return ScaleType::NONE;
  } else if (str == "INCEPTION") {
    return ScaleType::INCEPTION;
  } else if (str == "VGG") {
    return ScaleType::VGG;
  }

  std::cerr << "unexpected scale type \"" << str
            << "\", expecting NONE, INCEPTION or VGG" << std::endl;
  exit(1);

  return ScaleType::NONE;
}

ProtocolType
ParseProtocol(const std::string& str)
{
  std::string protocol(str);
  std::transform(protocol.begin(), protocol.end(), protocol.begin(), ::tolower);
  if (protocol == "http") {
    return ProtocolType::HTTP;
  } else if (protocol == "grpc") {
    return ProtocolType::GRPC;
  }

  std::cerr << "unexpected protocol type \"" << str
            << "\", expecting HTTP or gRPC" << std::endl;
  exit(1);

  return ProtocolType::HTTP;
}

bool
ParseType(const std::string& dtype, int* type1, int* type3)
{
  if (dtype.compare("UINT8") == 0) {
    *type1 = CV_8UC1;
    *type3 = CV_8UC3;
  } else if (dtype.compare("INT8") == 0) {
    *type1 = CV_8SC1;
    *type3 = CV_8SC3;
  } else if (dtype.compare("UINT16") == 0) {
    *type1 = CV_16UC1;
    *type3 = CV_16UC3;
  } else if (dtype.compare("INT16") == 0) {
    *type1 = CV_16SC1;
    *type3 = CV_16SC3;
  } else if (dtype.compare("INT32") == 0) {
    *type1 = CV_32SC1;
    *type3 = CV_32SC3;
  } else if (dtype.compare("FP32") == 0) {
    *type1 = CV_32FC1;
    *type3 = CV_32FC3;
  } else if (dtype.compare("FP64") == 0) {
    *type1 = CV_64FC1;
    *type3 = CV_64FC3;
  } else {
    return false;
  }

  return true;
}

void
ParseModelGrpc(
    const inference::ModelMetadataResponse& model_metadata,
    const inference::ModelConfigResponse& model_config, const size_t batch_size,
    ModelInfo* model_info)
{
  if (model_metadata.inputs().size() != 1) {
    std::cerr << "expecting 1 input, got " << model_metadata.inputs().size()
              << std::endl;
    exit(1);
  }

  if (model_metadata.outputs().size() != 1) {
    std::cerr << "expecting 1 output, got " << model_metadata.outputs().size()
              << std::endl;
    exit(1);
  }

  if (model_config.config().input().size() != 1) {
    std::cerr << "expecting 1 input in model configuration, got "
              << model_config.config().input().size() << std::endl;
    exit(1);
  }

  auto input_metadata = model_metadata.inputs(0);
  auto input_config = model_config.config().input(0);
  auto output_metadata = model_metadata.outputs(0);

  if (output_metadata.datatype().compare("FP32") != 0) {
    std::cerr << "expecting output datatype to be FP32, model '"
              << model_metadata.name() << "' output type is '"
              << output_metadata.datatype() << "'" << std::endl;
    exit(1);
  }

  model_info->max_batch_size_ = model_config.config().max_batch_size();

  // Model specifying maximum batch size of 0 indicates that batching
  // is not supported and so the input tensors do not expect a "N"
  // dimension (and 'batch_size' should be 1 so that only a single
  // image instance is inferred at a time).
  if (model_info->max_batch_size_ == 0) {
    if (batch_size != 1) {
      std::cerr << "batching not supported for model \""
                << model_metadata.name() << "\"" << std::endl;
      exit(1);
    }
  } else {
    //  model_info->max_batch_size_ > 0
    if (batch_size > (size_t)model_info->max_batch_size_) {
      std::cerr << "expecting batch size <= " << model_info->max_batch_size_
                << " for model '" << model_metadata.name() << "'" << std::endl;
      exit(1);
    }
  }

  // Output is expected to be a vector. But allow any number of
  // dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
  // }, { 10, 1, 1 } are all ok).
  bool output_batch_dim = (model_info->max_batch_size_ > 0);
  size_t non_one_cnt = 0;
  for (const auto dim : output_metadata.shape()) {
    if (output_batch_dim) {
      output_batch_dim = false;
    } else if (dim == -1) {
      std::cerr << "variable-size dimension in model output not supported"
                << std::endl;
      exit(1);
    } else if (dim > 1) {
      non_one_cnt += 1;
      if (non_one_cnt > 1) {
        std::cerr << "expecting model output to be a vector" << std::endl;
        exit(1);
      }
    }
  }

  // Model input must have 3 dims, either CHW or HWC (not counting the
  // batch dimension), either CHW or HWC
  const bool input_batch_dim = (model_info->max_batch_size_ > 0);
  const int expected_input_dims = 3 + (input_batch_dim ? 1 : 0);
  if (input_metadata.shape().size() != expected_input_dims) {
    std::cerr << "expecting input to have " << expected_input_dims
              << " dimensions, model '" << model_metadata.name()
              << "' input has " << input_metadata.shape().size() << std::endl;
    exit(1);
  }

  if ((input_config.format() != inference::ModelInput::FORMAT_NCHW) &&
      (input_config.format() != inference::ModelInput::FORMAT_NHWC)) {
    std::cerr << "unexpected input format "
              << inference::ModelInput_Format_Name(input_config.format())
              << ", expecting "
              << inference::ModelInput_Format_Name(inference::ModelInput::FORMAT_NHWC)
              << " or "
              << inference::ModelInput_Format_Name(inference::ModelInput::FORMAT_NCHW)
              << std::endl;
    exit(1);
  }

  model_info->output_name_ = output_metadata.name();
  model_info->input_name_ = input_metadata.name();
  model_info->input_datatype_ = input_metadata.datatype();

  if (input_config.format() == inference::ModelInput::FORMAT_NHWC) {
    model_info->input_format_ = "FORMAT_NHWC";
    model_info->input_h_ = input_metadata.shape(input_batch_dim ? 1 : 0);
    model_info->input_w_ = input_metadata.shape(input_batch_dim ? 2 : 1);
    model_info->input_c_ = input_metadata.shape(input_batch_dim ? 3 : 2);
  } else {
    model_info->input_format_ = "FORMAT_NCHW";
    model_info->input_c_ = input_metadata.shape(input_batch_dim ? 1 : 0);
    model_info->input_h_ = input_metadata.shape(input_batch_dim ? 2 : 1);
    model_info->input_w_ = input_metadata.shape(input_batch_dim ? 3 : 2);
  }

  if (!ParseType(
          model_info->input_datatype_, &(model_info->type1_),
          &(model_info->type3_))) {
    std::cerr << "unexpected input datatype '" << model_info->input_datatype_
              << "' for model \"" << model_metadata.name() << std::endl;
    exit(1);
  }
}

void
ParseModelHttp(
    const rapidjson::Document& model_metadata,
    const rapidjson::Document& model_config, const size_t batch_size,
    ModelInfo* model_info)
{
  const auto& input_itr = model_metadata.FindMember("inputs");
  size_t input_count = 0;
  if (input_itr != model_metadata.MemberEnd()) {
    input_count = input_itr->value.Size();
  }
  if (input_count != 1) {
    std::cerr << "expecting 1 input, got " << input_count << std::endl;
    exit(1);
  }

  const auto& output_itr = model_metadata.FindMember("outputs");
  size_t output_count = 0;
  if (output_itr != model_metadata.MemberEnd()) {
    output_count = output_itr->value.Size();
  }
  if (output_count != 1) {
    std::cerr << "expecting 1 output, got " << output_count << std::endl;
    exit(1);
  }

  const auto& input_config_itr = model_config.FindMember("input");
  input_count = 0;
  if (input_config_itr != model_config.MemberEnd()) {
    input_count = input_config_itr->value.Size();
  }
  if (input_count != 1) {
    std::cerr << "expecting 1 input in model configuration, got " << input_count
              << std::endl;
    exit(1);
  }

  const auto& input_metadata = *input_itr->value.Begin();
  const auto& input_config = *input_config_itr->value.Begin();
  const auto& output_metadata = *output_itr->value.Begin();

  const auto& output_dtype_itr = output_metadata.FindMember("datatype");
  if (output_dtype_itr == output_metadata.MemberEnd()) {
    std::cerr << "output missing datatype in the metadata for model'"
              << model_metadata["name"].GetString() << "'" << std::endl;
    exit(1);
  }
  auto datatype = std::string(
      output_dtype_itr->value.GetString(),
      output_dtype_itr->value.GetStringLength());
  if (datatype.compare("FP32") != 0) {
    std::cerr << "expecting output datatype to be FP32, model '"
              << model_metadata["name"].GetString() << "' output type is '"
              << datatype << "'" << std::endl;
    exit(1);
  }

  int max_batch_size = 0;
  const auto bs_itr = model_config.FindMember("max_batch_size");
  if (bs_itr != model_config.MemberEnd()) {
    max_batch_size = bs_itr->value.GetUint();
  }
  model_info->max_batch_size_ = max_batch_size;

  // Model specifying maximum batch size of 0 indicates that batching
  // is not supported and so the input tensors do not expect a "N"
  // dimension (and 'batch_size' should be 1 so that only a single
  // image instance is inferred at a time).
  if (max_batch_size == 0) {
    if (batch_size != 1) {
      std::cerr << "batching not supported for model '"
                << model_metadata["name"].GetString() << "'" << std::endl;
      exit(1);
    }
  } else {
    // max_batch_size > 0
    if (batch_size > (size_t)max_batch_size) {
      std::cerr << "expecting batch size <= " << max_batch_size
                << " for model '" << model_metadata["name"].GetString() << "'"
                << std::endl;
      exit(1);
    }
  }

  // Output is expected to be a vector. But allow any number of
  // dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
  // }, { 10, 1, 1 } are all ok).
  bool output_batch_dim = (max_batch_size > 0);
  size_t non_one_cnt = 0;
  const auto output_shape_itr = output_metadata.FindMember("shape");
  if (output_shape_itr != output_metadata.MemberEnd()) {
    const rapidjson::Value& shape_json = output_shape_itr->value;
    for (rapidjson::SizeType i = 0; i < shape_json.Size(); i++) {
      if (output_batch_dim) {
        output_batch_dim = false;
      } else if (shape_json[i].GetInt() == -1) {
        std::cerr << "variable-size dimension in model output not supported"
                  << std::endl;
        exit(1);
      } else if (shape_json[i].GetInt() > 1) {
        non_one_cnt += 1;
        if (non_one_cnt > 1) {
          std::cerr << "expecting model output to be a vector" << std::endl;
          exit(1);
        }
      }
    }
  } else {
    std::cerr << "output missing shape in the metadata for model'"
              << model_metadata["name"].GetString() << "'" << std::endl;
    exit(1);
  }


  // Model input must have 3 dims, either CHW or HWC (not counting the
  // batch dimension), either CHW or HWC
  const bool input_batch_dim = (max_batch_size > 0);
  const size_t expected_input_dims = 3 + (input_batch_dim ? 1 : 0);
  const auto input_shape_itr = input_metadata.FindMember("shape");
  if (input_shape_itr != input_metadata.MemberEnd()) {
    if (input_shape_itr->value.Size() != expected_input_dims) {
      std::cerr << "expecting input to have " << expected_input_dims
                << " dimensions, model '" << model_metadata["name"].GetString()
                << "' input has " << input_shape_itr->value.Size() << std::endl;
      exit(1);
    }
  } else {
    std::cerr << "input missing shape in the metadata for model'"
              << model_metadata["name"].GetString() << "'" << std::endl;
    exit(1);
  }

  model_info->input_format_ = std::string(
      input_config["format"].GetString(),
      input_config["format"].GetStringLength());
  if ((model_info->input_format_.compare("FORMAT_NCHW") != 0) &&
      (model_info->input_format_.compare("FORMAT_NHWC") != 0)) {
    std::cerr << "unexpected input format " << model_info->input_format_
              << ", expecting FORMAT_NCHW or FORMAT_NHWC" << std::endl;
    exit(1);
  }

  model_info->output_name_ = std::string(
      output_metadata["name"].GetString(),
      output_metadata["name"].GetStringLength());
  model_info->input_name_ = std::string(
      input_metadata["name"].GetString(),
      input_metadata["name"].GetStringLength());
  model_info->input_datatype_ = std::string(
      input_metadata["datatype"].GetString(),
      input_metadata["datatype"].GetStringLength());

  if (model_info->input_format_.compare("FORMAT_NHWC") == 0) {
    model_info->input_h_ =
        input_shape_itr->value[input_batch_dim ? 1 : 0].GetInt();
    model_info->input_w_ =
        input_shape_itr->value[input_batch_dim ? 2 : 1].GetInt();
    model_info->input_c_ =
        input_shape_itr->value[input_batch_dim ? 3 : 2].GetInt();
  } else {
    model_info->input_c_ =
        input_shape_itr->value[input_batch_dim ? 1 : 0].GetInt();
    model_info->input_h_ =
        input_shape_itr->value[input_batch_dim ? 2 : 1].GetInt();
    model_info->input_w_ =
        input_shape_itr->value[input_batch_dim ? 3 : 2].GetInt();
  }

  if (!ParseType(
          model_info->input_datatype_, &(model_info->type1_),
          &(model_info->type3_))) {
    std::cerr << "unexpected input datatype '" << model_info->input_datatype_
              << "' for model \"" << model_metadata["name"].GetString()
              << std::endl;
    exit(1);
  }
}

void
FileToInputData(
    const std::string& filename, size_t c, size_t h, size_t w,
    const std::string& format, int type1, int type3, ScaleType scale,
    std::vector<uint8_t>* input_data)
{
  // Load the specified image.
  std::ifstream file(filename);
  std::vector<char> data;
  file >> std::noskipws;
  std::copy(
      std::istream_iterator<char>(file), std::istream_iterator<char>(),
      std::back_inserter(data));
  if (data.empty()) {
    std::cerr << "error: unable to read image file " << filename << std::endl;
    exit(1);
  }

  cv::Mat img = imdecode(cv::Mat(data), 1);
  if (img.empty()) {
    std::cerr << "error: unable to decode image " << filename << std::endl;
    exit(1);
  }

  // Pre-process the image to match input size expected by the model.
  Preprocess(img, format, type1, type3, c, cv::Size(w, h), scale, input_data);
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
  bool async = false;
  bool streaming = false;
  int batch_size = 1;
  int topk = 1;
  ScaleType scale = ScaleType::NONE;
  std::string preprocess_output_filename;
  std::string model_name;
  std::string model_version = "";
  std::string url("localhost:8000");
  ProtocolType protocol = ProtocolType::HTTP;
  nic::Headers http_headers;

  static struct option long_options[] = {{"streaming", 0, 0, 0}, {0, 0, 0, 0}};

  // Parse commandline...
  int opt;
  while ((opt = getopt_long(
              argc, argv, "vau:m:x:b:c:s:p:i:H:", long_options, NULL)) != -1) {
    switch (opt) {
      case 0:
        streaming = true;
        break;
      case 'v':
        verbose = true;
        break;
      case 'a':
        async = true;
        break;
      case 'u':
        url = optarg;
        break;
      case 'm':
        model_name = optarg;
        break;
      case 'x':
        model_version = optarg;
        break;
      case 'b':
        batch_size = std::atoi(optarg);
        break;
      case 'c':
        topk = std::atoi(optarg);
        break;
      case 's':
        scale = ParseScale(optarg);
        break;
      case 'p':
        preprocess_output_filename = optarg;
        break;
      case 'i':
        protocol = ParseProtocol(optarg);
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

  if (model_name.empty()) {
    Usage(argv, "-m flag must be specified");
  }
  if (batch_size <= 0) {
    Usage(argv, "batch size must be > 0");
  }
  if (topk <= 0) {
    Usage(argv, "topk must be > 0");
  }
  if (optind >= argc) {
    Usage(argv, "image file or image folder must be specified");
  }
  if (streaming && (protocol != ProtocolType::GRPC)) {
    Usage(argv, "Streaming is only allowed with gRPC protocol");
  }
  if (streaming && (!async)) {
    Usage(argv, "Only async operation is supported in streaming");
  }
  if (!http_headers.empty() && (protocol != ProtocolType::HTTP)) {
    std::cerr << "WARNING: HTTP headers specified with -H are ignored when "
                 "using non-HTTP protocol."
              << std::endl;
  }

  // Create the inference client for the server. From it
  // extract and validate that the model meets the requirements for
  // image classification.
  TritonClient triton_client;
  nic::Error err;
  if (protocol == ProtocolType::HTTP) {
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

  ModelInfo model_info;
  if (protocol == ProtocolType::HTTP) {
    std::string model_metadata;
    err = triton_client.http_client_->ModelMetadata(
        &model_metadata, model_name, model_version, http_headers);
    if (!err.IsOk()) {
      std::cerr << "error: failed to get model metadata: " << err << std::endl;
    }
    rapidjson::Document model_metadata_json;
    err = nic::ParseJson(&model_metadata_json, model_metadata);
    if (!err.IsOk()) {
      std::cerr << "error: failed to parse model metadata: " << err
                << std::endl;
    }
    std::string model_config;
    err = triton_client.http_client_->ModelConfig(
        &model_config, model_name, model_version, http_headers);
    if (!err.IsOk()) {
      std::cerr << "error: failed to get model config: " << err << std::endl;
    }
    rapidjson::Document model_config_json;
    err = nic::ParseJson(&model_config_json, model_config);
    if (!err.IsOk()) {
      std::cerr << "error: failed to parse model config: " << err << std::endl;
    }
    ParseModelHttp(
        model_metadata_json, model_config_json, batch_size, &model_info);
  } else {
    inference::ModelMetadataResponse model_metadata;
    err = triton_client.grpc_client_->ModelMetadata(
        &model_metadata, model_name, model_version, http_headers);
    if (!err.IsOk()) {
      std::cerr << "error: failed to get model metadata: " << err << std::endl;
    }
    inference::ModelConfigResponse model_config;
    err = triton_client.grpc_client_->ModelConfig(
        &model_config, model_name, model_version, http_headers);
    if (!err.IsOk()) {
      std::cerr << "error: failed to get model config: " << err << std::endl;
    }
    ParseModelGrpc(model_metadata, model_config, batch_size, &model_info);
  }

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

  // Preprocess the images into input data according to model
  // requirements
  std::vector<std::vector<uint8_t>> image_data;
  for (const auto& fn : image_filenames) {
    image_data.emplace_back();
    FileToInputData(
        fn, model_info.input_c_, model_info.input_h_, model_info.input_w_,
        model_info.input_format_, model_info.type1_, model_info.type3_, scale,
        &(image_data.back()));

    if ((image_data.size() == 1) && !preprocess_output_filename.empty()) {
      std::ofstream output_file(preprocess_output_filename);
      std::ostream_iterator<uint8_t> output_iterator(output_file);
      std::copy(image_data[0].begin(), image_data[0].end(), output_iterator);
    }
  }

  std::vector<int64_t> shape;
  // Include the batch dimension if required
  if (model_info.max_batch_size_ != 0) {
    shape.push_back(batch_size);
  }
  if (model_info.input_format_.compare("FORMAT_NHWC") == 0) {
    shape.push_back(model_info.input_h_);
    shape.push_back(model_info.input_w_);
    shape.push_back(model_info.input_c_);
  } else {
    shape.push_back(model_info.input_c_);
    shape.push_back(model_info.input_h_);
    shape.push_back(model_info.input_w_);
  }

  // Initialize the inputs with the data.
  nic::InferInput* input;
  err = nic::InferInput::Create(
      &input, model_info.input_name_, shape, model_info.input_datatype_);
  if (!err.IsOk()) {
    std::cerr << "unable to get input: " << err << std::endl;
    exit(1);
  }
  std::shared_ptr<nic::InferInput> input_ptr(input);


  nic::InferRequestedOutput* output;
  // Set the number of classification expected
  err =
      nic::InferRequestedOutput::Create(&output, model_info.output_name_, topk);
  if (!err.IsOk()) {
    std::cerr << "unable to get output: " << err << std::endl;
    exit(1);
  }
  std::shared_ptr<nic::InferRequestedOutput> output_ptr(output);

  std::vector<nic::InferInput*> inputs = {input_ptr.get()};
  std::vector<const nic::InferRequestedOutput*> outputs = {output_ptr.get()};

  // Configure context for 'batch_size' and 'topk'
  nic::InferOptions options(model_name);
  options.model_version_ = model_version;

  // Send requests of 'batch_size' images. If the number of images
  // isn't an exact multiple of 'batch_size' then just start over with
  // the first images until the batch is filled.
  //
  // Number of requests sent = ceil(number of images / batch_size)
  std::vector<std::unique_ptr<nic::InferResult>> results;
  std::vector<std::vector<std::string>> result_filenames;
  size_t image_idx = 0;
  size_t done_cnt = 0;
  size_t sent_count = 0;
  bool last_request = false;
  std::mutex mtx;
  std::condition_variable cv;

  auto callback_func = [&](nic::InferResult* result) {
    {
      // Defer the response retrieval to main thread
      std::lock_guard<std::mutex> lk(mtx);
      results.emplace_back(result);
      done_cnt++;
    }
    cv.notify_all();
  };

  if (streaming) {
    err = triton_client.grpc_client_->StartStream(
        callback_func, true /* enable_stats */, 0 /* stream_timeout */,
        http_headers);
    if (!err.IsOk()) {
      std::cerr << "failed to establish the stream: " << err << std::endl;
    }
  }

  while (!last_request) {
    // Reset the input for new request.
    err = input_ptr->Reset();
    if (!err.IsOk()) {
      std::cerr << "failed resetting input: " << err << std::endl;
      exit(1);
    }

    // Set input to be the next 'batch_size' images (preprocessed).
    std::vector<std::string> input_filenames;
    for (int idx = 0; idx < batch_size; ++idx) {
      input_filenames.push_back(image_filenames[image_idx]);
      err = input_ptr->AppendRaw(image_data[image_idx]);
      if (!err.IsOk()) {
        std::cerr << "failed setting input: " << err << std::endl;
        exit(1);
      }

      image_idx = (image_idx + 1) % image_data.size();
      if (image_idx == 0) {
        last_request = true;
      }
    }

    result_filenames.emplace_back(std::move(input_filenames));

    options.request_id_ = std::to_string(sent_count);

    // Send request.
    if (!async) {
      nic::InferResult* result;
      if (protocol == ProtocolType::HTTP) {
        err = triton_client.http_client_->Infer(
            &result, options, inputs, outputs, http_headers);
      } else {
        err = triton_client.grpc_client_->Infer(
            &result, options, inputs, outputs, http_headers);
      }
      if (!err.IsOk()) {
        std::cerr << "failed sending synchronous infer request: " << err
                  << std::endl;
        exit(1);
      }
      results.emplace_back(result);
    } else {
      if (streaming) {
        err = triton_client.grpc_client_->AsyncStreamInfer(
            options, inputs, outputs);
      } else {
        if (protocol == ProtocolType::HTTP) {
          err = triton_client.http_client_->AsyncInfer(
              callback_func, options, inputs, outputs, http_headers);
        } else {
          err = triton_client.grpc_client_->AsyncInfer(
              callback_func, options, inputs, outputs, http_headers);
        }
      }
      if (!err.IsOk()) {
        std::cerr << "failed sending asynchronous infer request: " << err
                  << std::endl;
        exit(1);
      }
    }
    sent_count++;
  }

  // For async, retrieve results according to the send order
  if (async) {
    // Wait until all callbacks are invoked
    {
      std::unique_lock<std::mutex> lk(mtx);
      cv.wait(lk, [&]() {
        if (done_cnt >= sent_count) {
          return true;
        } else {
          return false;
        }
      });
    }
  }

  // Post-process the results to make prediction(s)
  for (size_t idx = 0; idx < results.size(); idx++) {
    std::cout << "Request " << idx << ", batch size " << batch_size
              << std::endl;
    Postprocess(
        std::move(results[idx]), result_filenames[idx], batch_size,
        model_info.output_name_, topk, model_info.max_batch_size_ != 0);
  }

  return 0;
}
