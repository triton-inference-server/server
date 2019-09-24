// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include "src/clients/c++/library/request_grpc.h"
#include "src/clients/c++/library/request_http.h"
#include "src/core/model_config.pb.h"

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

void
Preprocess(
    const cv::Mat& img, ni::ModelInput::Format format, int img_type1,
    int img_type3, size_t img_channels, const cv::Size& img_size,
    const ScaleType scale, std::vector<uint8_t>* input_data)
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
  if (format == ni::ModelInput::FORMAT_NHWC) {
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
    // (format == ni::ModelInput::FORMAT_NCHW)
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

  if (filenames.size() != batch_size) {
    std::cerr << "expected " << batch_size << " filenames, got "
              << filenames.size() << std::endl;
    exit(1);
  }

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
ParseType(const ni::DataType& dtype, int* type1, int* type3)
{
  if (dtype == ni::DataType::TYPE_UINT8) {
    *type1 = CV_8UC1;
    *type3 = CV_8UC3;
  } else if (dtype == ni::DataType::TYPE_INT8) {
    *type1 = CV_8SC1;
    *type3 = CV_8SC3;
  } else if (dtype == ni::DataType::TYPE_UINT16) {
    *type1 = CV_16UC1;
    *type3 = CV_16UC3;
  } else if (dtype == ni::DataType::TYPE_INT16) {
    *type1 = CV_16SC1;
    *type3 = CV_16SC3;
  } else if (dtype == ni::DataType::TYPE_INT32) {
    *type1 = CV_32SC1;
    *type3 = CV_32SC3;
  } else if (dtype == ni::DataType::TYPE_FP32) {
    *type1 = CV_32FC1;
    *type3 = CV_32FC3;
  } else if (dtype == ni::DataType::TYPE_FP64) {
    *type1 = CV_64FC1;
    *type3 = CV_64FC3;
  } else {
    return false;
  }

  return true;
}

void
ParseModel(
    const std::unique_ptr<nic::InferContext>& ctx, const size_t batch_size,
    size_t* c, size_t* h, size_t* w, ni::ModelInput::Format* format, int* type1,
    int* type3, bool verbose = false)
{
  if (ctx->Inputs().size() != 1) {
    std::cerr << "expecting 1 input, model \"" << ctx->ModelName() << "\" has "
              << ctx->Inputs().size() << std::endl;
    exit(1);
  }

  if (ctx->Outputs().size() != 1) {
    std::cerr << "expecting 1 output, model \"" << ctx->ModelName() << "\" has "
              << ctx->Outputs().size() << std::endl;
    exit(1);
  }

  const auto& input = ctx->Inputs()[0];
  const auto& output = ctx->Outputs()[0];

  if (output->DType() != ni::DataType::TYPE_FP32) {
    std::cerr << "expecting model output datatype to be TYPE_FP32, model \""
              << ctx->ModelName() << "\" output type is "
              << ni::DataType_Name(output->DType()) << std::endl;
    exit(1);
  }

  // Output is expected to be a vector. But allow any number of
  // dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
  // }, { 10, 1, 1 } are all ok). Variable-size dimensions are not
  // currently supported.
  size_t non_one_cnt = 0;
  for (const auto dim : output->Dims()) {
    if (dim == -1) {
      std::cerr << "variable-size dimension in model output not supported"
                << std::endl;
      exit(1);
    }

    if (dim > 1) {
      non_one_cnt++;
      if (non_one_cnt > 1) {
        std::cerr << "expecting model output to be a vector" << std::endl;
        exit(1);
      }
    }
  }

  *format = input->Format();

  int max_batch_size = ctx->MaxBatchSize();

  // Model specifying maximum batch size of 0 indicates that batching
  // is not supported and so the input tensors do not expect a "N"
  // dimension (and 'batch_size' should be 1 so that only a single
  // image instance is inferred at a time).
  if (max_batch_size == 0) {
    if (batch_size != 1) {
      std::cerr << "batching not supported for model \"" << ctx->ModelName()
                << "\"" << std::endl;
      exit(1);
    }
  } else {
    // max_batch_size > 0
    if (batch_size > (size_t)max_batch_size) {
      std::cerr << "expecting batch size <= " << max_batch_size
                << " for model \"" << ctx->ModelName() << "\"" << std::endl;
      exit(1);
    }
  }

  if (input->Dims().size() != 3) {
    std::cerr << "expecting model input to have 3 dimensions, model \""
              << ctx->ModelName() << "\" input has " << input->Dims().size()
              << std::endl;
    exit(1);
  }

  // Variable-size dimensions are not currently supported.
  for (const auto dim : input->Dims()) {
    if (dim == -1) {
      std::cerr << "variable-size dimension in model input not supported"
                << std::endl;
      exit(1);
    }
  }

  // Input must be NHWC or NCHW...
  if ((*format != ni::ModelInput::FORMAT_NCHW) &&
      (*format != ni::ModelInput::FORMAT_NHWC)) {
    std::cerr << "unexpected input format "
              << ni::ModelInput_Format_Name(*format) << ", expecting "
              << ni::ModelInput_Format_Name(ni::ModelInput::FORMAT_NHWC)
              << " or "
              << ni::ModelInput_Format_Name(ni::ModelInput::FORMAT_NCHW)
              << std::endl;
    exit(1);
  }

  if (*format == ni::ModelInput::FORMAT_NHWC) {
    *h = input->Dims()[0];
    *w = input->Dims()[1];
    *c = input->Dims()[2];
  } else if (*format == ni::ModelInput::FORMAT_NCHW) {
    *c = input->Dims()[0];
    *h = input->Dims()[1];
    *w = input->Dims()[2];
  }

  if (!ParseType(input->DType(), type1, type3)) {
    std::cerr << "unexpected input datatype \""
              << ni::DataType_Name(input->DType()) << "\" for model \""
              << ctx->ModelName() << std::endl;
    exit(1);
  }
}

void
FileToInputData(
    const std::string& filename, size_t c, size_t h, size_t w,
    ni::ModelInput::Format format, int type1, int type3, ScaleType scale,
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

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  bool async = false;
  bool streaming = false;
  size_t batch_size = 1;
  size_t topk = 1;
  ScaleType scale = ScaleType::NONE;
  std::string preprocess_output_filename;
  std::string model_name;
  int64_t model_version = -1;
  std::string url("localhost:8000");
  ProtocolType protocol = ProtocolType::HTTP;
  std::map<std::string, std::string> http_headers;

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
        model_version = std::atoll(optarg);
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
  if (!http_headers.empty() && (protocol != ProtocolType::HTTP)) {
    std::cerr << "WARNING: HTTP headers specified with -H are ignored when "
                 "using non-HTTP protocol."
              << std::endl;
  }

  // Create the context for inference of the specified model. From it
  // extract and validate that the model meets the requirements for
  // image classification.
  std::unique_ptr<nic::InferContext> ctx;
  nic::Error err;
  if (streaming) {
    err = nic::InferGrpcStreamContext::Create(
        &ctx, url, model_name, model_version, verbose);
  } else if (protocol == ProtocolType::HTTP) {
    err = nic::InferHttpContext::Create(
        &ctx, url, http_headers, model_name, model_version, verbose);
  } else {
    err = nic::InferGrpcContext::Create(
        &ctx, url, model_name, model_version, verbose);
  }
  if (!err.IsOk()) {
    std::cerr << "error: unable to create inference context: " << err
              << std::endl;
    exit(1);
  }

  size_t c, h, w;
  ni::ModelInput::Format format;
  int type1, type3;
  ParseModel(ctx, batch_size, &c, &h, &w, &format, &type1, &type3, verbose);

  //

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
        fn, c, h, w, format, type1, type3, scale, &(image_data.back()));

    if ((image_data.size() == 1) && !preprocess_output_filename.empty()) {
      std::ofstream output_file(preprocess_output_filename);
      std::ostream_iterator<uint8_t> output_iterator(output_file);
      std::copy(image_data[0].begin(), image_data[0].end(), output_iterator);
    }
  }

  // Configure context for 'batch_size' and 'topk'
  std::unique_ptr<nic::InferContext::Options> options;
  err = nic::InferContext::Options::Create(&options);
  if (!err.IsOk()) {
    std::cerr << "failed initializing infer options: " << err << std::endl;
    exit(1);
  }

  options->SetBatchSize(batch_size);
  options->AddClassResult(ctx->Outputs()[0], topk);
  err = ctx->SetRunOptions(*options);
  if (!err.IsOk()) {
    std::cerr << "failed initializing batch size: " << err << std::endl;
    exit(1);
  }

  // Send requests of 'batch_size' images. If the number of images
  // isn't an exact multiple of 'batch_size' then just start over with
  // the first images until the batch is filled.
  //
  // Number of requests sent = ceil(number of images / batch_size)
  std::vector<std::map<std::string, std::unique_ptr<nic::InferContext::Result>>>
      results;
  std::vector<std::vector<std::string>> result_filenames;
  std::vector<std::shared_ptr<nic::InferContext::Request>> requests;
  size_t image_idx = 0;
  bool last_request = false;
  while (!last_request) {
    // Already verified that there is 1 input...
    const auto& input = ctx->Inputs()[0];

    // Reset the input for new request.
    err = input->Reset();
    if (!err.IsOk()) {
      std::cerr << "failed resetting input: " << err << std::endl;
      exit(1);
    }

    // Set input to be the next 'batch_size' images (preprocessed).
    std::vector<std::string> input_filenames;
    for (size_t idx = 0; idx < batch_size; ++idx) {
      input_filenames.push_back(image_filenames[image_idx]);
      err = input->SetRaw(image_data[image_idx]);
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

    // Send request.
    if (!async) {
      results.emplace_back();
      err = ctx->Run(&(results.back()));
      if (!err.IsOk()) {
        std::cerr << "failed sending synchronous infer request: " << err
                  << std::endl;
        exit(1);
      }
    } else {
      std::shared_ptr<nic::InferContext::Request> req;
      err = ctx->AsyncRun(&req);
      if (!err.IsOk()) {
        std::cerr << "failed sending asynchronous infer request: " << err
                  << std::endl;
        exit(1);
      }

      requests.emplace_back(std::move(req));
    }
  }

  // For async, retrieve results according to the send order
  if (async) {
    bool is_ready;
    for (auto& request : requests) {
      results.emplace_back();
      err =
          ctx->GetAsyncRunResults(&(results.back()), &is_ready, request, true);
      if (!err.IsOk()) {
        std::cerr << "failed receiving infer response: " << err << std::endl;
        exit(1);
      }
    }
  }

  // Post-process the results to make prediction(s)
  for (size_t idx = 0; idx < results.size(); idx++) {
    std::cout << "Request " << idx << ", batch size " << batch_size
              << std::endl;
    Postprocess(results[idx], result_filenames[idx], batch_size);
  }

  return 0;
}
