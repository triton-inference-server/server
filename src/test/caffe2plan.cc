// Copyright 2018-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <NvCaffeParser.h>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <errno.h>
#include <stddef.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace {

class Logger : public nvinfer1::ILogger {
 public:
  void log(
      nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
  {
    // Don't show INFO messages...
    if (severity == Severity::kINFO) {
      return;
    }

    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        std::cerr << "INFO: ";
        break;
      default:
        std::cerr << "UNKNOWN: ";
        break;
    }

    std::cerr << msg << std::endl;
  }
};

static Logger gLogger;

static const int CAL_BATCH_SIZE = 50;
static const int FIRST_CAL_BATCH = 0, NB_CAL_BATCHES = 10;

class BatchStream {
 public:
  BatchStream(const std::string& path, int batchSize, int maxBatches)
      : mPath(path), mBatchSize(batchSize), mMaxBatches(maxBatches)
  {
    const std::string cpath = mPath + "/batches/batch0";
    FILE* file = fopen(cpath.c_str(), "rb");
    if (file == nullptr) {
      std::cerr << "Can't open " << cpath << ": " << std::strerror(errno)
                << std::endl;
      exit(1);
    }

    int d[4];
    if (fread(d, sizeof(int), 4, file) != 4) {
      std::cerr << "Unexpected BatchStream failure" << std::endl;
      exit(1);
    }
    mDims = nvinfer1::Dims4{d[0], d[1], d[2], d[3]};
    fclose(file);
    mImageSize = mDims.d[1] * mDims.d[2] * mDims.d[3];
    mBatch.resize(mBatchSize * mImageSize, 0);
    mLabels.resize(mBatchSize, 0);
    mFileBatch.resize(mDims.d[0] * mImageSize, 0);
    mFileLabels.resize(mDims.d[0], 0);
    reset(0);
  }

  void reset(int firstBatch)
  {
    mBatchCount = 0;
    mFileCount = 0;
    mFileBatchPos = mDims.d[0];
    skip(firstBatch);
  }

  bool next()
  {
    if (mBatchCount == mMaxBatches)
      return false;

    for (int csize = 1, batchPos = 0; batchPos < mBatchSize;
         batchPos += csize, mFileBatchPos += csize) {
      if (mFileBatchPos == mDims.d[0] && !update())
        return false;

      // copy the smaller of: elements left to fulfill the request, or elements
      // left in the file buffer.
      csize = std::min(mBatchSize - batchPos, mDims.d[0] - mFileBatchPos);
      std::copy_n(
          getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize,
          getBatch() + batchPos * mImageSize);
      std::copy_n(
          getFileLabels() + mFileBatchPos, csize, getLabels() + batchPos);
    }
    mBatchCount++;
    return true;
  }

  void skip(int skipCount)
  {
    if (mBatchSize >= mDims.d[0] && mBatchSize % mDims.d[0] == 0 &&
        mFileBatchPos == mDims.d[0]) {
      mFileCount += skipCount * mBatchSize / mDims.d[0];
      return;
    }

    int x = mBatchCount;
    for (int i = 0; i < skipCount; i++) next();
    mBatchCount = x;
  }

  float* getBatch() { return &mBatch[0]; }
  float* getLabels() { return &mLabels[0]; }
  int getBatchesRead() const { return mBatchCount; }
  int getBatchSize() const { return mBatchSize; }
  nvinfer1::Dims4 getDims() const { return mDims; }

 private:
  float* getFileBatch() { return &mFileBatch[0]; }
  float* getFileLabels() { return &mFileLabels[0]; }

  bool update()
  {
    const std::string inputFileName =
        mPath + "/batches/batch" + std::to_string(mFileCount++);
    FILE* file = fopen(inputFileName.c_str(), "rb");
    if (!file)
      return false;

    int d[4];
    if (fread(d, sizeof(int), 4, file) != 4) {
      return false;
    }

    size_t readInputCount =
        fread(getFileBatch(), sizeof(float), mDims.d[0] * mImageSize, file);
    size_t readLabelCount =
        fread(getFileLabels(), sizeof(float), mDims.d[0], file);
    ;
    if ((readInputCount != size_t(mDims.d[0] * mImageSize)) ||
        (readLabelCount != size_t(mDims.d[0]))) {
      return false;
    }

    fclose(file);
    mFileBatchPos = 0;
    return true;
  }

  const std::string mPath;
  int mBatchSize{0};
  int mMaxBatches{0};
  int mBatchCount{0};

  int mFileCount{0}, mFileBatchPos{0};
  int mImageSize{0};

  nvinfer1::Dims4 mDims;
  std::vector<float> mBatch;
  std::vector<float> mLabels;
  std::vector<float> mFileBatch;
  std::vector<float> mFileLabels;
};

class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator {
 public:
  Int8EntropyCalibrator(
      BatchStream& stream, int firstBatch, bool readCache = true)
      : mStream(stream), mReadCache(readCache)
  {
    nvinfer1::Dims4 dims = mStream.getDims();
    mInputCount = mStream.getBatchSize() * dims.d[1] * dims.d[2] * dims.d[3];
    cudaMalloc(&mDeviceInput, mInputCount * sizeof(float));
    mStream.reset(firstBatch);
  }

  virtual ~Int8EntropyCalibrator() { cudaFree(mDeviceInput); }

  int getBatchSize() const noexcept override { return mStream.getBatchSize(); }

  bool getBatch(
      void* bindings[], const char* names[], int nbBindings) noexcept override
  {
    if (!mStream.next())
      return false;

    cudaMemcpy(
        mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float),
        cudaMemcpyHostToDevice);
    bindings[0] = mDeviceInput;
    return true;
  }

  const void* readCalibrationCache(size_t& length) noexcept override
  {
    mCalibrationCache.clear();
    std::ifstream input(calibrationTableName(), std::ios::binary);
    input >> std::noskipws;
    if (mReadCache && input.good())
      std::copy(
          std::istream_iterator<char>(input), std::istream_iterator<char>(),
          std::back_inserter(mCalibrationCache));

    length = mCalibrationCache.size();
    return length ? &mCalibrationCache[0] : nullptr;
  }

  void writeCalibrationCache(const void* cache, size_t length) noexcept override
  {
    std::ofstream output(calibrationTableName(), std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
  }

 private:
  static std::string calibrationTableName()
  {
    return std::string("CalibrationTable");
  }
  BatchStream mStream;
  bool mReadCache{true};

  size_t mInputCount;
  void* mDeviceInput{nullptr};
  std::vector<char> mCalibrationCache;
};

bool
CaffeToPlan(
    const std::string& output_filename, const std::string& prototxt_filename,
    const std::string& model_filename,
    const std::vector<std::string>& output_names,
    nvinfer1::DataType model_dtype, const std::string& calibration_filename,
    const size_t max_batch_size, const size_t max_workspace_size)
{
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();

  if ((model_dtype == nvinfer1::DataType::kINT8) &&
      !builder->platformHasFastInt8()) {
    std::cerr << "WARNING: GPU does not support int8, using fp32" << std::endl;
    model_dtype = nvinfer1::DataType::kFLOAT;
  } else if (
      (model_dtype == nvinfer1::DataType::kHALF) &&
      !builder->platformHasFastFp16()) {
    std::cerr << "WARNING: GPU does not support fp16, using fp32" << std::endl;
    model_dtype = nvinfer1::DataType::kFLOAT;
  }

  const nvcaffeparser1::IBlobNameToTensor* name_to_tensor = parser->parse(
      prototxt_filename.c_str(), model_filename.c_str(), *network,
      (model_dtype == nvinfer1::DataType::kINT8) ? nvinfer1::DataType::kFLOAT
                                                 : model_dtype);
  if (name_to_tensor == nullptr) {
    return false;
  }

  for (const auto& s : output_names) {
    network->markOutput(*name_to_tensor->find(s.c_str()));
  }

  builder->setMaxBatchSize(max_batch_size);
  config->setMaxWorkspaceSize(max_workspace_size);
  if (model_dtype == nvinfer1::DataType::kHALF) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  } else if (model_dtype == nvinfer1::DataType::kINT8) {
    BatchStream* calibrationStream =
        new BatchStream(calibration_filename, CAL_BATCH_SIZE, NB_CAL_BATCHES);
    Int8EntropyCalibrator* calibrator =
        new Int8EntropyCalibrator(*calibrationStream, FIRST_CAL_BATCH);
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    config->setInt8Calibrator(calibrator);
  }

  nvinfer1::IHostMemory* plan =
      builder->buildSerializedNetwork(*network, *config);
  if (plan == nullptr) {
    return false;
  }

  std::ofstream output(
      output_filename, std::ios::binary | std::ios::out | std::ios::app);
  output.write(reinterpret_cast<const char*>(plan->data()), plan->size());
  output.close();

  parser->destroy();
  network->destroy();
  builder->destroy();
  nvcaffeparser1::shutdownProtobufLibrary();

  return true;
}

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0]
            << " [options] <caffe prototxt> <caffe model>" << std::endl;
  std::cerr << "\t-b <max batch size>" << std::endl;
  std::cerr << "\t-w <max workspace byte size>" << std::endl;
  std::cerr << "\t-h" << std::endl;
  std::cerr << "\t-i <path to calibration data>" << std::endl;
  std::cerr << "\t-o <output PLAN filename>" << std::endl;
  std::cerr << "\t-n <model output name>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "-h, generate FP16 PLAN, default is false." << std::endl;
  std::cerr << "-i, generate INT8 PLAN, using given calibration data"
            << std::endl;
  std::cerr << "-n, may be specified multiple times." << std::endl;

  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
{
  size_t max_batch_size = 1;
  size_t max_workspace_size = 256 * 1024 * 1024;
  bool use_fp16 = false;
  bool use_int8 = false;
  std::string calibration_filename;
  std::string output_filename;
  std::vector<std::string> output_names;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "hb:w:o:n:i:")) != -1) {
    switch (opt) {
      case 'h':
        use_fp16 = true;
        break;
      case 'i':
        use_int8 = true;
        calibration_filename = optarg;
        break;
      case 'b':
        max_batch_size = atoi(optarg);
        break;
      case 'w':
        max_workspace_size = atoi(optarg);
        break;
      case 'o':
        output_filename = optarg;
        break;
      case 'n':
        output_names.push_back(optarg);
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  if (use_fp16 && use_int8) {
    Usage(argv, "can't specify both -h and -i flag");
  }
  if (output_filename.empty()) {
    Usage(argv, "-o flag must be specified");
  }
  if (output_names.empty()) {
    Usage(argv, "-n flag must be specified");
  }
  if ((optind + 1) >= argc) {
    Usage(argv, "prototxt and model must be specified");
  }

  std::string prototxt_filename = argv[optind];
  std::string model_filename = argv[optind + 1];

  if (!CaffeToPlan(
          output_filename, prototxt_filename, model_filename, output_names,
          (use_fp16) ? nvinfer1::DataType::kHALF
                     : (use_int8) ? nvinfer1::DataType::kINT8
                                  : nvinfer1::DataType::kFLOAT,
          calibration_filename, max_batch_size, max_workspace_size)) {
    std::cerr << "Failed to create PLAN file" << std::endl;
    return 1;
  }

  std::cout << "Success" << std::endl;
  return 0;
}
