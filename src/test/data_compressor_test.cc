// Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gtest/gtest.h"

// Undefine the FAIL() macro inside Triton code to avoid redefine error
// from gtest. Okay as FAIL() is not used in data_compressor
#ifdef FAIL
#undef FAIL
#endif

#include <event2/buffer.h>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <future>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include "data_compressor.h"

namespace ni = triton::server;

namespace {

struct TritonServerError {
  TritonServerError(TRITONSERVER_Error_Code code, const char* msg)
      : code_(code), msg_(msg)
  {
  }
  TRITONSERVER_Error_Code code_;
  std::string msg_;
};

void
WriteEVBufferToFile(const std::string& file_name, evbuffer* evb)
{
  std::ofstream fs(file_name);
  struct evbuffer_iovec* buffer_array = nullptr;
  int buffer_count = evbuffer_peek(evb, -1, NULL, NULL, 0);
  if (buffer_count > 0) {
    buffer_array = static_cast<struct evbuffer_iovec*>(
        alloca(sizeof(struct evbuffer_iovec) * buffer_count));
    ASSERT_EQ(
        evbuffer_peek(evb, -1, NULL, buffer_array, buffer_count), buffer_count)
        << "unexpected error getting buffers for result";
  }
  for (int idx = 0; idx < buffer_count; ++idx) {
    fs.write(
        reinterpret_cast<const char*>(buffer_array[idx].iov_base),
        buffer_array[idx].iov_len);
  }
}

void
EVBufferToContiguousBuffer(evbuffer* evb, std::vector<char>* buffer)
{
  *buffer = std::vector<char>(evbuffer_get_length(evb));
  {
    struct evbuffer_iovec* buffer_array = nullptr;
    int buffer_count = evbuffer_peek(evb, -1, NULL, NULL, 0);
    if (buffer_count > 0) {
      buffer_array = static_cast<struct evbuffer_iovec*>(
          alloca(sizeof(struct evbuffer_iovec) * buffer_count));
      ASSERT_EQ(
          evbuffer_peek(evb, -1, NULL, buffer_array, buffer_count),
          buffer_count)
          << "unexpected error getting buffers for result";
    }
    size_t offset = 0;
    for (int idx = 0; idx < buffer_count; ++idx) {
      memcpy(
          buffer->data() + offset, buffer_array[idx].iov_base,
          buffer_array[idx].iov_len);
      offset += buffer_array[idx].iov_len;
    }
  }
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

TRITONSERVER_Error*
TRITONSERVER_ErrorNew(TRITONSERVER_Error_Code code, const char* msg)
{
  return reinterpret_cast<TRITONSERVER_Error*>(
      new TritonServerError(code, msg));
}

TRITONSERVER_Error_Code
TRITONSERVER_ErrorCode(TRITONSERVER_Error* error)
{
  return (reinterpret_cast<TritonServerError*>(error))->code_;
}

const char*
TRITONSERVER_ErrorMessage(TRITONSERVER_Error* error)
{
  return (reinterpret_cast<TritonServerError*>(error))->msg_.c_str();
}

#ifdef __cplusplus
}
#endif

namespace {

class DataCompressorTest : public ::testing::Test {
 public:
  DataCompressorTest()
      : raw_data_length_(0), deflate_compressed_length_(0),
        gzip_compressed_length_(0)
  {
    std::vector<std::string> files{"raw_data", "deflate_compressed_data",
                                   "gzip_compressed_data"};
    for (const auto& file : files) {
      std::fstream fs(file);
      // get length of file
      fs.seekg(0, fs.end);
      int length = fs.tellg();
      fs.seekg(0, fs.beg);

      // allocate memory
      char* data = nullptr;
      if (file == "raw_data") {
        raw_data_.reset(new char[length]);
        data = raw_data_.get();
        raw_data_length_ = length;
      } else if (file == "deflate_compressed_data") {
        deflate_compressed_data_.reset(new char[length]);
        data = deflate_compressed_data_.get();
        deflate_compressed_length_ = length;
      } else {
        gzip_compressed_data_.reset(new char[length]);
        data = gzip_compressed_data_.get();
        gzip_compressed_length_ = length;
      }

      fs.read(data, length);
    }
  }

  std::unique_ptr<char[]> raw_data_;
  size_t raw_data_length_;
  std::unique_ptr<char[]> deflate_compressed_data_;
  size_t deflate_compressed_length_;
  std::unique_ptr<char[]> gzip_compressed_data_;
  size_t gzip_compressed_length_;
};

TEST_F(DataCompressorTest, DeflateOneBuffer)
{
  // Convert the raw data into evbuffer format
  auto source = evbuffer_new();
  ASSERT_TRUE((source != nullptr)) << "Failed to create source evbuffer";
  ASSERT_EQ(evbuffer_add(source, raw_data_.get(), raw_data_length_), 0)
      << "Failed to initialize source evbuffer";

  auto compressed = evbuffer_new();
  ASSERT_TRUE((compressed != nullptr))
      << "Failed to create compressed evbuffer";
  auto decompressed = evbuffer_new();
  ASSERT_TRUE((decompressed != nullptr))
      << "Failed to create decompressed evbuffer";

  auto err = ni::DataCompressor::CompressData(
      ni::DataCompressor::Type::DEFLATE, source, compressed);
  ASSERT_TRUE((err == nullptr))
      << "Failed to compress data: " << TRITONSERVER_ErrorMessage(err);

  err = ni::DataCompressor::DecompressData(
      ni::DataCompressor::Type::DEFLATE, compressed, decompressed);
  ASSERT_TRUE((err == nullptr))
      << "Failed to decompress data: " << TRITONSERVER_ErrorMessage(err);

  size_t destination_byte_size = evbuffer_get_length(decompressed);
  ASSERT_EQ(destination_byte_size, raw_data_length_) << "Mismatched byte size";

  std::vector<char> res;
  EVBufferToContiguousBuffer(decompressed, &res);
  for (size_t idx = 0; idx < raw_data_length_; ++idx) {
    ASSERT_TRUE(raw_data_[idx] == res[idx]);
  }
}

TEST_F(DataCompressorTest, GzipOneBuffer)
{
  // Convert the raw data into evbuffer format
  auto source = evbuffer_new();
  ASSERT_TRUE((source != nullptr)) << "Failed to create source evbuffer";
  ASSERT_EQ(evbuffer_add(source, raw_data_.get(), raw_data_length_), 0)
      << "Failed to initialize source evbuffer";

  auto compressed = evbuffer_new();
  ASSERT_TRUE((compressed != nullptr))
      << "Failed to create compressed evbuffer";
  auto decompressed = evbuffer_new();
  ASSERT_TRUE((decompressed != nullptr))
      << "Failed to create decompressed evbuffer";

  auto err = ni::DataCompressor::CompressData(
      ni::DataCompressor::Type::GZIP, source, compressed);
  ASSERT_TRUE((err == nullptr))
      << "Failed to compress data: " << TRITONSERVER_ErrorMessage(err);
  err = ni::DataCompressor::DecompressData(
      ni::DataCompressor::Type::GZIP, compressed, decompressed);
  ASSERT_TRUE((err == nullptr))
      << "Failed to decompress data: " << TRITONSERVER_ErrorMessage(err);

  size_t destination_byte_size = evbuffer_get_length(decompressed);
  ASSERT_EQ(destination_byte_size, raw_data_length_) << "Mismatched byte size";

  std::vector<char> res;
  EVBufferToContiguousBuffer(decompressed, &res);
  for (size_t idx = 0; idx < raw_data_length_; ++idx) {
    ASSERT_TRUE(raw_data_[idx] == res[idx]);
  }
}

TEST_F(DataCompressorTest, DeflateTwoBuffer)
{
  // Convert the raw data into evbuffer format with two buffers
  auto source = evbuffer_new();
  ASSERT_TRUE((source != nullptr)) << "Failed to create source evbuffer";
  size_t half_length = raw_data_length_ / 2;
  ASSERT_EQ(evbuffer_add(source, raw_data_.get(), half_length), 0)
      << "Failed to initialize source evbuffer";
  // verify evbuffer has two extend
  {
    auto second_source = evbuffer_new();
    ASSERT_EQ(
        evbuffer_add(
            second_source, raw_data_.get() + half_length,
            raw_data_length_ - half_length),
        0)
        << "Failed to initialize source evbuffer";
    ASSERT_EQ(evbuffer_add_buffer(source, second_source), 0)
        << "Failed to initialize source evbuffer";
    int buffer_count = evbuffer_peek(source, -1, NULL, NULL, 0);
    ASSERT_EQ(buffer_count, 2) << "Expect two buffers as source";
  }

  auto compressed = evbuffer_new();
  ASSERT_TRUE((compressed != nullptr))
      << "Failed to create compressed evbuffer";
  // Reconstruct the compressed buffer to be two buffers
  if (evbuffer_peek(compressed, -1, NULL, NULL, 0) == 1) {
    struct evbuffer_iovec* buffer_array = nullptr;
    int buffer_count = evbuffer_peek(compressed, -1, NULL, NULL, 0);
    if (buffer_count > 0) {
      buffer_array = static_cast<struct evbuffer_iovec*>(
          alloca(sizeof(struct evbuffer_iovec) * buffer_count));
      ASSERT_EQ(
          evbuffer_peek(compressed, -1, NULL, buffer_array, buffer_count),
          buffer_count)
          << "unexpected error getting buffers for result";
    }

    auto first_compressed = evbuffer_new();
    auto second_compressed = evbuffer_new();
    size_t half_length = buffer_array[0].iov_len / 2;
    ASSERT_EQ(
        evbuffer_add(first_compressed, buffer_array[0].iov_base, half_length),
        0)
        << "Failed to split compressed buffer";
    ASSERT_EQ(
        evbuffer_add(
            second_compressed,
            reinterpret_cast<char*>(buffer_array[0].iov_base) + half_length,
            buffer_array[0].iov_len - half_length),
        0)
        << "Failed to split compressed buffer";
    ASSERT_EQ(evbuffer_add_buffer(first_compressed, second_compressed), 0)
        << "Failed to initialize source evbuffer";
    compressed = first_compressed;
  }

  auto decompressed = evbuffer_new();
  ASSERT_TRUE((decompressed != nullptr))
      << "Failed to create decompressed evbuffer";

  auto err = ni::DataCompressor::CompressData(
      ni::DataCompressor::Type::DEFLATE, source, compressed);
  ASSERT_TRUE((err == nullptr))
      << "Failed to compress data: " << TRITONSERVER_ErrorMessage(err);
  err = ni::DataCompressor::DecompressData(
      ni::DataCompressor::Type::DEFLATE, compressed, decompressed);
  ASSERT_TRUE((err == nullptr))
      << "Failed to decompress data: " << TRITONSERVER_ErrorMessage(err);

  size_t destination_byte_size = evbuffer_get_length(decompressed);
  ASSERT_EQ(destination_byte_size, raw_data_length_) << "Mismatched byte size";

  std::vector<char> res;
  EVBufferToContiguousBuffer(decompressed, &res);
  for (size_t idx = 0; idx < raw_data_length_; ++idx) {
    ASSERT_TRUE(raw_data_[idx] == res[idx]);
  }
}

TEST_F(DataCompressorTest, GzipTwoBuffer)
{
  // Convert the raw data into evbuffer format with two buffers
  auto source = evbuffer_new();
  ASSERT_TRUE((source != nullptr)) << "Failed to create source evbuffer";
  size_t half_length = raw_data_length_ / 2;
  ASSERT_EQ(evbuffer_add(source, raw_data_.get(), half_length), 0)
      << "Failed to initialize source evbuffer";
  // verify evbuffer has two extend
  {
    auto second_source = evbuffer_new();
    ASSERT_EQ(
        evbuffer_add(
            second_source, raw_data_.get() + half_length,
            raw_data_length_ - half_length),
        0)
        << "Failed to initialize source evbuffer";
    ASSERT_EQ(evbuffer_add_buffer(source, second_source), 0)
        << "Failed to initialize source evbuffer";
    int buffer_count = evbuffer_peek(source, -1, NULL, NULL, 0);
    ASSERT_EQ(buffer_count, 2) << "Expect two buffers as source";
  }

  auto compressed = evbuffer_new();
  ASSERT_TRUE((compressed != nullptr))
      << "Failed to create compressed evbuffer";
  // Reconstruct the compressed buffer to be two buffers
  if (evbuffer_peek(compressed, -1, NULL, NULL, 0) == 1) {
    struct evbuffer_iovec* buffer_array = nullptr;
    int buffer_count = evbuffer_peek(compressed, -1, NULL, NULL, 0);
    if (buffer_count > 0) {
      buffer_array = static_cast<struct evbuffer_iovec*>(
          alloca(sizeof(struct evbuffer_iovec) * buffer_count));
      ASSERT_EQ(
          evbuffer_peek(compressed, -1, NULL, buffer_array, buffer_count),
          buffer_count)
          << "unexpected error getting buffers for result";
    }

    auto first_compressed = evbuffer_new();
    auto second_compressed = evbuffer_new();
    size_t half_length = buffer_array[0].iov_len / 2;
    ASSERT_EQ(
        evbuffer_add(first_compressed, buffer_array[0].iov_base, half_length),
        0)
        << "Failed to split compressed buffer";
    ASSERT_EQ(
        evbuffer_add(
            second_compressed,
            reinterpret_cast<char*>(buffer_array[0].iov_base) + half_length,
            buffer_array[0].iov_len - half_length),
        0)
        << "Failed to split compressed buffer";
    ASSERT_EQ(evbuffer_add_buffer(first_compressed, second_compressed), 0)
        << "Failed to initialize source evbuffer";
    compressed = first_compressed;
  }

  auto decompressed = evbuffer_new();
  ASSERT_TRUE((decompressed != nullptr))
      << "Failed to create decompressed evbuffer";

  auto err = ni::DataCompressor::CompressData(
      ni::DataCompressor::Type::GZIP, source, compressed);
  ASSERT_TRUE((err == nullptr))
      << "Failed to compress data: " << TRITONSERVER_ErrorMessage(err);
  err = ni::DataCompressor::DecompressData(
      ni::DataCompressor::Type::GZIP, compressed, decompressed);
  ASSERT_TRUE((err == nullptr))
      << "Failed to decompress data: " << TRITONSERVER_ErrorMessage(err);

  size_t destination_byte_size = evbuffer_get_length(decompressed);
  ASSERT_EQ(destination_byte_size, raw_data_length_) << "Mismatched byte size";

  std::vector<char> res;
  EVBufferToContiguousBuffer(decompressed, &res);
  for (size_t idx = 0; idx < raw_data_length_; ++idx) {
    ASSERT_TRUE(raw_data_[idx] == res[idx]);
  }
}

TEST_F(DataCompressorTest, DeflateOneLargeBuffer)
{
  // Duplicate raw data 2^20 times
  {
    std::unique_ptr<char[]> extended_raw_data(
        new char[raw_data_length_ * (1 << 20)]);
    memcpy(extended_raw_data.get(), raw_data_.get(), raw_data_length_);
    size_t filled_size = raw_data_length_;
    for (size_t i = 1; i < 20; ++i) {
      memcpy(
          extended_raw_data.get() + filled_size, extended_raw_data.get(),
          filled_size);
      filled_size += filled_size;
    }
    raw_data_length_ = filled_size;
    raw_data_.swap(extended_raw_data);
  }
  // Convert the raw data into evbuffer format
  auto source = evbuffer_new();
  ASSERT_TRUE((source != nullptr)) << "Failed to create source evbuffer";
  ASSERT_EQ(evbuffer_add(source, raw_data_.get(), raw_data_length_), 0)
      << "Failed to initialize source evbuffer";

  auto compressed = evbuffer_new();
  ASSERT_TRUE((compressed != nullptr))
      << "Failed to create compressed evbuffer";
  ASSERT_GE(raw_data_length_ / 2, evbuffer_get_length(compressed))
      << "Compression should be desired for large data";

  auto decompressed = evbuffer_new();
  ASSERT_TRUE((decompressed != nullptr))
      << "Failed to create decompressed evbuffer";

  auto err = ni::DataCompressor::CompressData(
      ni::DataCompressor::Type::DEFLATE, source, compressed);
  ASSERT_TRUE((err == nullptr))
      << "Failed to compress data: " << TRITONSERVER_ErrorMessage(err);

  err = ni::DataCompressor::DecompressData(
      ni::DataCompressor::Type::DEFLATE, compressed, decompressed);
  ASSERT_TRUE((err == nullptr))
      << "Failed to decompress data: " << TRITONSERVER_ErrorMessage(err);
  size_t destination_byte_size = evbuffer_get_length(decompressed);
  ASSERT_EQ(destination_byte_size, raw_data_length_) << "Mismatched byte size";

  std::vector<char> res;
  EVBufferToContiguousBuffer(decompressed, &res);
  for (size_t idx = 0; idx < raw_data_length_; ++idx) {
    ASSERT_TRUE(raw_data_[idx] == res[idx]);
  }
}

TEST_F(DataCompressorTest, GzipOneLargeBuffer)
{
  // Duplicate raw data 2^20 times
  {
    std::unique_ptr<char[]> extended_raw_data(
        new char[raw_data_length_ * (1 << 20)]);
    memcpy(extended_raw_data.get(), raw_data_.get(), raw_data_length_);
    size_t filled_size = raw_data_length_;
    for (size_t i = 1; i < 20; ++i) {
      memcpy(
          extended_raw_data.get() + filled_size, extended_raw_data.get(),
          filled_size);
      filled_size += filled_size;
    }
    raw_data_length_ = filled_size;
    raw_data_.swap(extended_raw_data);
  }
  // Convert the raw data into evbuffer format
  auto source = evbuffer_new();
  ASSERT_TRUE((source != nullptr)) << "Failed to create source evbuffer";
  ASSERT_EQ(evbuffer_add(source, raw_data_.get(), raw_data_length_), 0)
      << "Failed to initialize source evbuffer";

  auto compressed = evbuffer_new();
  ASSERT_TRUE((compressed != nullptr))
      << "Failed to create compressed evbuffer";
  ASSERT_GE(raw_data_length_ / 2, evbuffer_get_length(compressed))
      << "Compression should be desired for large data";

  auto decompressed = evbuffer_new();
  ASSERT_TRUE((decompressed != nullptr))
      << "Failed to create decompressed evbuffer";

  auto err = ni::DataCompressor::CompressData(
      ni::DataCompressor::Type::GZIP, source, compressed);
  ASSERT_TRUE((err == nullptr))
      << "Failed to compress data: " << TRITONSERVER_ErrorMessage(err);
  err = ni::DataCompressor::DecompressData(
      ni::DataCompressor::Type::GZIP, compressed, decompressed);
  ASSERT_TRUE((err == nullptr))
      << "Failed to decompress data: " << TRITONSERVER_ErrorMessage(err);

  size_t destination_byte_size = evbuffer_get_length(decompressed);
  ASSERT_EQ(destination_byte_size, raw_data_length_) << "Mismatched byte size";

  std::vector<char> res;
  EVBufferToContiguousBuffer(decompressed, &res);
  for (size_t idx = 0; idx < raw_data_length_; ++idx) {
    ASSERT_TRUE(raw_data_[idx] == res[idx]);
  }
}

TEST_F(DataCompressorTest, DecompressDeflateBuffer)
{
  // Convert the compressed data into evbuffer format
  auto source = evbuffer_new();
  ASSERT_TRUE((source != nullptr)) << "Failed to create source evbuffer";
  ASSERT_EQ(
      evbuffer_add(
          source, deflate_compressed_data_.get(), deflate_compressed_length_),
      0)
      << "Failed to initialize source evbuffer";
  auto decompressed = evbuffer_new();
  ASSERT_TRUE((decompressed != nullptr))
      << "Failed to create decompressed evbuffer";

  auto err = ni::DataCompressor::DecompressData(
      ni::DataCompressor::Type::DEFLATE, source, decompressed);
  ASSERT_TRUE((err == nullptr))
      << "Failed to decompress data: " << TRITONSERVER_ErrorMessage(err);

  size_t destination_byte_size = evbuffer_get_length(decompressed);
  ASSERT_EQ(destination_byte_size, raw_data_length_) << "Mismatched byte size";

  std::vector<char> res;
  EVBufferToContiguousBuffer(decompressed, &res);
  for (size_t idx = 0; idx < raw_data_length_; ++idx) {
    ASSERT_TRUE(raw_data_[idx] == res[idx]);
  }
}

TEST_F(DataCompressorTest, DecompressGzipBuffer)
{
  // Convert the compressed data into evbuffer format
  auto source = evbuffer_new();
  ASSERT_TRUE((source != nullptr)) << "Failed to create source evbuffer";
  ASSERT_EQ(
      evbuffer_add(
          source, gzip_compressed_data_.get(), gzip_compressed_length_),
      0)
      << "Failed to initialize source evbuffer";
  auto decompressed = evbuffer_new();
  ASSERT_TRUE((decompressed != nullptr))
      << "Failed to create decompressed evbuffer";

  auto err = ni::DataCompressor::DecompressData(
      ni::DataCompressor::Type::GZIP, source, decompressed);
  ASSERT_TRUE((err == nullptr))
      << "Failed to decompress data: " << TRITONSERVER_ErrorMessage(err);

  size_t destination_byte_size = evbuffer_get_length(decompressed);
  ASSERT_EQ(destination_byte_size, raw_data_length_) << "Mismatched byte size";

  std::vector<char> res;
  EVBufferToContiguousBuffer(decompressed, &res);
  for (size_t idx = 0; idx < raw_data_length_; ++idx) {
    ASSERT_TRUE(raw_data_[idx] == res[idx]);
  }
}

TEST_F(DataCompressorTest, CompressDeflateBuffer)
{
  // Convert the raw data into evbuffer format
  auto source = evbuffer_new();
  ASSERT_TRUE((source != nullptr)) << "Failed to create source evbuffer";
  ASSERT_EQ(evbuffer_add(source, raw_data_.get(), raw_data_length_), 0)
      << "Failed to initialize source evbuffer";

  auto compressed = evbuffer_new();
  ASSERT_TRUE((compressed != nullptr))
      << "Failed to create compressed evbuffer";

  auto err = ni::DataCompressor::CompressData(
      ni::DataCompressor::Type::DEFLATE, source, compressed);
  ASSERT_TRUE((err == nullptr))
      << "Failed to compress data: " << TRITONSERVER_ErrorMessage(err);

  // Write compressed data to file which will be validated by other compression
  // tool
  WriteEVBufferToFile("generated_deflate_compressed_data", compressed);
}

TEST_F(DataCompressorTest, CompressGzipBuffer)
{
  // Convert the raw data into evbuffer format
  auto source = evbuffer_new();
  ASSERT_TRUE((source != nullptr)) << "Failed to create source evbuffer";
  ASSERT_EQ(evbuffer_add(source, raw_data_.get(), raw_data_length_), 0)
      << "Failed to initialize source evbuffer";

  auto compressed = evbuffer_new();
  ASSERT_TRUE((compressed != nullptr))
      << "Failed to create compressed evbuffer";

  auto err = ni::DataCompressor::CompressData(
      ni::DataCompressor::Type::GZIP, source, compressed);
  ASSERT_TRUE((err == nullptr))
      << "Failed to compress data: " << TRITONSERVER_ErrorMessage(err);

  // Write compressed data to file which will be validated by other compression
  // tool
  WriteEVBufferToFile("generated_gzip_compressed_data", compressed);
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
