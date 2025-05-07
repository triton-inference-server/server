// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>
#include <vector>

#include "triton/backend/backend_common.h"
#include "triton/core/tritonserver.h"

namespace tb = triton::backend;

namespace {

struct TritonServerError {
  TritonServerError(TRITONSERVER_Error_Code code, const char* msg)
      : code_(code), msg_(msg)
  {
  }
  TRITONSERVER_Error_Code code_;
  std::string msg_;
};

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

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

enum class ErrorCode {
  kInvalidDim = -2,
  kOverflow = -3,
};

void
assert_get_element_count_success(
    std::vector<int64_t>& shape, int64_t expected_cnt)
{
  int64_t cnt;
  TRITONSERVER_Error* err;

  // assert old APIs
  ASSERT_EQ(expected_cnt, tb::GetElementCount(shape.data(), shape.size()));
  ASSERT_EQ(expected_cnt, tb::GetElementCount(shape));

  // assert new APIs
  err = tb::GetElementCount(shape.data(), shape.size(), &cnt);
  ASSERT_EQ(err, nullptr);
  ASSERT_EQ(cnt, expected_cnt);
  err = tb::GetElementCount(shape, &cnt);
  ASSERT_EQ(err, nullptr);
  ASSERT_EQ(cnt, expected_cnt);
}

void
assert_get_element_count_error(
    std::vector<int64_t>& shape, ErrorCode error_code,
    const std::string& error_msg)
{
  int64_t cnt;
  TRITONSERVER_Error* err;

  // assert old APIs
  ASSERT_EQ(
      static_cast<int>(error_code),
      tb::GetElementCount(shape.data(), shape.size()));
  ASSERT_EQ(static_cast<int>(error_code), tb::GetElementCount(shape));

  // assert new APIs
  err = tb::GetElementCount(shape.data(), shape.size(), &cnt);
  ASSERT_NE(err, nullptr);
  ASSERT_EQ(TRITONSERVER_ERROR_INVALID_ARG, TRITONSERVER_ErrorCode(err));
  ASSERT_STREQ(error_msg.c_str(), TRITONSERVER_ErrorMessage(err));
  err = tb::GetElementCount(shape, &cnt);
  ASSERT_NE(err, nullptr);
  ASSERT_EQ(TRITONSERVER_ERROR_INVALID_ARG, TRITONSERVER_ErrorCode(err));
  ASSERT_STREQ(error_msg.c_str(), TRITONSERVER_ErrorMessage(err));
}

void
assert_get_byte_size_success(
    TRITONSERVER_DataType dtype, std::vector<int64_t>& shape,
    int64_t expected_size)
{
  int64_t size;
  TRITONSERVER_Error* err;

  // assert old API
  ASSERT_EQ(expected_size, tb::GetByteSize(dtype, shape));

  // assert new API
  err = tb::GetByteSize(dtype, shape, &size);
  ASSERT_EQ(err, nullptr);
  ASSERT_EQ(expected_size, size);
}

void
assert_get_byte_size_error(
    TRITONSERVER_DataType dtype, std::vector<int64_t>& shape,
    ErrorCode error_code, const std::string& error_msg)
{
  int64_t size;
  TRITONSERVER_Error* err;

  // assert old API
  ASSERT_EQ(static_cast<int>(error_code), tb::GetByteSize(dtype, shape));

  // assert new API
  err = tb::GetByteSize(dtype, shape, &size);
  ASSERT_NE(err, nullptr);
  ASSERT_EQ(TRITONSERVER_ERROR_INVALID_ARG, TRITONSERVER_ErrorCode(err));
  ASSERT_EQ(error_msg, TRITONSERVER_ErrorMessage(err));
}

class GetElementCountTest : public ::testing::Test {
 public:
  GetElementCountTest() {}
};

TEST_F(GetElementCountTest, GetElementCount)
{
  std::vector<int64_t> shape;
  int64_t expected_cnt;

  // Test 1: empty shape
  shape = {};
  expected_cnt = 0;
  assert_get_element_count_success(shape, expected_cnt);

  // Test 2: single dim
  shape = {8};
  expected_cnt = 8;
  assert_get_element_count_success(shape, expected_cnt);

  // Test 3: multiple dims
  shape = {1, 2, 3, 4};
  expected_cnt = 24;
  assert_get_element_count_success(shape, expected_cnt);
}

TEST_F(GetElementCountTest, GetElementCountNegative)
{
  std::vector<int64_t> shape;
  int64_t expected_cnt = -1;

  // Test 1: -1 dim
  shape = {-1};
  assert_get_element_count_success(shape, expected_cnt);

  // Test 2: one -1 dim
  shape = {-1, 8, 8};
  assert_get_element_count_success(shape, expected_cnt);

  // Test 3: multiple -1 dims
  shape = {8, -1, -1};
  assert_get_element_count_success(shape, expected_cnt);

  // Test 4: -1 dim before overflow
  shape = {-1, 1LL << 32, 1LL << 31};
  assert_get_element_count_success(shape, expected_cnt);
}

TEST_F(GetElementCountTest, GetElementCountZero)
{
  std::vector<int64_t> shape;
  int64_t expected_cnt = 0;

  // Test 1: 0 dim
  shape = {0};
  assert_get_element_count_success(shape, expected_cnt);

  // Test 2: one 0 dim
  shape = {1, 8, 0};
  assert_get_element_count_success(shape, expected_cnt);

  // Test 2: one 0 dim
  shape = {0, 1, 8};
  assert_get_element_count_success(shape, expected_cnt);

  // Test 3: multiple 0 dims
  shape = {8, 0, 0};
  assert_get_element_count_success(shape, expected_cnt);
}

TEST_F(GetElementCountTest, GetElementCountInvalidDim)
{
  std::vector<int64_t> shape;
  std::string error_msg;

  // Test 1: single invalid dim
  shape = {1, -2};
  error_msg = std::string("shape") + tb::ShapeToString(shape) +
              " contains an invalid dim.";
  assert_get_element_count_error(shape, ErrorCode::kInvalidDim, error_msg);

  // Test 2: multiple invalid dims
  shape = {1, -2, -3};
  error_msg = std::string("shape") + tb::ShapeToString(shape) +
              " contains an invalid dim.";
  assert_get_element_count_error(shape, ErrorCode::kInvalidDim, error_msg);

  // Test 3: valid but overflow dim
  shape = {1, 1LL << 63};
  error_msg = std::string("shape") + tb::ShapeToString(shape) +
              " contains an invalid dim.";
  assert_get_element_count_error(shape, ErrorCode::kInvalidDim, error_msg);
}

TEST_F(GetElementCountTest, GetElementCountOverflow)
{
  std::vector<int64_t> shape;
  std::string error_msg;

  // Test 1: no overflow
  shape = {1LL << 31, 1LL << 31};
  int64_t expected_cnt = 1LL << 62;
  assert_get_element_count_success(shape, expected_cnt);

  // Test 2: overflows
  shape = {1LL << 32, 1LL << 31};
  error_msg = "unexpected integer overflow while calculating element count.";
  assert_get_element_count_error(shape, ErrorCode::kOverflow, error_msg);

  // Test 3: overflows before -1 dim
  shape = {1LL << 32, 1LL << 31, -1};
  error_msg = "unexpected integer overflow while calculating element count.";
  assert_get_element_count_error(shape, ErrorCode::kOverflow, error_msg);
}

class GetByteSizeTest : public ::testing::Test {
 public:
  GetByteSizeTest() {}
};

TEST_F(GetByteSizeTest, GetByteSize)
{
  TRITONSERVER_DataType dtype = TRITONSERVER_TYPE_INT32;
  std::vector<int64_t> shape;
  int64_t expected_size;

  // Test 1: empty shape
  shape = {};
  expected_size = 0;
  assert_get_byte_size_success(dtype, shape, expected_size);

  // Test 2: single dim
  shape = {8};
  expected_size = 8 * TRITONSERVER_DataTypeByteSize(dtype);
  assert_get_byte_size_success(dtype, shape, expected_size);

  // Test 3: multiple dims
  shape = {1, 2, 3, 4};
  expected_size = 24 * TRITONSERVER_DataTypeByteSize(dtype);
  assert_get_byte_size_success(dtype, shape, expected_size);

  // Test 4: multiple dims with 0
  shape = {0, 1, 8};
  expected_size = 0;
  assert_get_byte_size_success(dtype, shape, expected_size);
}

TEST_F(GetByteSizeTest, GetByteSizeNegative)
{
  TRITONSERVER_DataType dtype;
  std::vector<int64_t> shape;
  int64_t expected_size = -1;

  // Test 1: invalid dtype
  dtype = TRITONSERVER_TYPE_INVALID;
  shape = {8, 8};
  assert_get_byte_size_success(dtype, shape, expected_size);

  // Test 2: bytes dtype
  dtype = TRITONSERVER_TYPE_BYTES;
  shape = {8, 8};
  assert_get_byte_size_success(dtype, shape, expected_size);

  // Test 3: invalid shape and element count overflows
  dtype = TRITONSERVER_TYPE_INVALID;
  shape = {1LL << 40, 1LL << 40};
  assert_get_byte_size_success(dtype, shape, expected_size);

  // Test 4: negative shape
  dtype = TRITONSERVER_TYPE_INT32;
  shape = {-1, 8};
  assert_get_byte_size_success(dtype, shape, expected_size);
}

TEST_F(GetByteSizeTest, GetByteSizeZero)
{
  TRITONSERVER_DataType dtype = TRITONSERVER_TYPE_INT32;
  std::vector<int64_t> shape;
  int64_t expected_cnt = 0;

  // Test 1: 0 dim
  shape = {0};
  assert_get_byte_size_success(dtype, shape, expected_cnt);

  // Test 2: one 0 dim
  shape = {1, 8, 0};
  assert_get_byte_size_success(dtype, shape, expected_cnt);

  // Test 2: one 0 dim
  shape = {0, 1, 8};
  assert_get_byte_size_success(dtype, shape, expected_cnt);

  // Test 3: multiple 0 dims
  shape = {8, 0, 0};
  assert_get_byte_size_success(dtype, shape, expected_cnt);
}

TEST_F(GetByteSizeTest, GetByteSizeInvalidDim)
{
  TRITONSERVER_DataType dtype = TRITONSERVER_TYPE_INT32;
  std::vector<int64_t> shape;
  std::string error_msg;

  // Test 1: single invalid dim
  shape = {1, -2};
  error_msg = std::string("shape") + tb::ShapeToString(shape) +
              " contains an invalid dim.";
  assert_get_byte_size_error(dtype, shape, ErrorCode::kInvalidDim, error_msg);

  // Test 2: multiple invalid dims
  shape = {1, -2, -3};
  error_msg = std::string("shape") + tb::ShapeToString(shape) +
              " contains an invalid dim.";
  assert_get_byte_size_error(dtype, shape, ErrorCode::kInvalidDim, error_msg);

  // Test 3: valid but overflow dim
  shape = {1, 1LL << 63};
  error_msg = std::string("shape") + tb::ShapeToString(shape) +
              " contains an invalid dim.";
  assert_get_byte_size_error(dtype, shape, ErrorCode::kInvalidDim, error_msg);
}

TEST_F(GetByteSizeTest, GetByteSizeOverflow)
{
  TRITONSERVER_DataType dtype = TRITONSERVER_TYPE_INT32;
  std::vector<int64_t> shape;
  std::string error_msg;

  // Test 1: no overflow
  shape = {1LL << 30, 1LL << 30};
  int64_t expected_size = (1LL << 60) * TRITONSERVER_DataTypeByteSize(dtype);
  assert_get_byte_size_success(dtype, shape, expected_size);

  // Test 2: element count overflows
  shape = {1LL << 32, 1LL << 31};
  error_msg = "unexpected integer overflow while calculating byte size.";
  assert_get_byte_size_error(dtype, shape, ErrorCode::kOverflow, error_msg);

  // Test 3: valid element count but byte size overflows
  shape = {1LL << 31, 1LL << 30};
  error_msg = "unexpected integer overflow while calculating byte size.";
  assert_get_byte_size_error(dtype, shape, ErrorCode::kOverflow, error_msg);
}
}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
