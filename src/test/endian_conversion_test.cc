// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// from gtest. Okay as FAIL() is not used in little endian conversion functions
#ifdef FAIL
#undef FAIL
#endif

// Tests written to operate assuming big endian byte_order

#undef __BYTE_ORDER__
#define __BYTE_ORDER__ __ORDER_BIG_ENDIAN__

#include <algorithm>

#include "common.h"

namespace triton { namespace server {

template<typename T>
void ConversionTest(TRITONSERVER_DataType datatype,
	       const T &original_values,
	       const T &swapped_values) {

  T current_values;

  char *current_values_base = reinterpret_cast<char*>(&current_values[0]);
  size_t byte_size = sizeof(current_values);
    
  std::copy(std::begin(original_values), std::end(original_values), std::begin(current_values));

  HostToLittleEndian(datatype, current_values_base, byte_size);

  for (int i = 0; i < current_values.size(); ++i) {
    EXPECT_EQ(current_values[i], swapped_values[i]);
  }

  LittleEndianToHost(datatype, current_values_base, byte_size);

  for (int i = 0; i < current_values.size(); ++i) {
    EXPECT_EQ(current_values[i], original_values[i]);
  }
  
  // Test Non Contiguous Arrays

  for (int num_partial_arrays = 1; num_partial_arrays <= byte_size;
       num_partial_arrays++) {

    size_t partial_byte_size = byte_size / num_partial_arrays;

    std::copy(
        std::begin(swapped_values), std::end(swapped_values),
        std::begin(current_values));

    std::vector<char*> partial_result;
    size_t next_offset = 0;
    char* partial_array = current_values_base;
    
    for (int partial_array_offset = 0; partial_array_offset < byte_size;
         partial_array_offset += partial_byte_size) {
      size_t current_byte_size = std::min(
          partial_byte_size, (byte_size - partial_array_offset));
      LittleEndianToHost(
          datatype, partial_array, current_byte_size,
          partial_result, next_offset);
      partial_array += current_byte_size;
    }

    for (int i = 0; i < current_values.size(); i++) {
      EXPECT_EQ(current_values[i], original_values[i]);
    }
  }
}


TEST(EndianConversionTest, ConvertBYTES)
{
  // BYTES array are prepended by size of content bytes
  // That is the only portion which needs to switch byte order

  const std::array<uint32_t, 10> original_values = {
      0x00000004, 0xdeadbeef, 0x00000008, 0xdeadbeef, 0xdeadbeef,
      0x0000000C, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0x00000000};


  const std::array<uint32_t, 10> swapped_values = {
      0x04000000, 0xdeadbeef, 0x08000000, 0xdeadbeef, 0xdeadbeef,
      0x0C000000, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0x00000000};

  ConversionTest(TRITONSERVER_TYPE_BYTES, original_values, swapped_values);
}

TEST(EndianConversionTest, ConvertUINT32)
{
  const std::array<uint32_t, 10> original_values = {
      0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef,
      0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef};

  const std::array<uint32_t, 10> swapped_values = {
      0xefbeadde, 0xefbeadde, 0xefbeadde, 0xefbeadde, 0xefbeadde,
      0xefbeadde, 0xefbeadde, 0xefbeadde, 0xefbeadde, 0xefbeadde};

  ConversionTest(TRITONSERVER_TYPE_UINT32, original_values, swapped_values);
}

TEST(EndianConversionTest, ConvertFP32)
{
  std::array<uint32_t, 10> original_values_temp = {
      0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef,
      0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef};

  const std::array<float, 10>* original_values =
      reinterpret_cast<std::array<float, 10>*>(&original_values_temp[0]);

  std::array<uint32_t, 10> swapped_values_temp = {
      0xefbeadde, 0xefbeadde, 0xefbeadde, 0xefbeadde, 0xefbeadde,
      0xefbeadde, 0xefbeadde, 0xefbeadde, 0xefbeadde, 0xefbeadde};

  const std::array<float, 10>* swapped_values =
      reinterpret_cast<std::array<float, 10>*>(&swapped_values_temp[0]);


  ConversionTest(TRITONSERVER_TYPE_FP32, *original_values, *swapped_values);
}

TEST(EndianConversionTest, ConvertUINT16)
{
  const std::array<uint16_t, 10> original_values = {
      0xdead, 0xdead, 0xdead, 0xdead, 0xdead,
      0xdead, 0xdead, 0xdead, 0xdead, 0xdead};

  const std::array<uint16_t, 10> swapped_values = {
      0xadde, 0xadde, 0xadde, 0xadde, 0xadde,
      0xadde, 0xadde, 0xadde, 0xadde, 0xadde};

  ConversionTest(TRITONSERVER_TYPE_UINT16, original_values, swapped_values);
}

TEST(EndianConversionTest, ConvertUINT64)
{
  const std::array<uint64_t, 10> original_values = {
      0xdeadbeefdeadbeef, 0xdeadbeefdeadbeef, 0xdeadbeefdeadbeef,
      0xdeadbeefdeadbeef, 0xdeadbeefdeadbeef, 0xdeadbeefdeadbeef,
      0xdeadbeefdeadbeef, 0xdeadbeefdeadbeef, 0xdeadbeefdeadbeef,
      0xdeadbeefdeadbeef};

  const std::array<uint64_t, 10> swapped_values = {
      0xefbeaddeefbeadde, 0xefbeaddeefbeadde, 0xefbeaddeefbeadde,
      0xefbeaddeefbeadde, 0xefbeaddeefbeadde, 0xefbeaddeefbeadde,
      0xefbeaddeefbeadde, 0xefbeaddeefbeadde, 0xefbeaddeefbeadde,
      0xefbeaddeefbeadde};

  ConversionTest(TRITONSERVER_TYPE_UINT64, original_values, swapped_values);
}

TEST(EndianConversionTest, ConvertFP64)
{
  const std::array<uint64_t, 10> original_values_temp = {
      0xdeadbeefdeadbeef, 0xdeadbeefdeadbeef, 0xdeadbeefdeadbeef,
      0xdeadbeefdeadbeef, 0xdeadbeefdeadbeef, 0xdeadbeefdeadbeef,
      0xdeadbeefdeadbeef, 0xdeadbeefdeadbeef, 0xdeadbeefdeadbeef,
      0xdeadbeefdeadbeef};

  const std::array<uint64_t, 10> swapped_values_temp = {
      0xefbeaddeefbeadde, 0xefbeaddeefbeadde, 0xefbeaddeefbeadde,
      0xefbeaddeefbeadde, 0xefbeaddeefbeadde, 0xefbeaddeefbeadde,
      0xefbeaddeefbeadde, 0xefbeaddeefbeadde, 0xefbeaddeefbeadde,
      0xefbeaddeefbeadde};

  const std::array<double, 10>* original_values =
      reinterpret_cast<const std::array<double, 10>*>(&original_values_temp[0]);

  const std::array<double, 10>* swapped_values =
      reinterpret_cast<const std::array<double, 10>*>(&swapped_values_temp[0]);

  ConversionTest(TRITONSERVER_TYPE_UINT64, *original_values, *swapped_values);
}

TEST(EndianConversionTest, UINT8)
{
  const std::array<uint8_t, 10> original_values = {
      0xde, 0xad, 0xde, 0xad, 0xde, 0xad, 0xde, 0xad, 0xde, 0xad};


  ConversionTest(TRITONSERVER_TYPE_UINT8, original_values, original_values);
}


}}  // namespace triton::server

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
