// Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#undef __BYTE_ORDER__
#define __BYTE_ORDER__ __ORDER_BIG_ENDIAN__

#include <algorithm>

#include "common.h"

namespace triton { namespace server {

TEST(LittleEndianConversionTest, ConvertBYTES)
{

  const std::array<uint32_t, 10> swapped_values = {   0x04000000, 0xdeadbeef,
						      0x08000000, 0xdeadbeef, 0xdeadbeef,
						      0x0C000000, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef,
						      0x00000000 };

  const std::array<uint32_t, 10> original_values = {0x00000004, 0xdeadbeef,
						    0x00000008, 0xdeadbeef, 0xdeadbeef,
						    0x0000000C, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef,
						    0x00000000 };

  
  std::array<uint32_t, 10> values;

  std::copy(
      std::begin(original_values), std::end(original_values),
      std::begin(values));

  HostToLittleEndian(
      TRITONSERVER_TYPE_BYTES, reinterpret_cast<char*>(&values[0]),
      sizeof(values));

  for (int i = 0; i < values.size(); i++) {
    EXPECT_EQ(values[i], swapped_values[i]);
  }

  LittleEndianToHost(
      TRITONSERVER_TYPE_BYTES, reinterpret_cast<char*>(&values[0]),
      sizeof(values));


  for (int i = 0; i < values.size(); i++) {
    EXPECT_EQ(values[i], original_values[i]);
  }

  // partial support

  for (int num_partial_buffers = 1; num_partial_buffers <= sizeof(values); num_partial_buffers++) {

    size_t partial_buffer_size = sizeof(values) / num_partial_buffers;

    std::copy(
	      std::begin(swapped_values), std::end(swapped_values),
	      std::begin(values));

    std::vector<char*> partial_result;
    size_t offset = 0;
    char * partial_buffer = reinterpret_cast<char*>(&values[0]);
    size_t count = 0; 
    for (int partial_buffer_offset = 0; partial_buffer_offset < sizeof(values); partial_buffer_offset+=partial_buffer_size) {
      size_t current_buffer_size = std::min(partial_buffer_size, (sizeof(values) - partial_buffer_offset));
      LittleEndianToHost(TRITONSERVER_TYPE_BYTES,
			 partial_buffer,
			 current_buffer_size,
			 partial_result,
			 offset);
      partial_buffer += partial_buffer_size;
    }
    
    for (int i = 0; i < values.size(); i++) {
      EXPECT_EQ(values[i], original_values[i]);
    }
    
  }

 
}

TEST(LittleEndianConversionTest, ConvertUINT32)
{
  const std::array<uint32_t, 10> original_values = {
      0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef,
      0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef};

  const std::array<uint32_t, 10> swapped_values = {
      0xefbeadde, 0xefbeadde, 0xefbeadde, 0xefbeadde, 0xefbeadde,
      0xefbeadde, 0xefbeadde, 0xefbeadde, 0xefbeadde, 0xefbeadde};

  std::array<uint32_t, 10> values;

  std::copy(
      std::begin(original_values), std::end(original_values),
      std::begin(values));

  HostToLittleEndian(
      TRITONSERVER_TYPE_UINT32, reinterpret_cast<char*>(&values[0]),
      sizeof(values));

  for (int i = 0; i < values.size(); i++) {
    EXPECT_EQ(values[i], swapped_values[i]);
  }

  HostToLittleEndian(
      TRITONSERVER_TYPE_UINT32, reinterpret_cast<char*>(&values[0]),
      sizeof(values));


  for (int i = 0; i < values.size(); i++) {
    EXPECT_EQ(values[i], original_values[i]);
  }

  // partial support

  for (int num_partial_buffers = 1; num_partial_buffers <= sizeof(values); num_partial_buffers++) {

    size_t partial_buffer_size = sizeof(values) / num_partial_buffers;

    std::copy(
	      std::begin(original_values), std::end(original_values),
	      std::begin(values));

    std::vector<char*> partial_result;
    size_t offset = 0;
    char * partial_buffer = reinterpret_cast<char*>(&values[0]);
    for (int partial_buffer_offset = 0; partial_buffer_offset < sizeof(values); partial_buffer_offset+=partial_buffer_size) {
      size_t current_buffer_size = std::min(partial_buffer_size, (sizeof(values) - partial_buffer_offset));
      LittleEndianToHost(TRITONSERVER_TYPE_UINT32,
			 partial_buffer,
			 current_buffer_size,
			 partial_result,
			 offset);
      partial_buffer += partial_buffer_size;
    }
    
    for (int i = 0; i < values.size(); i++) {
      EXPECT_EQ(values[i], swapped_values[i]);
    }
    
  }

}

TEST(LittleEndianConversionTest, ConvertUINT16)
{
  uint16_t values[10] = {0xdead, 0xdead, 0xdead, 0xdead, 0xdead,
                         0xdead, 0xdead, 0xdead, 0xdead, 0xdead};

  HostToLittleEndian(
      TRITONSERVER_TYPE_UINT16, reinterpret_cast<char*>(&values[0]),
      sizeof(values));

  uint16_t swapped[10] = {0xadde, 0xadde, 0xadde, 0xadde, 0xadde,
                          0xadde, 0xadde, 0xadde, 0xadde, 0xadde};

  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(values[i], swapped[i]);
  }
}

}}  // namespace triton::server

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
