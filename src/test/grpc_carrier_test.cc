// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// from gtest. Okay as FAIL() is not used in infer_handler.
#ifdef FAIL
#undef FAIL
#endif

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

#include "grpc/infer_handler.h"

namespace triton { namespace server { namespace grpc { namespace {

TEST(GrpcServerCarrierTest, PreservesMetadataValueLength)
{
  constexpr std::string_view traceparent =
      "00-0af7651916cd43dd8448eb211c12666c-b7ad6b7169242424-01";
  static_assert(traceparent.size() == 55);

  std::array<char, 57> storage{};
  std::copy(traceparent.begin(), traceparent.end(), storage.begin());
  storage[traceparent.size()] = 'X';

  const ::grpc::string_ref metadata_value(storage.data(), traceparent.size());
  const auto view = GrpcMetadataValueView(metadata_value);

  EXPECT_EQ(view.size(), traceparent.size());
  EXPECT_EQ(
      std::string(view.data(), view.size()),
      std::string(traceparent.data(), traceparent.size()));
  EXPECT_EQ(storage[traceparent.size()], 'X');
}

}}}}  // namespace triton::server::grpc
