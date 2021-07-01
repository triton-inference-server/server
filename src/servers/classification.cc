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

#include "src/servers/classification.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include "src/servers/common.h"

namespace nvidia { namespace inferenceserver {

namespace {

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 9)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

template <typename T>
TRITONSERVER_Error*
AddClassResults(
    TRITONSERVER_InferenceResponse* response, const uint32_t output_idx,
    const char* base, const size_t element_cnt, const uint32_t req_class_cnt,
    std::vector<std::string>* class_strs)
{
  const T* probs = reinterpret_cast<const T*>(base);

  std::vector<size_t> idx(element_cnt);
  iota(idx.begin(), idx.end(), 0);
  sort(idx.begin(), idx.end(), [&probs](size_t i1, size_t i2) {
    return probs[i1] > probs[i2];
  });

  const size_t class_cnt = std::min(element_cnt, (size_t)req_class_cnt);
  for (size_t k = 0; k < class_cnt; ++k) {
    class_strs->push_back(
        to_string_with_precision(probs[idx[k]]) + ":" + std::to_string(idx[k]));

    const char* label;
    RETURN_IF_ERR(TRITONSERVER_InferenceResponseOutputClassificationLabel(
        response, output_idx, idx[k], &label));
    if (label != nullptr) {
      class_strs->back() += ":";
      class_strs->back().append(label);
    }
  }

  return nullptr;  // success
}

}  // namespace


TRITONSERVER_Error*
TopkClassifications(
    TRITONSERVER_InferenceResponse* response, const uint32_t output_idx,
    const char* base, const size_t byte_size,
    const TRITONSERVER_DataType datatype, const uint32_t req_class_count,
    std::vector<std::string>* class_strs)
{
  const size_t element_cnt =
      byte_size / TRITONSERVER_DataTypeByteSize(datatype);

  switch (datatype) {
    case TRITONSERVER_TYPE_UINT8:
      return AddClassResults<uint8_t>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);
    case TRITONSERVER_TYPE_UINT16:
      return AddClassResults<uint16_t>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);
    case TRITONSERVER_TYPE_UINT32:
      return AddClassResults<uint32_t>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);
    case TRITONSERVER_TYPE_UINT64:
      return AddClassResults<uint64_t>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);

    case TRITONSERVER_TYPE_INT8:
      return AddClassResults<int8_t>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);
    case TRITONSERVER_TYPE_INT16:
      return AddClassResults<int16_t>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);
    case TRITONSERVER_TYPE_INT32:
      return AddClassResults<int32_t>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);
    case TRITONSERVER_TYPE_INT64:
      return AddClassResults<int64_t>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);

    case TRITONSERVER_TYPE_FP32:
      return AddClassResults<float>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);
    case TRITONSERVER_TYPE_FP64:
      return AddClassResults<double>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);

    default:
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              std::string("class result not available for output due to "
                          "unsupported type '") +
              std::string(TRITONSERVER_DataTypeString(datatype)) + "'")
              .c_str());
  }

  return nullptr;  // success
}

}}  // namespace nvidia::inferenceserver
