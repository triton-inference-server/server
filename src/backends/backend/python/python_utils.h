// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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
#pragma once

namespace nvidia { namespace inferenceserver { namespace backend {

static const std::unordered_map<std::string, TRITONSERVER_DataType>
    str2tritontype = {
        {"TYPE_BOOL", TRITONSERVER_DataType::TRITONSERVER_TYPE_BOOL},
        {"TYPE_BYTES", TRITONSERVER_DataType::TRITONSERVER_TYPE_BYTES},
        {"TYPE_FP16", TRITONSERVER_DataType::TRITONSERVER_TYPE_FP16},
        {"TYPE_FP32", TRITONSERVER_DataType::TRITONSERVER_TYPE_FP32},
        {"TYPE_FP64", TRITONSERVER_DataType::TRITONSERVER_TYPE_FP64},
        {"TYPE_INT8", TRITONSERVER_DataType::TRITONSERVER_TYPE_INT8},
        {"TYPE_INT16", TRITONSERVER_DataType::TRITONSERVER_TYPE_INT16},
        {"TYPE_INT32", TRITONSERVER_DataType::TRITONSERVER_TYPE_INT32},
        {"TYPE_INT64", TRITONSERVER_DataType::TRITONSERVER_TYPE_INT64},
        {"TYPE_INVALID", TRITONSERVER_DataType::TRITONSERVER_TYPE_INVALID},
        {"TYPE_UINT8", TRITONSERVER_DataType::TRITONSERVER_TYPE_UINT8},
        {"TYPE_UINT16", TRITONSERVER_DataType::TRITONSERVER_TYPE_UINT16},
        {"TYPE_UINT32", TRITONSERVER_DataType::TRITONSERVER_TYPE_UINT32},
        {"TYPE_UINT64", TRITONSERVER_DataType::TRITONSERVER_TYPE_UINT64}};

static const std::unordered_map<std::string, size_t> str2typesize = {
    {"TYPE_BOOL", sizeof(bool)}, /* Depends on implementation. Not specified
                                    by CPP standard. */
    {"TYPE_BYTES", 0x1},         {"TYPE_FP16", 0x2},   {"TYPE_FP32", 0x4},
    {"TYPE_FP64", 0x8},          {"TYPE_INT8", 0x1},   {"TYPE_INT16", 0x2},
    {"TYPE_INT32", 0x4},         {"TYPE_INT64", 0x8},  {"TYPE_INVALID", 0x0},
    {"TYPE_UINT8", 0x1},         {"TYPE_UINT16", 0x2}, {"TYPE_UINT32", 0x4},
    {"TYPE_UINT64", 0x8}};
const char* input_name;
}}}  // namespace nvidia::inferenceserver::backend
