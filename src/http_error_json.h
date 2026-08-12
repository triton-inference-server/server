// Copyright 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Lightweight {"error":"..."} serialization for HTTP error responses.
#pragma once

#include <event2/buffer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <cstring>

#include "triton/core/tritonserver.h"

namespace triton { namespace server {

inline void EVBufferAddErrorJson(evbuffer* buffer, const char* message) {
  if (message == nullptr) {
    message = "";
  }

  rapidjson::StringBuffer sb;
  sb.Reserve(static_cast<rapidjson::SizeType>(std::strlen(message) + 16));
  rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
  writer.StartObject();
  writer.Key("error");
  writer.String(message);
  writer.EndObject();

  evbuffer_add(buffer, sb.GetString(), sb.GetSize());
}

inline void EVBufferAddErrorJson(evbuffer* buffer, TRITONSERVER_Error* err) {
  EVBufferAddErrorJson(buffer, TRITONSERVER_ErrorMessage(err));
}

}}  // namespace triton::server
