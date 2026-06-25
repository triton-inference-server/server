// Copyright 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Shared preprocessor macros for HTTP API handlers. Macros are not C++ symbols
// and are not "in" a namespace; this header is included from within
// triton::server for consistency with http_server.cc.
//
// Prerequisites at expansion sites: HttpCodeFromError, EVBufferAddErrorJson
// (from http_error_json.h),
// and (for RETURN_AND_RESPOND_IF_RESTRICTED) RespondIfRestricted must be
// visible — typically from the same translation unit's anonymous namespace and
// HTTPAPIServer member functions respectively.

#pragma once

#define RETURN_AND_RESPOND_IF_ERR(REQ, X)                \
  do {                                                   \
    TRITONSERVER_Error* err__ = (X);                     \
    if (err__ != nullptr) {                              \
      EVBufferAddErrorJson((REQ)->buffer_out, err__);    \
      evhtp_send_reply((REQ), HttpCodeFromError(err__)); \
      TRITONSERVER_ErrorDelete(err__);                   \
      return;                                            \
    }                                                    \
  } while (false)

#define RETURN_AND_RESPOND_WITH_ERR(REQ, CODE, MSG) \
  do {                                              \
    EVBufferAddErrorJson((REQ)->buffer_out, MSG);   \
    evhtp_send_reply((REQ), CODE);                  \
    return;                                         \
  } while (false)

#define RETURN_AND_RESPOND_IF_RESTRICTED(                               \
    REQ, RESTRICTED_CATEGORY, RESTRICTED_APIS)                          \
  do {                                                                  \
    auto const& is_restricted_api =                                     \
        RESTRICTED_APIS.IsRestricted(RESTRICTED_CATEGORY);              \
    auto const& restriction = RESTRICTED_APIS.Get(RESTRICTED_CATEGORY); \
    if (is_restricted_api && RespondIfRestricted(REQ, restriction)) {   \
      return;                                                           \
    }                                                                   \
  } while (false)
