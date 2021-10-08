// Copyright 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifdef _WIN32
#define NOMINMAX
#endif

#include "src/servers/http_server.h"

#include <event2/buffer.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>
#include <re2/re2.h>
#include <algorithm>
#include <list>
#include <thread>
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config.h"
#include "src/servers/classification.h"
#include "src/servers/data_compressor.h"

#define TRITONJSON_STATUSTYPE TRITONSERVER_Error*
#define TRITONJSON_STATUSRETURN(M) \
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (M).c_str())
#define TRITONJSON_STATUSSUCCESS nullptr
#include "triton/common/triton_json.h"

#ifdef TRITON_ENABLE_GPU
extern "C" {
#include <b64/cdecode.h>
}
#endif  // TRITON_ENABLE_GPU

#ifdef TRITON_ENABLE_TRACING
#include "src/servers/tracer.h"
#endif  // TRITON_ENABLE_TRACING

namespace nvidia { namespace inferenceserver {

TRITONSERVER_Error*
HTTPServer::Start()
{
  if (!worker_.joinable()) {
    evbase_ = event_base_new();
    htp_ = evhtp_new(evbase_, NULL);
    evhtp_enable_flag(htp_, EVHTP_FLAG_ENABLE_NODELAY);
    evhtp_set_gencb(htp_, HTTPServer::Dispatch, this);
    evhtp_use_threads_wexit(htp_, NULL, NULL, thread_cnt_, NULL);
    evhtp_bind_socket(htp_, "0.0.0.0", port_, 1024);

    // Set listening event for breaking event loop
    evutil_socketpair(AF_UNIX, SOCK_STREAM, 0, fds_);
    break_ev_ = event_new(evbase_, fds_[0], EV_READ, StopCallback, evbase_);
    event_add(break_ev_, NULL);
    worker_ = std::thread(event_base_loop, evbase_, 0);

    return nullptr;
  }

  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_ALREADY_EXISTS, "HTTP server is already running.");
}

TRITONSERVER_Error*
HTTPServer::Stop()
{
  if (worker_.joinable()) {
    // Notify event loop to break via fd write
    send(fds_[1], (const char*)&evbase_, sizeof(event_base*), 0);
    worker_.join();
    event_free(break_ev_);
    evutil_closesocket(fds_[0]);
    evutil_closesocket(fds_[1]);
    evhtp_unbind_socket(htp_);
    evhtp_free(htp_);
    event_base_free(evbase_);
    return nullptr;
  }

  return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNAVAILABLE, "HTTP server is not running.");
}

void
HTTPServer::StopCallback(evutil_socket_t sock, short events, void* arg)
{
  struct event_base* base = (struct event_base*)arg;
  event_base_loopbreak(base);
}

void
HTTPServer::Dispatch(evhtp_request_t* req, void* arg)
{
  (static_cast<HTTPServer*>(arg))->Handle(req);
}

#ifdef TRITON_ENABLE_METRICS

void
HTTPMetricsServer::Handle(evhtp_request_t* req)
{
  LOG_VERBOSE(1) << "HTTP request: " << req->method << " "
                 << req->uri->path->full;

  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  evhtp_res res = EVHTP_RES_BADREQ;

  // Call to metric endpoint should not have any trailing string
  if (RE2::FullMatch(std::string(req->uri->path->full), api_regex_)) {
    TRITONSERVER_Metrics* metrics = nullptr;
    TRITONSERVER_Error* err =
        TRITONSERVER_ServerMetrics(server_.get(), &metrics);
    if (err == nullptr) {
      const char* base;
      size_t byte_size;
      err = TRITONSERVER_MetricsFormatted(
          metrics, TRITONSERVER_METRIC_PROMETHEUS, &base, &byte_size);
      if (err == nullptr) {
        res = EVHTP_RES_OK;
        evbuffer_add(req->buffer_out, base, byte_size);
      }
    }

    TRITONSERVER_MetricsDelete(metrics);
    TRITONSERVER_ErrorDelete(err);
  }

  evhtp_send_reply(req, res);
}

TRITONSERVER_Error*
HTTPMetricsServer::Create(
    const std::shared_ptr<TRITONSERVER_Server>& server, const int32_t port,
    const int thread_cnt, std::unique_ptr<HTTPServer>* metrics_server)
{
  metrics_server->reset(new HTTPMetricsServer(server, port, thread_cnt));

  const std::string addr = "0.0.0.0:" + std::to_string(port);
  LOG_INFO << "Started Metrics Service at " << addr;

  return nullptr;
}

#endif  // TRITON_ENABLE_METRICS


namespace {

// Allocate an evbuffer of size 'byte_size'. Return the 'evb' and
// the 'base' address of the buffer contents.
TRITONSERVER_Error*
AllocEVBuffer(const size_t byte_size, evbuffer** evb, void** base)
{
  evbuffer* evhttp_buffer = evbuffer_new();
  if (evhttp_buffer == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "failed to create evbuffer for output tensor");
  }

  // Reserve requested space in evbuffer...
  struct evbuffer_iovec output_iovec;
  if (evbuffer_reserve_space(evhttp_buffer, byte_size, &output_iovec, 1) != 1) {
    evbuffer_free(evhttp_buffer);
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string(
            "failed to reserve " + std::to_string(byte_size) +
            " bytes in output tensor buffer")
            .c_str());
  }

  if (output_iovec.iov_len < byte_size) {
    evbuffer_free(evhttp_buffer);
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string(
            "reserved " + std::to_string(output_iovec.iov_len) +
            " bytes in output tensor buffer, need " + std::to_string(byte_size))
            .c_str());
  }

  output_iovec.iov_len = byte_size;
  *base = output_iovec.iov_base;

  // Immediately commit the buffer space. We are relying on evbuffer
  // not to relocate this space. Because we request a contiguous
  // chunk every time (above by allowing only a single entry in
  // output_iovec), this seems to be a valid assumption.
  if (evbuffer_commit_space(evhttp_buffer, &output_iovec, 1) != 0) {
    *base = nullptr;
    evbuffer_free(evhttp_buffer);
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "failed to commit output tensors to output buffer");
  }

  *evb = evhttp_buffer;

  return nullptr;  // success
}

TRITONSERVER_Error*
JsonBytesArrayByteSize(
    triton::common::TritonJson::Value& tensor_data, size_t* byte_size)
{
  *byte_size = 0;

  for (size_t i = 0; i < tensor_data.ArraySize(); i++) {
    triton::common::TritonJson::Value el;
    RETURN_IF_ERR(tensor_data.At(i, &el));

    // Recurse if not last dimension...
    TRITONSERVER_Error* assert_err =
        el.AssertType(triton::common::TritonJson::ValueType::ARRAY);
    if (assert_err == nullptr) {
      RETURN_IF_ERR(JsonBytesArrayByteSize(el, byte_size));
    } else {
      // Serialized data size is the length of the string itself plus
      // 4 bytes to record the string length.
      const char* str;
      size_t len = 0;
      RETURN_MSG_IF_ERR(
          el.AsString(&str, &len), "Unable to parse JSON bytes array");
      *byte_size += len + sizeof(uint32_t);
    }
    TRITONSERVER_ErrorDelete(assert_err);
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ReadDataFromJsonHelper(
    char* base, const TRITONSERVER_DataType dtype,
    triton::common::TritonJson::Value& tensor_data, int* counter,
    int64_t expected_cnt)
{
  // FIXME should invert loop and switch so don't have to do a switch
  // each iteration.
  for (size_t i = 0; i < tensor_data.ArraySize(); i++) {
    triton::common::TritonJson::Value el;
    RETURN_IF_ERR(tensor_data.At(i, &el));

    // Recurse if not last dimension...
    TRITONSERVER_Error* assert_err =
        el.AssertType(triton::common::TritonJson::ValueType::ARRAY);
    if (assert_err == nullptr) {
      RETURN_IF_ERR(
          ReadDataFromJsonHelper(base, dtype, el, counter, expected_cnt));
    } else {
      // Check if writing to 'serialized' is overrunning the expected byte_size
      if (*counter >= expected_cnt) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            "Shape does not match true shape of 'data' field");
      }
      switch (dtype) {
        case TRITONSERVER_TYPE_BOOL: {
          bool b = false;
          RETURN_IF_ERR(el.AsBool(&b));
          uint8_t* data_vec = reinterpret_cast<uint8_t*>(base);
          // FIXME for unsigned should bounds check and raise error
          // since otherwise the actually used value will be
          // unexpected.
          data_vec[*counter] = (uint8_t)(b ? 1 : 0);
          *counter += 1;
          break;
        }
        case TRITONSERVER_TYPE_UINT8: {
          uint64_t ui = 0;
          RETURN_IF_ERR(el.AsUInt(&ui));
          uint8_t* data_vec = reinterpret_cast<uint8_t*>(base);
          data_vec[*counter] = (uint8_t)ui;
          *counter += 1;
          break;
        }
        case TRITONSERVER_TYPE_UINT16: {
          uint64_t ui = 0;
          RETURN_IF_ERR(el.AsUInt(&ui));
          uint16_t* data_vec = reinterpret_cast<uint16_t*>(base);
          data_vec[*counter] = (uint16_t)ui;
          *counter += 1;
          break;
        }
        case TRITONSERVER_TYPE_UINT32: {
          uint64_t ui = 0;
          RETURN_IF_ERR(el.AsUInt(&ui));
          uint32_t* data_vec = reinterpret_cast<uint32_t*>(base);
          data_vec[*counter] = (uint32_t)ui;
          *counter += 1;
          break;
        }
        case TRITONSERVER_TYPE_UINT64: {
          uint64_t ui = 0;
          RETURN_IF_ERR(el.AsUInt(&ui));
          uint64_t* data_vec = reinterpret_cast<uint64_t*>(base);
          data_vec[*counter] = ui;
          *counter += 1;
          break;
        }
        case TRITONSERVER_TYPE_INT8: {
          // FIXME for signed type just assigning to smaller type is
          // "implementation defined" and so really need to bounds
          // check.
          int64_t si = 0;
          RETURN_IF_ERR(el.AsInt(&si));
          int8_t* data_vec = reinterpret_cast<int8_t*>(base);
          data_vec[*counter] = (int8_t)si;
          *counter += 1;
          break;
        }
        case TRITONSERVER_TYPE_INT16: {
          int64_t si = 0;
          RETURN_IF_ERR(el.AsInt(&si));
          int16_t* data_vec = reinterpret_cast<int16_t*>(base);
          data_vec[*counter] = (int16_t)si;
          *counter += 1;
          break;
        }
        case TRITONSERVER_TYPE_INT32: {
          int64_t si = 0;
          RETURN_IF_ERR(el.AsInt(&si));
          int32_t* data_vec = reinterpret_cast<int32_t*>(base);
          data_vec[*counter] = (int32_t)si;
          *counter += 1;
          break;
        }
        case TRITONSERVER_TYPE_INT64: {
          int64_t si = 0;
          RETURN_IF_ERR(el.AsInt(&si));
          int64_t* data_vec = reinterpret_cast<int64_t*>(base);
          data_vec[*counter] = si;
          *counter += 1;
          break;
        }
        case TRITONSERVER_TYPE_FP32: {
          double fp64 = 0;
          RETURN_IF_ERR(el.AsDouble(&fp64));
          float* data_vec = reinterpret_cast<float*>(base);
          data_vec[*counter] = fp64;
          *counter += 1;
          break;
        }
        case TRITONSERVER_TYPE_FP64: {
          double fp64 = 0;
          RETURN_IF_ERR(el.AsDouble(&fp64));
          double* data_vec = reinterpret_cast<double*>(base);
          data_vec[*counter] = fp64;
          *counter += 1;
          break;
        }
        case TRITONSERVER_TYPE_BYTES: {
          const char* cstr;
          size_t len = 0;
          RETURN_IF_ERR(el.AsString(&cstr, &len));
          if (static_cast<int64_t>(*counter + len + sizeof(uint32_t)) >
              expected_cnt) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                "Shape does not match true shape of 'data' field");
          }
          memcpy(
              base + *counter, reinterpret_cast<char*>(&len), sizeof(uint32_t));
          std::copy(cstr, cstr + len, base + *counter + sizeof(uint32_t));
          *counter += len + sizeof(uint32_t);
          break;
        }
        default:
          break;
      }
    }
    TRITONSERVER_ErrorDelete(assert_err);
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ReadDataFromJson(
    const char* tensor_name, triton::common::TritonJson::Value& tensor_data,
    char* base, const TRITONSERVER_DataType dtype, int64_t expected_cnt)
{
  int counter = 0;
  switch (dtype) {
    // FP16 not supported via JSON
    case TRITONSERVER_TYPE_FP16:
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "receiving FP16 data via JSON is not supported. Please use the "
              "binary data format for input " +
              std::string(tensor_name))
              .c_str());

    case TRITONSERVER_TYPE_INVALID:
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string("invalid datatype for input " + std::string(tensor_name))
              .c_str());

    default:
      RETURN_MSG_IF_ERR(
          ReadDataFromJsonHelper(
              base, dtype, tensor_data, &counter, expected_cnt),
          "Unable to parse 'data'");
      break;
  }

  // Check if 'ReadDataFromJsonHelper' reads less than the expected byte size
  if (counter != expected_cnt) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "Unable to parse 'data': Shape does not match true shape of 'data' "
        "field");
  }

  return nullptr;
}

TRITONSERVER_Error*
WriteDataToJsonCheck(
    const std::string& output_name, const size_t byte_size,
    const size_t expected_size)
{
  if (byte_size != expected_size) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string(
            "output tensor shape does not match size of output for '" +
            output_name + "'")
            .c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
WriteDataToJson(
    triton::common::TritonJson::Value* data_json,
    const std::string& output_name, const TRITONSERVER_DataType datatype,
    const void* base, const size_t byte_size, const size_t element_count)
{
  switch (datatype) {
    case TRITONSERVER_TYPE_BOOL: {
      const uint8_t* bool_base = reinterpret_cast<const uint8_t*>(base);
      if (byte_size != (element_count * sizeof(uint8_t))) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "output tensor shape does not match size of output for '" +
                output_name + "'")
                .c_str());
      }
      for (size_t e = 0; e < element_count; ++e) {
        data_json->AppendBool((bool_base[e] == 0) ? false : true);
      }
      break;
    }

    case TRITONSERVER_TYPE_UINT8: {
      RETURN_IF_ERR(WriteDataToJsonCheck(
          output_name, byte_size, sizeof(uint8_t) * element_count));
      const uint8_t* cbase = reinterpret_cast<const uint8_t*>(base);
      for (size_t e = 0; e < element_count; ++e) {
        data_json->AppendUInt(cbase[e]);
      }
      break;
    }

    case TRITONSERVER_TYPE_UINT16: {
      RETURN_IF_ERR(WriteDataToJsonCheck(
          output_name, byte_size, sizeof(uint16_t) * element_count));
      const uint16_t* cbase = reinterpret_cast<const uint16_t*>(base);
      for (size_t e = 0; e < element_count; ++e) {
        data_json->AppendUInt(cbase[e]);
      }
      break;
    }

    case TRITONSERVER_TYPE_UINT32: {
      RETURN_IF_ERR(WriteDataToJsonCheck(
          output_name, byte_size, sizeof(uint32_t) * element_count));
      const uint32_t* cbase = reinterpret_cast<const uint32_t*>(base);
      for (size_t e = 0; e < element_count; ++e) {
        data_json->AppendUInt(cbase[e]);
      }
      break;
    }

    case TRITONSERVER_TYPE_UINT64: {
      RETURN_IF_ERR(WriteDataToJsonCheck(
          output_name, byte_size, sizeof(uint64_t) * element_count));
      const uint64_t* cbase = reinterpret_cast<const uint64_t*>(base);
      for (size_t e = 0; e < element_count; ++e) {
        data_json->AppendUInt(cbase[e]);
      }
      break;
    }

    case TRITONSERVER_TYPE_INT8: {
      RETURN_IF_ERR(WriteDataToJsonCheck(
          output_name, byte_size, sizeof(int8_t) * element_count));
      const int8_t* cbase = reinterpret_cast<const int8_t*>(base);
      for (size_t e = 0; e < element_count; ++e) {
        data_json->AppendInt(cbase[e]);
      }
      break;
    }

    case TRITONSERVER_TYPE_INT16: {
      RETURN_IF_ERR(WriteDataToJsonCheck(
          output_name, byte_size, sizeof(int16_t) * element_count));
      const int16_t* cbase = reinterpret_cast<const int16_t*>(base);
      for (size_t e = 0; e < element_count; ++e) {
        data_json->AppendInt(cbase[e]);
      }
      break;
    }

    case TRITONSERVER_TYPE_INT32: {
      RETURN_IF_ERR(WriteDataToJsonCheck(
          output_name, byte_size, sizeof(int32_t) * element_count));
      const int32_t* cbase = reinterpret_cast<const int32_t*>(base);
      for (size_t e = 0; e < element_count; ++e) {
        data_json->AppendInt(cbase[e]);
      }
      break;
    }

    case TRITONSERVER_TYPE_INT64: {
      RETURN_IF_ERR(WriteDataToJsonCheck(
          output_name, byte_size, sizeof(int64_t) * element_count));
      const int64_t* cbase = reinterpret_cast<const int64_t*>(base);
      for (size_t e = 0; e < element_count; ++e) {
        data_json->AppendInt(cbase[e]);
      }
      break;
    }

    // FP16 not supported via JSON
    case TRITONSERVER_TYPE_FP16:
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "sending FP16 data via JSON is not supported. Please use the "
          "binary data format for output");

    case TRITONSERVER_TYPE_FP32: {
      RETURN_IF_ERR(WriteDataToJsonCheck(
          output_name, byte_size, sizeof(float) * element_count));
      const float* cbase = reinterpret_cast<const float*>(base);
      for (size_t e = 0; e < element_count; ++e) {
        data_json->AppendDouble(cbase[e]);
      }
      break;
    }

    case TRITONSERVER_TYPE_FP64: {
      RETURN_IF_ERR(WriteDataToJsonCheck(
          output_name, byte_size, sizeof(double) * element_count));
      const double* cbase = reinterpret_cast<const double*>(base);
      for (size_t e = 0; e < element_count; ++e) {
        data_json->AppendDouble(cbase[e]);
      }
      break;
    }

    case TRITONSERVER_TYPE_BYTES: {
      const char* cbase = reinterpret_cast<const char*>(base);
      size_t offset = 0;
      for (size_t e = 0; e < element_count; ++e) {
        if ((offset + sizeof(uint32_t)) > byte_size) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "output tensor shape does not match size of output for '" +
                  output_name + "'")
                  .c_str());
        }

        const size_t len = *(reinterpret_cast<const uint32_t*>(cbase + offset));
        offset += sizeof(uint32_t);

        if ((offset + len) > byte_size) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "output tensor shape does not match size of output for '" +
                  output_name + "'")
                  .c_str());
        }

        // Can use stringref because 'base' buffer is not deleted
        // until response is deleted and that happens after this json
        // is serialized.
        data_json->AppendStringRef(cbase + offset, len);
        offset += len;
      }
      break;
    }

    case TRITONSERVER_TYPE_INVALID:
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "Invalid data type for output tensor");
  }

  return nullptr;  // success
}

void
EVBufferAddErrorJson(evbuffer* buffer, TRITONSERVER_Error* err)
{
  const char* message = TRITONSERVER_ErrorMessage(err);

  triton::common::TritonJson::Value response(
      triton::common::TritonJson::ValueType::OBJECT);
  response.AddStringRef("error", message, strlen(message));

  triton::common::TritonJson::WriteBuffer buffer_json;
  response.Write(&buffer_json);

  evbuffer_add(buffer, buffer_json.Base(), buffer_json.Size());
}

TRITONSERVER_Error*
CheckBinaryInputData(
    triton::common::TritonJson::Value& request_input, bool* is_binary,
    size_t* byte_size)
{
  *is_binary = false;

  triton::common::TritonJson::Value params_json;
  if (request_input.Find("parameters", &params_json)) {
    triton::common::TritonJson::Value binary_data_size_json;
    if (params_json.Find("binary_data_size", &binary_data_size_json)) {
      RETURN_MSG_IF_ERR(
          binary_data_size_json.AsUInt(byte_size),
          "Unable to parse 'binary_data_size'");
      *is_binary = true;
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
CheckBinaryOutputData(
    triton::common::TritonJson::Value& request_output, bool* is_binary)
{
  *is_binary = false;

  triton::common::TritonJson::Value params_json;
  if (request_output.Find("parameters", &params_json)) {
    triton::common::TritonJson::Value binary_data_json;
    if (params_json.Find("binary_data", &binary_data_json)) {
      RETURN_MSG_IF_ERR(
          binary_data_json.AsBool(is_binary), "Unable to parse 'binary_data'");
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
CheckSharedMemoryData(
    triton::common::TritonJson::Value& request_input, bool* use_shm,
    const char** shm_region, uint64_t* offset, uint64_t* byte_size)
{
  *use_shm = false;
  *offset = 0;
  *byte_size = 0;

  triton::common::TritonJson::Value params_json;
  if (request_input.Find("parameters", &params_json)) {
    {
      triton::common::TritonJson::Value region_json;
      if (params_json.Find("shared_memory_region", &region_json)) {
        *use_shm = true;
        size_t len;
        RETURN_MSG_IF_ERR(
            region_json.AsString(shm_region, &len),
            "Unable to parse 'shared_memory_region'");
      }
    }

    {
      triton::common::TritonJson::Value offset_json;
      if (params_json.Find("shared_memory_offset", &offset_json)) {
        RETURN_MSG_IF_ERR(
            offset_json.AsUInt(offset),
            "Unable to parse 'shared_memory_offset'");
      }
    }

    {
      triton::common::TritonJson::Value size_json;
      if (params_json.Find("shared_memory_byte_size", &size_json)) {
        RETURN_MSG_IF_ERR(
            size_json.AsUInt(byte_size),
            "Unable to parse 'shared_memory_byte_size'");
      }
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
CheckClassificationOutput(
    triton::common::TritonJson::Value& request_output, uint64_t* num_classes)
{
  *num_classes = 0;

  triton::common::TritonJson::Value params_json;
  if (request_output.Find("parameters", &params_json)) {
    triton::common::TritonJson::Value cls_json;
    if (params_json.Find("classification", &cls_json)) {
      RETURN_MSG_IF_ERR(
          cls_json.AsUInt(num_classes), "Unable to set 'classification'");
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ValidateInputContentType(triton::common::TritonJson::Value& io)
{
  bool has_data = false;
  bool has_binary = false;
  bool has_shared_memory = false;

  has_data = io.Find("data");

  triton::common::TritonJson::Value params_json;
  if (io.Find("parameters", &params_json)) {
    has_binary = params_json.Find("binary_data_size");
    has_shared_memory = params_json.Find("shared_memory_region");
  }

  int set_count = has_data + has_binary + has_shared_memory;
  if (set_count != 1) {
    std::string err_str =
        "Input must set only one of the following fields: 'data', "
        "'binary_data_size' in 'parameters', 'shared_memory_region' in "
        "'parameters'. But";
    if (set_count == 0) {
      err_str += " no field is set";
    } else {
      err_str += " set";
      if (has_data) {
        err_str += " 'data'";
      }
      if (has_binary) {
        err_str += " 'binary_data_size'";
      }
      if (has_shared_memory) {
        err_str += " 'shared_memory_region'";
      }
    }
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG, err_str.c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ValidateOutputParameter(triton::common::TritonJson::Value& io)
{
  triton::common::TritonJson::Value params_json;
  if (io.Find("parameters", &params_json)) {
    const bool has_shared_memory = params_json.Find("shared_memory_region");
    if (has_shared_memory) {
      // Currently shared memory can't set with classification because
      // cls results are not stored in shared memory, internally it is computed
      // based on results in shared memory.
      if (params_json.Find("classification")) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "Output can't set both 'shared_memory_region' and "
            "'classification'");
      }

      triton::common::TritonJson::Value binary_data_json;
      if (params_json.Find("binary_data", &binary_data_json)) {
        bool is_binary;
        RETURN_MSG_IF_ERR(
            binary_data_json.AsBool(&is_binary), "Unable to set 'binary_data'");
        if (is_binary) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              "Output can't set both 'shared_memory_region' and 'binary_data'");
        }
      }
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
EVBufferToJson(
    triton::common::TritonJson::Value* document, evbuffer_iovec* v, int* v_idx,
    const size_t length, int n)
{
  size_t offset = 0, remaining_length = length;
  char* json_base;
  std::vector<char> json_buffer;

  // No need to memcpy when number of iovecs is 1
  if ((n > 0) && (v[0].iov_len >= remaining_length)) {
    json_base = static_cast<char*>(v[0].iov_base);
    if (v[0].iov_len > remaining_length) {
      v[0].iov_base = static_cast<void*>(json_base + remaining_length);
      v[0].iov_len -= remaining_length;
      remaining_length = 0;
    } else if (v[0].iov_len == remaining_length) {
      remaining_length = 0;
      *v_idx += 1;
    }
  } else {
    json_buffer.resize(length);
    json_base = json_buffer.data();
    while ((remaining_length > 0) && (*v_idx < n)) {
      char* base = static_cast<char*>(v[*v_idx].iov_base);
      size_t base_size;
      if (v[*v_idx].iov_len > remaining_length) {
        base_size = remaining_length;
        v[*v_idx].iov_base = static_cast<void*>(base + remaining_length);
        v[*v_idx].iov_len -= remaining_length;
        remaining_length = 0;
      } else {
        base_size = v[*v_idx].iov_len;
        remaining_length -= v[*v_idx].iov_len;
        *v_idx += 1;
      }

      memcpy(json_base + offset, base, base_size);
      offset += base_size;
    }
  }

  if (remaining_length != 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected size for request JSON, expecting " +
            std::to_string(remaining_length) + " more bytes")
            .c_str());
  }

  RETURN_IF_ERR(document->Parse(json_base, length));

  return nullptr;  // success
}

std::string
CompressionTypeUsed(const std::string accept_encoding)
{
  std::vector<std::string> encodings;
  size_t offset = 0;
  size_t delimeter_pos = accept_encoding.find(',');
  while (delimeter_pos != std::string::npos) {
    encodings.emplace_back(
        accept_encoding.substr(offset, delimeter_pos - offset));
    offset = delimeter_pos + 1;
    delimeter_pos = accept_encoding.find(',', offset);
  }
  std::string res = "identity";
  double weight = 0;
  encodings.emplace_back(accept_encoding.substr(offset));
  for (const auto& encoding : encodings) {
    auto start_pos = encoding.find_first_not_of(' ');
    auto weight_pos = encoding.find(";q=");
    // Skip if the encoding is malformed
    if ((start_pos == std::string::npos) ||
        ((weight_pos != std::string::npos) && (start_pos >= weight_pos))) {
      continue;
    }
    const std::string type =
        (weight_pos == std::string::npos)
            ? encoding.substr(start_pos)
            : encoding.substr(start_pos, weight_pos - start_pos);
    double type_weight = 1;
    if (weight_pos != std::string::npos) {
      try {
        type_weight = std::stod(encoding.substr(weight_pos + 3));
      }
      catch (const std::invalid_argument& ia) {
        continue;
      }
    }
    if (((type == "identity") || (type == "deflate") || (type == "gzip")) &&
        (type_weight > weight)) {
      res = type;
      weight = type_weight;
    }
  }
  return res;
}

}  // namespace

HTTPAPIServer::HTTPAPIServer(
    const std::shared_ptr<TRITONSERVER_Server>& server,
    nvidia::inferenceserver::TraceManager* trace_manager,
    const std::shared_ptr<SharedMemoryManager>& shm_manager, const int32_t port,
    const int thread_cnt)
    : HTTPServer(port, thread_cnt), server_(server),
      trace_manager_(trace_manager), shm_manager_(shm_manager),
      allocator_(nullptr), server_regex_(R"(/v2(?:/health/(live|ready))?)"),
      model_regex_(
          R"(/v2/models/([^/]+)(?:/versions/([0-9]+))?(?:/(infer|ready|config|stats))?)"),
      modelcontrol_regex_(
          R"(/v2/repository(?:/([^/]+))?/(index|models/([^/]+)/(load|unload)))"),
      systemsharedmemory_regex_(
          R"(/v2/systemsharedmemory(?:/region/([^/]+))?/(status|register|unregister))"),
      cudasharedmemory_regex_(
          R"(/v2/cudasharedmemory(?:/region/([^/]+))?/(status|register|unregister))")
{
  // FIXME, don't cache server metadata. The http endpoint should
  // not be deciding that server metadata will not change during
  // execution.
  TRITONSERVER_Message* message = nullptr;
  server_metadata_err_ = TRITONSERVER_ServerMetadata(server_.get(), &message);
  if (server_metadata_err_ == nullptr) {
    const char* buffer;
    size_t byte_size;
    server_metadata_err_ =
        TRITONSERVER_MessageSerializeToJson(message, &buffer, &byte_size);
    server_metadata_ = std::string(buffer, byte_size);
  }

  if (message != nullptr) {
    TRITONSERVER_MessageDelete(message);
  }

  FAIL_IF_ERR(
      TRITONSERVER_ResponseAllocatorNew(
          &allocator_, InferResponseAlloc, InferResponseFree,
          nullptr /* start_fn */),
      "creating response allocator");
}

HTTPAPIServer::~HTTPAPIServer()
{
  if (server_metadata_err_ != nullptr) {
    TRITONSERVER_ErrorDelete(server_metadata_err_);
  }
  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_ResponseAllocatorDelete(allocator_),
      "deleting response allocator");
}

TRITONSERVER_Error*
HTTPAPIServer::InferResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  AllocPayload* payload = reinterpret_cast<AllocPayload*>(userp);
  std::unordered_map<std::string, AllocPayload::OutputInfo*>& output_map =
      payload->output_map_;
  const AllocPayload::OutputInfo::Kind default_output_kind =
      payload->default_output_kind_;

  *buffer = nullptr;
  *buffer_userp = nullptr;
  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;

  AllocPayload::OutputInfo* info = nullptr;

  // If we don't find an output then it means that the output wasn't
  // explicitly specified in the request. In that case we create an
  // OutputInfo for it that uses default setting of JSON.
  auto pr = output_map.find(tensor_name);
  if (pr == output_map.end()) {
    info = new AllocPayload::OutputInfo(default_output_kind, 0);
  } else {
    // Take ownership of the OutputInfo object.
    info = pr->second;
    output_map.erase(pr);
  }

  // If the output is in shared memory...
  if (info->kind_ == AllocPayload::OutputInfo::SHM) {
    // ...then make sure shared memory size is at least as big as
    // the size of the output.
    if (byte_size > info->byte_size_) {
      delete info;
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          std::string(
              "shared memory size specified with the request for output '" +
              std::string(tensor_name) + "' (" +
              std::to_string(info->byte_size_) + " bytes) should be at least " +
              std::to_string(byte_size) + " bytes to hold the results")
              .c_str());
    }

    *buffer = const_cast<void*>(info->base_);
    *actual_memory_type = info->memory_type_;
    *actual_memory_type_id = info->device_id_;

    // Don't need info for shared-memory output...
    delete info;

    LOG_VERBOSE(1) << "HTTP: using shared-memory for '" << tensor_name
                   << "', size: " << byte_size << ", addr: " << *buffer;
    return nullptr;  // Success
  }

  // Don't need to do anything if no memory was requested.
  if (byte_size > 0) {
    // Can't allocate for any memory type other than CPU. If asked to
    // allocate on GPU memory then force allocation on CPU instead.
    if (*actual_memory_type != TRITONSERVER_MEMORY_CPU) {
      LOG_VERBOSE(1) << "HTTP: unable to provide '" << tensor_name << "' in "
                     << TRITONSERVER_MemoryTypeString(*actual_memory_type)
                     << ", will use "
                     << TRITONSERVER_MemoryTypeString(TRITONSERVER_MEMORY_CPU);
      *actual_memory_type = TRITONSERVER_MEMORY_CPU;
      *actual_memory_type_id = 0;
    }

    evbuffer* evhttp_buffer;
    TRITONSERVER_Error* err = AllocEVBuffer(byte_size, &evhttp_buffer, buffer);
    if (err != nullptr) {
      delete info;
      return err;
    }

    // Associate info with the evbuffer with this allocation.
    // Ownership passes to 'buffer_userp' which has the same lifetime
    // as the buffer itself.
    info->evbuffer_ = evhttp_buffer;

    LOG_VERBOSE(1) << "HTTP using buffer for: '" << tensor_name
                   << "', size: " << byte_size << ", addr: " << *buffer;
  }

  *buffer_userp = reinterpret_cast<void*>(info);

  return nullptr;  // Success
}

TRITONSERVER_Error*
HTTPAPIServer::InferResponseFree(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  LOG_VERBOSE(1) << "HTTP release: "
                 << "size " << byte_size << ", addr " << buffer;

  // 'buffer' is backed by shared memory or evbuffer so we don't
  // delete directly.
  auto info = reinterpret_cast<AllocPayload::OutputInfo*>(buffer_userp);
  delete info;

  return nullptr;  // Success
}

void
HTTPAPIServer::HandleServerHealth(evhtp_request_t* req, const std::string& kind)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  TRITONSERVER_Error* err = nullptr;
  bool ready = false;

  if (kind == "live") {
    err = TRITONSERVER_ServerIsLive(server_.get(), &ready);
  } else {
    err = TRITONSERVER_ServerIsReady(server_.get(), &ready);
  }

  evhtp_send_reply(
      req, (ready && (err == nullptr)) ? EVHTP_RES_OK : EVHTP_RES_BADREQ);

  TRITONSERVER_ErrorDelete(err);
}

void
HTTPAPIServer::HandleRepositoryIndex(
    evhtp_request_t* req, const std::string& repository_name)
{
  if (req->method != htp_method_POST) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  TRITONSERVER_Error* err = nullptr;

  struct evbuffer_iovec* v = nullptr;
  int v_idx = 0;
  int n = evbuffer_peek(req->buffer_in, -1, NULL, NULL, 0);
  if (n > 0) {
    v = static_cast<struct evbuffer_iovec*>(
        alloca(sizeof(struct evbuffer_iovec) * n));
    if (evbuffer_peek(req->buffer_in, -1, NULL, v, n) != n) {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "unexpected error getting registry index request body");
    }
  }

  bool ready = false;

  if (err == nullptr) {
    // If no request json then just use all default values.
    size_t buffer_len = evbuffer_get_length(req->buffer_in);
    if (buffer_len > 0) {
      triton::common::TritonJson::Value index_request;
      err = EVBufferToJson(&index_request, v, &v_idx, buffer_len, n);
      if (err == nullptr) {
        triton::common::TritonJson::Value ready_json;
        if (index_request.Find("ready", &ready_json)) {
          err = ready_json.AsBool(&ready);
        }
      }
    }
  }

  evhtp_headers_add_header(
      req->headers_out,
      evhtp_header_new(kContentTypeHeader, "application/json", 1, 1));

  if (err == nullptr) {
    uint32_t flags = 0;
    if (ready) {
      flags |= TRITONSERVER_INDEX_FLAG_READY;
    }

    TRITONSERVER_Message* message = nullptr;
    auto err = TRITONSERVER_ServerModelIndex(server_.get(), flags, &message);
    if (err == nullptr) {
      const char* buffer;
      size_t byte_size;
      err = TRITONSERVER_MessageSerializeToJson(message, &buffer, &byte_size);
      if (err == nullptr) {
        evbuffer_add(req->buffer_out, buffer, byte_size);
        evhtp_send_reply(req, EVHTP_RES_OK);
      }

      TRITONSERVER_MessageDelete(message);
    }
  }

  if (err != nullptr) {
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    TRITONSERVER_ErrorDelete(err);
  }
}

void
HTTPAPIServer::HandleRepositoryControl(
    evhtp_request_t* req, const std::string& repository_name,
    const std::string& model_name, const std::string& action)
{
  if (req->method != htp_method_POST) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  evhtp_headers_add_header(
      req->headers_out,
      evhtp_header_new(kContentTypeHeader, "application/json", 1, 1));

  TRITONSERVER_Error* err = nullptr;
  if (!repository_name.empty()) {
    err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "'repository_name' specification is not supported");
  } else {
    if (action == "load") {
      err = TRITONSERVER_ServerLoadModel(server_.get(), model_name.c_str());
    } else if (action == "unload") {
      // Check if the dependent models should be removed
      bool unload_dependents = false;
      {
        struct evbuffer_iovec* v = nullptr;
        int v_idx = 0;
        int n = evbuffer_peek(req->buffer_in, -1, NULL, NULL, 0);
        if (n > 0) {
          v = static_cast<struct evbuffer_iovec*>(
              alloca(sizeof(struct evbuffer_iovec) * n));
          if (evbuffer_peek(req->buffer_in, -1, NULL, v, n) != n) {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                "unexpected error getting model control request body");
          }
        }

        size_t buffer_len = evbuffer_get_length(req->buffer_in);
        if (buffer_len > 0) {
          triton::common::TritonJson::Value control_request;
          err = EVBufferToJson(&control_request, v, &v_idx, buffer_len, n);
          if (err == nullptr) {
            triton::common::TritonJson::Value params_json;
            if (control_request.Find("parameters", &params_json)) {
              triton::common::TritonJson::Value ud_json;
              if (params_json.Find("unload_dependents", &ud_json)) {
                auto parse_err = ud_json.AsBool(&unload_dependents);
                if (parse_err != nullptr) {
                  err = TRITONSERVER_ErrorNew(
                      TRITONSERVER_ErrorCode(parse_err),
                      (std::string("Unable to parse 'unload_dependents': ") +
                       TRITONSERVER_ErrorMessage(parse_err))
                          .c_str());
                  TRITONSERVER_ErrorDelete(parse_err);
                }
              }
            }
          }
        }
      }
      if (unload_dependents) {
        err = TRITONSERVER_ServerUnloadModelAndDependents(
            server_.get(), model_name.c_str());
      } else {
        err = TRITONSERVER_ServerUnloadModel(server_.get(), model_name.c_str());
      }
    }
  }

  if (err == nullptr) {
    evhtp_send_reply(req, EVHTP_RES_OK);
  } else {
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    TRITONSERVER_ErrorDelete(err);
  }
}

void
HTTPAPIServer::HandleModelReady(
    evhtp_request_t* req, const std::string& model_name,
    const std::string& model_version_str)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  if (model_name.empty()) {
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    return;
  }

  bool ready = false;

  int64_t requested_model_version;
  auto err =
      GetModelVersionFromString(model_version_str, &requested_model_version);
  if (err == nullptr) {
    err = TRITONSERVER_ServerModelIsReady(
        server_.get(), model_name.c_str(), requested_model_version, &ready);
  }

  evhtp_send_reply(
      req, (ready && (err == nullptr)) ? EVHTP_RES_OK : EVHTP_RES_BADREQ);

  TRITONSERVER_ErrorDelete(err);
}

void
HTTPAPIServer::HandleModelMetadata(
    evhtp_request_t* req, const std::string& model_name,
    const std::string& model_version_str)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  if (model_name.empty()) {
    std::string message_json =
        "{ \"error\" : \"missing model name in ModelMetadata request\" }";
    evbuffer_add(req->buffer_out, message_json.c_str(), message_json.size());
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    return;
  }

  evhtp_headers_add_header(
      req->headers_out,
      evhtp_header_new(kContentTypeHeader, "application/json", 1, 1));

  TRITONSERVER_Message* message = nullptr;

  int64_t requested_model_version;
  auto err =
      GetModelVersionFromString(model_version_str, &requested_model_version);
  if (err == nullptr) {
    err = TRITONSERVER_ServerModelMetadata(
        server_.get(), model_name.c_str(), requested_model_version, &message);
    if (err == nullptr) {
      const char* buffer;
      size_t byte_size;
      err = TRITONSERVER_MessageSerializeToJson(message, &buffer, &byte_size);
      if (err == nullptr) {
        evbuffer_add(req->buffer_out, buffer, byte_size);
        evhtp_send_reply(req, EVHTP_RES_OK);
      }
      TRITONSERVER_MessageDelete(message);
    }
  }

  if (err != nullptr) {
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    TRITONSERVER_ErrorDelete(err);
  }
}

void
HTTPAPIServer::HandleModelConfig(
    evhtp_request_t* req, const std::string& model_name,
    const std::string& model_version_str)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  if (model_name.empty()) {
    std::string message_json =
        "{ \"error\" : \"missing model name in ModelConfig request\" }";
    evbuffer_add(req->buffer_out, message_json.c_str(), message_json.size());
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    return;
  }

  evhtp_headers_add_header(
      req->headers_out,
      evhtp_header_new(kContentTypeHeader, "application/json", 1, 1));

  TRITONSERVER_Message* message = nullptr;

  int64_t requested_model_version;
  auto err =
      GetModelVersionFromString(model_version_str, &requested_model_version);
  if (err == nullptr) {
    err = TRITONSERVER_ServerModelConfig(
        server_.get(), model_name.c_str(), requested_model_version,
        1 /* config_version */, &message);
    if (err == nullptr) {
      const char* buffer;
      size_t byte_size;
      err = TRITONSERVER_MessageSerializeToJson(message, &buffer, &byte_size);
      if (err == nullptr) {
        evbuffer_add(req->buffer_out, buffer, byte_size);
        evhtp_send_reply(req, EVHTP_RES_OK);
      }
      TRITONSERVER_MessageDelete(message);
    }
  }

  if (err != nullptr) {
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    TRITONSERVER_ErrorDelete(err);
  }
}

void
HTTPAPIServer::HandleModelStats(
    evhtp_request_t* req, const std::string& model_name,
    const std::string& model_version_str)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  evhtp_headers_add_header(
      req->headers_out,
      evhtp_header_new(kContentTypeHeader, "application/json", 1, 1));

#ifdef TRITON_ENABLE_STATS
  TRITONSERVER_Message* model_stats_message = nullptr;

  int64_t requested_model_version;
  auto err =
      GetModelVersionFromString(model_version_str, &requested_model_version);
  if (err == nullptr) {
    err = TRITONSERVER_ServerModelStatistics(
        server_.get(), model_name.c_str(), requested_model_version,
        &model_stats_message);
    if (err == nullptr) {
      const char* buffer;
      size_t byte_size;
      err = TRITONSERVER_MessageSerializeToJson(
          model_stats_message, &buffer, &byte_size);
      if (err == nullptr) {
        // Add the statistics to the response
        evbuffer_add(req->buffer_out, buffer, byte_size);
        evhtp_send_reply(req, EVHTP_RES_OK);
      }
      TRITONSERVER_MessageDelete(model_stats_message);
    }
  }

#else
  auto err = TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNAVAILABLE,
      "the server does not suppport model statistics");
#endif

  if (err != nullptr) {
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    TRITONSERVER_ErrorDelete(err);
  }
}

void
HTTPAPIServer::HandleServerMetadata(evhtp_request_t* req)
{
  if (req->method != htp_method_GET) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  evhtp_headers_add_header(
      req->headers_out,
      evhtp_header_new(kContentTypeHeader, "application/json", 1, 1));

  if (server_metadata_err_ == nullptr) {
    evbuffer_add(
        req->buffer_out, server_metadata_.c_str(), server_metadata_.size());
    evhtp_send_reply(req, EVHTP_RES_OK);
  } else {
    EVBufferAddErrorJson(req->buffer_out, server_metadata_err_);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
  }
}

void
HTTPAPIServer::HandleSystemSharedMemory(
    evhtp_request_t* req, const std::string& region_name,
    const std::string& action)
{
  if ((action == "status") && (req->method != htp_method_GET)) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  } else if ((action != "status") && (req->method != htp_method_POST)) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  evhtp_headers_add_header(
      req->headers_out,
      evhtp_header_new(kContentTypeHeader, "application/json", 1, 1));

  TRITONSERVER_Error* err = nullptr;
  if (action == "status") {
    triton::common::TritonJson::Value shm_status(
        triton::common::TritonJson::ValueType::ARRAY);
    err = shm_manager_->GetStatus(
        region_name, TRITONSERVER_MEMORY_CPU, &shm_status);
    if (err == nullptr) {
      triton::common::TritonJson::WriteBuffer buffer;
      err = shm_status.Write(&buffer);
      if (err == nullptr) {
        evbuffer_add(req->buffer_out, buffer.Base(), buffer.Size());
      }
    }
  } else if (action == "register") {
    if (region_name.empty()) {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "'region name' is necessary to register system shared memory region");
    } else {
      struct evbuffer_iovec* v = nullptr;
      int v_idx = 0;
      int n = evbuffer_peek(req->buffer_in, -1, NULL, NULL, 0);
      if (n > 0) {
        v = static_cast<struct evbuffer_iovec*>(
            alloca(sizeof(struct evbuffer_iovec) * n));
        if (evbuffer_peek(req->buffer_in, -1, NULL, v, n) != n) {
          err = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              "unexpected error getting register request buffers");
        }
      }

      if (err == nullptr) {
        triton::common::TritonJson::Value register_request;
        size_t buffer_len = evbuffer_get_length(req->buffer_in);
        err = EVBufferToJson(&register_request, v, &v_idx, buffer_len, n);
        if (err == nullptr) {
          triton::common::TritonJson::Value key_json;
          if (!register_request.Find("key", &key_json)) {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                "Shared memory register request has no 'key' field");
          }

          const char* shm_key = nullptr;
          if (err == nullptr) {
            size_t shm_key_len;
            err = key_json.AsString(&shm_key, &shm_key_len);
          }

          uint64_t offset = 0;
          if (err == nullptr) {
            triton::common::TritonJson::Value offset_json;
            if (register_request.Find("offset", &offset_json)) {
              err = offset_json.AsUInt(&offset);
            }
          }

          uint64_t byte_size = 0;
          if (err == nullptr) {
            triton::common::TritonJson::Value byte_size_json;
            if (!register_request.Find("byte_size", &byte_size_json)) {
              err = TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  "Shared memory register request has no 'byte_size' field");
            } else {
              err = byte_size_json.AsUInt(&byte_size);
            }
          }

          if (err == nullptr) {
            err = shm_manager_->RegisterSystemSharedMemory(
                region_name, shm_key, offset, byte_size);
          }
        }
      }
    }
  } else if (action == "unregister") {
    if (region_name.empty()) {
      err = shm_manager_->UnregisterAll(TRITONSERVER_MEMORY_CPU);
    } else {
      err = shm_manager_->Unregister(region_name, TRITONSERVER_MEMORY_CPU);
    }
  }

  if (err == nullptr) {
    evhtp_send_reply(req, EVHTP_RES_OK);
  } else {
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    TRITONSERVER_ErrorDelete(err);
  }
}

void
HTTPAPIServer::HandleCudaSharedMemory(
    evhtp_request_t* req, const std::string& region_name,
    const std::string& action)
{
  if ((action == "status") && (req->method != htp_method_GET)) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  } else if ((action != "status") && (req->method != htp_method_POST)) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  evhtp_headers_add_header(
      req->headers_out,
      evhtp_header_new(kContentTypeHeader, "application/json", 1, 1));

  TRITONSERVER_Error* err = nullptr;
  if (action == "status") {
    triton::common::TritonJson::Value shm_status(
        triton::common::TritonJson::ValueType::ARRAY);
    err = shm_manager_->GetStatus(
        region_name, TRITONSERVER_MEMORY_GPU, &shm_status);
    if (err == nullptr) {
      triton::common::TritonJson::WriteBuffer buffer;
      err = shm_status.Write(&buffer);
      if (err == nullptr) {
        evbuffer_add(req->buffer_out, buffer.Base(), buffer.Size());
      }
    }
  } else if (action == "register") {
    if (region_name.empty()) {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "'region name' is necessary to register cuda shared memory region");
    } else {
#ifdef TRITON_ENABLE_GPU
      struct evbuffer_iovec* v = nullptr;
      int v_idx = 0;
      int n = evbuffer_peek(req->buffer_in, -1, NULL, NULL, 0);
      if (n > 0) {
        v = static_cast<struct evbuffer_iovec*>(
            alloca(sizeof(struct evbuffer_iovec) * n));
        if (evbuffer_peek(req->buffer_in, -1, NULL, v, n) != n) {
          err = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              "unexpected error getting register request buffers");
        }
      }
      if (err == nullptr) {
        triton::common::TritonJson::Value register_request;
        size_t buffer_len = evbuffer_get_length(req->buffer_in);
        err = EVBufferToJson(&register_request, v, &v_idx, buffer_len, n);
        if (err == nullptr) {
          const char* b64_handle = nullptr;
          size_t b64_handle_len = 0;
          triton::common::TritonJson::Value raw_handle_json;
          if (!register_request.Find("raw_handle", &raw_handle_json)) {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                "Shared memory register request has no 'raw_handle' field");
          } else {
            err = raw_handle_json.MemberAsString(
                "b64", &b64_handle, &b64_handle_len);
          }

          uint64_t byte_size = 0;
          if (err == nullptr) {
            err = register_request.MemberAsUInt("byte_size", &byte_size);
          }

          uint64_t device_id = 0;
          if (err == nullptr) {
            err = register_request.MemberAsUInt("device_id", &device_id);
          }

          if (err == nullptr) {
            base64_decodestate s;
            base64_init_decodestate(&s);

            // The decoded can not be larger than the input...
            std::vector<char> raw_handle(b64_handle_len + 1);
            size_t decoded_size = base64_decode_block(
                b64_handle, b64_handle_len, raw_handle.data(), &s);
            if (decoded_size != sizeof(cudaIpcMemHandle_t)) {
              err = TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  "'raw_handle' must be a valid base64 encoded "
                  "cudaIpcMemHandle_t");
            } else {
              raw_handle.resize(sizeof(cudaIpcMemHandle_t));
              err = shm_manager_->RegisterCUDASharedMemory(
                  region_name.c_str(),
                  reinterpret_cast<const cudaIpcMemHandle_t*>(
                      raw_handle.data()),
                  byte_size, device_id);
            }
          }
        }
      }
#else
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              "failed to register CUDA shared memory region: '" + region_name +
              "', GPUs not supported")
              .c_str());
#endif  // TRITON_ENABLE_GPU
    }
  } else if (action == "unregister") {
    if (region_name.empty()) {
      err = shm_manager_->UnregisterAll(TRITONSERVER_MEMORY_GPU);
    } else {
      err = shm_manager_->Unregister(region_name, TRITONSERVER_MEMORY_GPU);
    }
  }

  if (err == nullptr) {
    evhtp_send_reply(req, EVHTP_RES_OK);
  } else {
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    TRITONSERVER_ErrorDelete(err);
  }
}

TRITONSERVER_Error*
HTTPAPIServer::GetInferenceHeaderLength(
    evhtp_request_t* req, int32_t content_length, size_t* header_length)
{
  // Find Inference-Header-Content-Length in header. If missing set to 0
  *header_length = 0;
  const char* header_length_c_str =
      evhtp_kv_find(req->headers_in, kInferHeaderContentLengthHTTPHeader);
  if (header_length_c_str != NULL) {
    int parsed_value;
    try {
      parsed_value = std::atoi(header_length_c_str);
    }
    catch (const std::invalid_argument& ia) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG, (std::string("Unable to parse ") +
                                           kInferHeaderContentLengthHTTPHeader +
                                           ", got: " + header_length_c_str)
                                              .c_str());
    }

    // Check if the content length is in proper range
    if ((parsed_value < 0) || (parsed_value > content_length)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("inference header size should be in range (0, ") +
           std::to_string(content_length) + "), got: " + header_length_c_str)
              .c_str());
    }
    *header_length = parsed_value;
  }
  return nullptr;
}

DataCompressor::Type
HTTPAPIServer::GetRequestCompressionType(evhtp_request_t* req)
{
  const char* content_encoding_c_str =
      evhtp_kv_find(req->headers_in, kContentEncodingHTTPHeader);
  if (content_encoding_c_str != NULL) {
    std::string content_encoding(content_encoding_c_str);
    if (content_encoding == "deflate") {
      return DataCompressor::Type::DEFLATE;
    } else if (content_encoding == "gzip") {
      return DataCompressor::Type::GZIP;
    } else if (!content_encoding.empty() && (content_encoding != "identity")) {
      return DataCompressor::Type::UNKNOWN;
    }
  }
  return DataCompressor::Type::IDENTITY;
}

DataCompressor::Type
HTTPAPIServer::GetResponseCompressionType(evhtp_request_t* req)
{
  // Find Accept-Encoding in header. Try to compress if found
  const char* accept_encoding_c_str =
      evhtp_kv_find(req->headers_in, kAcceptEncodingHTTPHeader);
  if (accept_encoding_c_str != NULL) {
    std::string accept_encoding = CompressionTypeUsed(accept_encoding_c_str);
    if (accept_encoding == "deflate") {
      return DataCompressor::Type::DEFLATE;
    } else if (accept_encoding == "gzip") {
      return DataCompressor::Type::GZIP;
    }
  }
  return DataCompressor::Type::IDENTITY;
}

TRITONSERVER_Error*
HTTPAPIServer::EVBufferToInput(
    const std::string& model_name, TRITONSERVER_InferenceRequest* irequest,
    evbuffer* input_buffer, InferRequestClass* infer_req, size_t header_length)
{
  // Extract individual input data from HTTP body and register in
  // 'irequest'. The HTTP body is not necessarily stored in contiguous
  // memory.
  //
  // Get the addr and size of each chunk of memory holding the HTTP
  // body.
  struct evbuffer_iovec* v = nullptr;
  int v_idx = 0;

  int n = evbuffer_peek(input_buffer, -1, NULL, NULL, 0);
  if (n > 0) {
    v = static_cast<struct evbuffer_iovec*>(
        alloca(sizeof(struct evbuffer_iovec) * n));
    if (evbuffer_peek(input_buffer, -1, NULL, v, n) != n) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "unexpected error getting input buffers");
    }
  }

  // Extract just the json header from the HTTP body. 'header_length'
  // == 0 means that the entire HTTP body should be parsed as json.
  triton::common::TritonJson::Value request_json;
  int json_header_len = 0;
  if (header_length == 0) {
    json_header_len = evbuffer_get_length(input_buffer);
  } else {
    json_header_len = header_length;
  }

  RETURN_IF_ERR(EVBufferToJson(&request_json, v, &v_idx, json_header_len, n));

  // Set InferenceRequest request_id
  triton::common::TritonJson::Value id_json;
  if (request_json.Find("id", &id_json)) {
    const char* id;
    size_t id_len;
    RETURN_MSG_IF_ERR(id_json.AsString(&id, &id_len), "Unable to parse 'id'");
    RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetId(irequest, id));
  }

  // The default setting for returned outputs (JSON or BINARY). This
  // is needed for the case when outputs are not explicitly specified.
  AllocPayload::OutputInfo::Kind default_output_kind =
      AllocPayload::OutputInfo::JSON;

  // Set sequence correlation ID and flags if any
  triton::common::TritonJson::Value params_json;
  if (request_json.Find("parameters", &params_json)) {
    triton::common::TritonJson::Value seq_json;
    if (params_json.Find("sequence_id", &seq_json)) {
      // Try to parse sequence_id as uint64_t
      uint64_t seq_id;
      if (seq_json.AsUInt(&seq_id) != nullptr) {
        // On failure try to parse as a string
        std::string seq_id;
        RETURN_MSG_IF_ERR(
            seq_json.AsString(&seq_id), "Unable to parse 'sequence_id'");
        RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetCorrelationIdString(
            irequest, seq_id.c_str()));
      } else {
        RETURN_IF_ERR(
            TRITONSERVER_InferenceRequestSetCorrelationId(irequest, seq_id));
      }
    }

    uint32_t flags = 0;

    {
      triton::common::TritonJson::Value start_json;
      if (params_json.Find("sequence_start", &start_json)) {
        bool start;
        RETURN_MSG_IF_ERR(
            start_json.AsBool(&start), "Unable to parse 'sequence_start'");
        if (start) {
          flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_START;
        }
      }

      triton::common::TritonJson::Value end_json;
      if (params_json.Find("sequence_end", &end_json)) {
        bool end;
        RETURN_MSG_IF_ERR(
            end_json.AsBool(&end), "Unable to parse 'sequence_end'");
        if (end) {
          flags |= TRITONSERVER_REQUEST_FLAG_SEQUENCE_END;
        }
      }
    }

    RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetFlags(irequest, flags));

    {
      triton::common::TritonJson::Value priority_json;
      if (params_json.Find("priority", &priority_json)) {
        uint64_t p;
        RETURN_MSG_IF_ERR(
            priority_json.AsUInt(&p), "Unable to parse 'priority'");
        RETURN_IF_ERR(TRITONSERVER_InferenceRequestSetPriority(irequest, p));
      }
    }

    {
      triton::common::TritonJson::Value timeout_json;
      if (params_json.Find("timeout", &timeout_json)) {
        uint64_t t;
        RETURN_MSG_IF_ERR(timeout_json.AsUInt(&t), "Unable to parse 'timeout'");
        RETURN_IF_ERR(
            TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(irequest, t));
      }
    }

    {
      triton::common::TritonJson::Value bdo_json;
      if (params_json.Find("binary_data_output", &bdo_json)) {
        bool bdo;
        RETURN_MSG_IF_ERR(
            bdo_json.AsBool(&bdo), "Unable to parse 'binary_data_output'");
        default_output_kind = (bdo) ? AllocPayload::OutputInfo::BINARY
                                    : AllocPayload::OutputInfo::JSON;
      }
    }
  }

  // Get the byte-size for each input and from that get the blocks
  // holding the data for that input
  triton::common::TritonJson::Value inputs_json;
  RETURN_MSG_IF_ERR(
      request_json.MemberAsArray("inputs", &inputs_json),
      "Unable to parse 'inputs'");

  for (size_t i = 0; i < inputs_json.ArraySize(); i++) {
    triton::common::TritonJson::Value request_input;
    RETURN_IF_ERR(inputs_json.At(i, &request_input));
    RETURN_IF_ERR(ValidateInputContentType(request_input));

    const char* input_name;
    size_t input_name_len;
    RETURN_MSG_IF_ERR(
        request_input.MemberAsString("name", &input_name, &input_name_len),
        "Unable to parse 'name'");

    const char* datatype;
    size_t datatype_len;
    RETURN_MSG_IF_ERR(
        request_input.MemberAsString("datatype", &datatype, &datatype_len),
        "Unable to parse 'datatype'");
    const TRITONSERVER_DataType dtype = TRITONSERVER_StringToDataType(datatype);

    triton::common::TritonJson::Value shape_json;
    RETURN_MSG_IF_ERR(
        request_input.MemberAsArray("shape", &shape_json),
        "Unable to parse 'shape'");
    std::vector<int64_t> shape_vec;
    for (size_t i = 0; i < shape_json.ArraySize(); i++) {
      uint64_t d = 0;
      RETURN_MSG_IF_ERR(
          shape_json.IndexAsUInt(i, &d), "Unable to parse 'shape'");
      shape_vec.push_back(d);
    }

    RETURN_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
        irequest, input_name, dtype, &shape_vec[0], shape_vec.size()));

    bool binary_input;
    size_t byte_size;
    RETURN_IF_ERR(
        CheckBinaryInputData(request_input, &binary_input, &byte_size));

    if ((byte_size == 0) && binary_input) {
      RETURN_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
          irequest, input_name, nullptr, 0 /* byte_size */,
          TRITONSERVER_MEMORY_CPU, 0 /* memory_type_id */));
    } else if (binary_input) {
      if (header_length == 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "must specify valid 'Infer-Header-Content-Length' in request "
            "header and 'binary_data_size' when passing inputs in binary "
            "data format");
      }

      // Process one block at a time
      while ((byte_size > 0) && (v_idx < n)) {
        char* base = static_cast<char*>(v[v_idx].iov_base);
        size_t base_size;
        if (v[v_idx].iov_len > byte_size) {
          base_size = byte_size;
          v[v_idx].iov_base = static_cast<void*>(base + byte_size);
          v[v_idx].iov_len -= byte_size;
          byte_size = 0;
        } else {
          base_size = v[v_idx].iov_len;
          byte_size -= v[v_idx].iov_len;
          v_idx++;
        }

        RETURN_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
            irequest, input_name, base, base_size, TRITONSERVER_MEMORY_CPU,
            0 /* memory_type_id */));
      }

      if (byte_size != 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string(
                "unexpected size for input '" + std::string(input_name) +
                "', expecting " + std::to_string(byte_size) +
                " additional bytes for model '" + model_name + "'")
                .c_str());
      }
    } else {
      // Process input if in shared memory.
      bool use_shm;
      uint64_t shm_offset;
      const char* shm_region;
      RETURN_IF_ERR(CheckSharedMemoryData(
          request_input, &use_shm, &shm_region, &shm_offset, &byte_size));
      if (use_shm) {
        void* base;
        TRITONSERVER_MemoryType memory_type;
        int64_t memory_type_id;
        RETURN_IF_ERR(shm_manager_->GetMemoryInfo(
            shm_region, shm_offset, &base, &memory_type, &memory_type_id));
        RETURN_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
            irequest, input_name, base, byte_size, memory_type,
            memory_type_id));
      } else {
        const int64_t element_cnt = GetElementCount(shape_vec);

        // FIXME, element count should never be 0 or negative so
        // shouldn't we just return an error here?
        if (element_cnt == 0) {
          RETURN_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
              irequest, input_name, nullptr, 0 /* byte_size */,
              TRITONSERVER_MEMORY_CPU, 0 /* memory_type_id */));
        } else {
          // JSON... presence of "data" already validated but still
          // checking here. Flow in this endpoint needs to be
          // reworked...
          triton::common::TritonJson::Value tensor_data;
          RETURN_MSG_IF_ERR(
              request_input.MemberAsArray("data", &tensor_data),
              "Unable to parse 'data'");

          if (dtype == TRITONSERVER_TYPE_BYTES) {
            RETURN_IF_ERR(JsonBytesArrayByteSize(tensor_data, &byte_size));
          } else {
            byte_size = element_cnt * TRITONSERVER_DataTypeByteSize(dtype);
          }

          infer_req->serialized_data_.emplace_back();
          std::vector<char>& serialized = infer_req->serialized_data_.back();
          serialized.resize(byte_size);

          RETURN_IF_ERR(ReadDataFromJson(
              input_name, tensor_data, &serialized[0], dtype,
              dtype == TRITONSERVER_TYPE_BYTES ? byte_size : element_cnt));
          RETURN_IF_ERR(TRITONSERVER_InferenceRequestAppendInputData(
              irequest, input_name, &serialized[0], serialized.size(),
              TRITONSERVER_MEMORY_CPU, 0 /* memory_type_id */));
        }
      }
    }
  }

  if (v_idx != n) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected additional input data for model '" + model_name + "'")
            .c_str());
  }

  // outputs is optional
  if (request_json.Find("outputs")) {
    triton::common::TritonJson::Value outputs_json;
    RETURN_MSG_IF_ERR(
        request_json.MemberAsArray("outputs", &outputs_json),
        "Unable to parse 'outputs'");
    for (size_t i = 0; i < outputs_json.ArraySize(); i++) {
      triton::common::TritonJson::Value request_output;
      RETURN_IF_ERR(outputs_json.At(i, &request_output));
      RETURN_IF_ERR(ValidateOutputParameter(request_output));

      const char* output_name;
      size_t output_name_len;
      RETURN_MSG_IF_ERR(
          request_output.MemberAsString("name", &output_name, &output_name_len),
          "Unable to parse 'name'");
      RETURN_IF_ERR(TRITONSERVER_InferenceRequestAddRequestedOutput(
          irequest, output_name));

      uint64_t class_size;
      RETURN_IF_ERR(CheckClassificationOutput(request_output, &class_size));

      bool use_shm;
      uint64_t offset, byte_size;
      const char* shm_region;
      RETURN_IF_ERR(CheckSharedMemoryData(
          request_output, &use_shm, &shm_region, &offset, &byte_size));

      // ValidateOutputParameter ensures that both shm and
      // classification cannot be true.
      if (use_shm) {
        void* base;
        TRITONSERVER_MemoryType memory_type;
        int64_t memory_type_id;
        RETURN_IF_ERR(shm_manager_->GetMemoryInfo(
            shm_region, offset, &base, &memory_type, &memory_type_id));

        infer_req->alloc_payload_.output_map_.emplace(
            std::piecewise_construct, std::forward_as_tuple(output_name),
            std::forward_as_tuple(new AllocPayload::OutputInfo(
                base, byte_size, memory_type, memory_type_id)));
      } else {
        bool use_binary;
        RETURN_IF_ERR(CheckBinaryOutputData(request_output, &use_binary));
        infer_req->alloc_payload_.output_map_.emplace(
            std::piecewise_construct, std::forward_as_tuple(output_name),
            std::forward_as_tuple(new AllocPayload::OutputInfo(
                use_binary ? AllocPayload::OutputInfo::BINARY
                           : AllocPayload::OutputInfo::JSON,
                class_size)));
      }
    }
  }

  infer_req->alloc_payload_.default_output_kind_ = default_output_kind;

  return nullptr;  // success
}

void
HTTPAPIServer::HandleInfer(
    evhtp_request_t* req, const std::string& model_name,
    const std::string& model_version_str)
{
  if (req->method != htp_method_POST) {
    evhtp_send_reply(req, EVHTP_RES_METHNALLOWED);
    return;
  }

  bool connection_paused = false;

  int64_t requested_model_version;
  auto err = GetModelVersionFromString(
      model_version_str.c_str(), &requested_model_version);

  if (err == nullptr) {
    uint32_t txn_flags;
    err = TRITONSERVER_ServerModelTransactionProperties(
        server_.get(), model_name.c_str(), requested_model_version, &txn_flags,
        nullptr /* voidp */);
    if ((err == nullptr) && (txn_flags & TRITONSERVER_TXN_DECOUPLED) != 0) {
      err = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "HTTP end point doesn't support models with decoupled "
          "transaction policy");
    }
  }

  // If tracing is enabled see if this request should be traced.
  TRITONSERVER_InferenceTrace* trace = nullptr;
#ifdef TRITON_ENABLE_TRACING
  uint64_t trace_id = 0;
  if ((err == nullptr) && (trace_manager_ != nullptr)) {
    trace = trace_manager_->SampleTrace();
    if (trace != nullptr) {
      TRITONSERVER_InferenceTraceId(trace, &trace_id);

      // Timestamps from evhtp are capture in 'req'. We record here
      // since this is the first place where we have access to trace
      // manager.
      trace_manager_->CaptureTimestamp(
          trace_id, TRITONSERVER_TRACE_LEVEL_MIN, "HTTP_RECV_START",
          req->recv_start_ns);
      trace_manager_->CaptureTimestamp(
          trace_id, TRITONSERVER_TRACE_LEVEL_MIN, "HTTP_RECV_END",
          req->recv_end_ns);
    }
  }
#endif  // TRITON_ENABLE_TRACING

  // Create the inference request object which provides all information needed
  // for an inference.
  TRITONSERVER_InferenceRequest* irequest = nullptr;
  if (err == nullptr) {
    err = TRITONSERVER_InferenceRequestNew(
        &irequest, server_.get(), model_name.c_str(), requested_model_version);
  }

  // Decompress request body if it is compressed in supported type
  evbuffer* decompressed_buffer = nullptr;
  if (err == nullptr) {
    auto compression_type = GetRequestCompressionType(req);
    switch (compression_type) {
      case DataCompressor::Type::DEFLATE:
      case DataCompressor::Type::GZIP: {
        decompressed_buffer = evbuffer_new();
        err = DataCompressor::DecompressData(
            compression_type, req->buffer_in, decompressed_buffer);
        break;
      }
      case DataCompressor::Type::UNKNOWN: {
        // Encounter unsupported compressed type,
        // send 415 error with supported types in Accept-Encoding
        evhtp_headers_add_header(
            req->headers_out,
            evhtp_header_new(kAcceptEncodingHTTPHeader, "gzip, deflate", 1, 1));
        evhtp_send_reply(req, EVHTP_RES_UNSUPPORTED);
        return;
      }
      case DataCompressor::Type::IDENTITY:
        // Do nothing
        break;
    }
  }

  // Get the header length
  size_t header_length;
  if (err == nullptr) {
    // Set to large value in case there is no Content-Length to compare with
    int32_t content_length = INT32_MAX;
    if (decompressed_buffer == nullptr) {
      const char* content_length_c_str =
          evhtp_kv_find(req->headers_in, kContentLengthHeader);
      if (content_length_c_str != nullptr) {
        try {
          content_length = std::atoi(content_length_c_str);
        }
        catch (const std::invalid_argument& ia) {
          err = TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("Unable to parse ") + kContentLengthHeader +
               ", got: " + content_length_c_str)
                  .c_str());
        }
      }
    } else {
      // The Content-Length doesn't reflect the actual request body size
      // if compression is used, set 'content_length' to the decompressed size
      content_length = evbuffer_get_length(decompressed_buffer);
    }

    if (err == nullptr) {
      err = GetInferenceHeaderLength(req, content_length, &header_length);
    }
  }

  if (err == nullptr) {
    connection_paused = true;

    auto infer_request = CreateInferRequest(req);
#ifdef TRITON_ENABLE_TRACING
    infer_request->trace_manager_ = trace_manager_;
    infer_request->trace_id_ = trace_id;
#endif  // TRITON_ENABLE_TRACING

    if (err == nullptr) {
      err = EVBufferToInput(
          model_name, irequest,
          (decompressed_buffer == nullptr) ? req->buffer_in
                                           : decompressed_buffer,
          infer_request.get(), header_length);
    }
    if (err == nullptr) {
      err = TRITONSERVER_InferenceRequestSetReleaseCallback(
          irequest, InferRequestClass::InferRequestComplete,
          decompressed_buffer);
      if (err == nullptr) {
        err = TRITONSERVER_InferenceRequestSetResponseCallback(
            irequest, allocator_,
            reinterpret_cast<void*>(&infer_request->alloc_payload_),
            InferRequestClass::InferResponseComplete,
            reinterpret_cast<void*>(infer_request.get()));
      }
      if (err == nullptr) {
        err = TRITONSERVER_ServerInferAsync(server_.get(), irequest, trace);
      }
      if (err == nullptr) {
        infer_request.release();
      }
    }
  }

  if (err != nullptr) {
    LOG_VERBOSE(1) << "Infer failed: " << TRITONSERVER_ErrorMessage(err);
    evhtp_headers_add_header(
        req->headers_out,
        evhtp_header_new(kContentTypeHeader, "application/json", 1, 1));
    EVBufferAddErrorJson(req->buffer_out, err);
    evhtp_send_reply(req, EVHTP_RES_BADREQ);
    if (connection_paused) {
      evhtp_request_resume(req);
    }
    TRITONSERVER_ErrorDelete(err);

    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceRequestDelete(irequest),
        "deleting HTTP/REST inference request");
  }
}

void
HTTPAPIServer::OKReplyCallback(evthr_t* thr, void* arg, void* shared)
{
  HTTPAPIServer::InferRequestClass* infer_request =
      reinterpret_cast<HTTPAPIServer::InferRequestClass*>(arg);

  evhtp_request_t* request = infer_request->EvHtpRequest();
  evhtp_send_reply(request, EVHTP_RES_OK);
  evhtp_request_resume(request);

#ifdef TRITON_ENABLE_TRACING
  if ((infer_request->trace_manager_ != nullptr) &&
      (infer_request->trace_id_ != 0)) {
    infer_request->trace_manager_->CaptureTimestamp(
        infer_request->trace_id_, TRITONSERVER_TRACE_LEVEL_MIN,
        "HTTP_SEND_START", request->send_start_ns);
    infer_request->trace_manager_->CaptureTimestamp(
        infer_request->trace_id_, TRITONSERVER_TRACE_LEVEL_MIN, "HTTP_SEND_END",
        request->send_end_ns);
  }
#endif  // TRITON_ENABLE_TRACING

  delete infer_request;
}

void
HTTPAPIServer::BADReplyCallback(evthr_t* thr, void* arg, void* shared)
{
  HTTPAPIServer::InferRequestClass* infer_request =
      reinterpret_cast<HTTPAPIServer::InferRequestClass*>(arg);

  evhtp_request_t* request = infer_request->EvHtpRequest();
  evhtp_send_reply(request, EVHTP_RES_BADREQ);
  evhtp_request_resume(request);

#ifdef TRITON_ENABLE_TRACING
  if ((infer_request->trace_manager_ != nullptr) &&
      (infer_request->trace_id_ != 0)) {
    infer_request->trace_manager_->CaptureTimestamp(
        infer_request->trace_id_, TRITONSERVER_TRACE_LEVEL_MIN,
        "HTTP_SEND_START", request->send_start_ns);
    infer_request->trace_manager_->CaptureTimestamp(
        infer_request->trace_id_, TRITONSERVER_TRACE_LEVEL_MIN, "HTTP_SEND_END",
        request->send_end_ns);
  }
#endif  // TRITON_ENABLE_TRACING

  delete infer_request;
}

HTTPAPIServer::InferRequestClass::InferRequestClass(
    TRITONSERVER_Server* server, evhtp_request_t* req,
    DataCompressor::Type response_compression_type)
    : server_(server), req_(req),
      response_compression_type_(response_compression_type), response_count_(0)
{
  evhtp_connection_t* htpconn = evhtp_request_get_connection(req);
  thread_ = htpconn->thread;
  evhtp_request_pause(req);
}

void
HTTPAPIServer::InferRequestClass::InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  // FIXME need to manage the lifetime of InferRequestClass so that we
  // delete it here.

  if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) != 0) {
    if (userp != nullptr) {
      evbuffer_free(reinterpret_cast<evbuffer*>(userp));
    }
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceRequestDelete(request),
        "deleting HTTP/REST inference request");
  }
}

void
HTTPAPIServer::InferRequestClass::InferResponseComplete(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  // FIXME can't use InferRequestClass object here since it's lifetime
  // is different than response. For response we need to know how to
  // send each output (as json, shm, or binary) and that information
  // has to be maintained in a way that allows us to clean it up
  // appropriately if connection closed or last response sent.
  //
  // But for now userp is the InferRequestClass object and the end of
  // its life is in the OK or BAD ReplyCallback.

  HTTPAPIServer::InferRequestClass* infer_request =
      reinterpret_cast<HTTPAPIServer::InferRequestClass*>(userp);

  auto response_count = infer_request->IncrementResponseCount();

  // Defer to the callback with the final response
  if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) == 0) {
    LOG_ERROR << "[INTERNAL] received a response without FINAL flag";
    return;
  }

  TRITONSERVER_Error* err = nullptr;
  if (response_count != 0) {
    err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, std::string(
                                         "expected a single response, got " +
                                         std::to_string(response_count + 1))
                                         .c_str());
  } else if (response == nullptr) {
    err = TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "received an unexpected null response");
  } else {
    err = infer_request->FinalizeResponse(response);
  }

  if (err == nullptr) {
    evthr_defer(infer_request->thread_, OKReplyCallback, infer_request);
  } else {
    EVBufferAddErrorJson(infer_request->req_->buffer_out, err);
    TRITONSERVER_ErrorDelete(err);
    evthr_defer(infer_request->thread_, BADReplyCallback, infer_request);
  }

  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_InferenceResponseDelete(response),
      "deleting inference response");
}

TRITONSERVER_Error*
HTTPAPIServer::InferRequestClass::FinalizeResponse(
    TRITONSERVER_InferenceResponse* response)
{
  RETURN_IF_ERR(TRITONSERVER_InferenceResponseError(response));

  triton::common::TritonJson::Value response_json(
      triton::common::TritonJson::ValueType::OBJECT);

  const char* request_id;
  RETURN_IF_ERR(TRITONSERVER_InferenceResponseId(response, &request_id));
  if ((request_id != nullptr) && (request_id[0] != '\0')) {
    RETURN_IF_ERR(response_json.AddStringRef("id", request_id));
  }

  const char* model_name;
  int64_t model_version;
  RETURN_IF_ERR(TRITONSERVER_InferenceResponseModel(
      response, &model_name, &model_version));
  RETURN_IF_ERR(response_json.AddStringRef("model_name", model_name));
  RETURN_IF_ERR(response_json.AddString(
      "model_version", std::move(std::to_string(model_version))));

  // If the response has any parameters, convert them to JSON.
  uint32_t parameter_count;
  RETURN_IF_ERR(
      TRITONSERVER_InferenceResponseParameterCount(response, &parameter_count));
  if (parameter_count > 0) {
    triton::common::TritonJson::Value params_json(
        response_json, triton::common::TritonJson::ValueType::OBJECT);

    for (uint32_t pidx = 0; pidx < parameter_count; ++pidx) {
      const char* name;
      TRITONSERVER_ParameterType type;
      const void* vvalue;
      RETURN_IF_ERR(TRITONSERVER_InferenceResponseParameter(
          response, pidx, &name, &type, &vvalue));
      switch (type) {
        case TRITONSERVER_PARAMETER_BOOL:
          RETURN_IF_ERR(params_json.AddBool(
              name, *(reinterpret_cast<const bool*>(vvalue))));
          break;
        case TRITONSERVER_PARAMETER_INT:
          RETURN_IF_ERR(params_json.AddInt(
              name, *(reinterpret_cast<const int64_t*>(vvalue))));
          break;
        case TRITONSERVER_PARAMETER_STRING:
          RETURN_IF_ERR(params_json.AddStringRef(
              name, reinterpret_cast<const char*>(vvalue)));
          break;
      }
    }

    RETURN_IF_ERR(response_json.Add("parameters", std::move(params_json)));
  }

  // Go through each response output and transfer information to JSON
  uint32_t output_count;
  RETURN_IF_ERR(
      TRITONSERVER_InferenceResponseOutputCount(response, &output_count));

  std::vector<evbuffer*> ordered_buffers;
  ordered_buffers.reserve(output_count);

  triton::common::TritonJson::Value response_outputs(
      response_json, triton::common::TritonJson::ValueType::ARRAY);

  for (uint32_t idx = 0; idx < output_count; ++idx) {
    const char* cname;
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint64_t dim_count;
    const void* base;
    size_t byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    void* userp;

    RETURN_IF_ERR(TRITONSERVER_InferenceResponseOutput(
        response, idx, &cname, &datatype, &shape, &dim_count, &base, &byte_size,
        &memory_type, &memory_type_id, &userp));

    triton::common::TritonJson::Value output_json(
        response_json, triton::common::TritonJson::ValueType::OBJECT);
    RETURN_IF_ERR(output_json.AddStringRef("name", cname));

    // Handle data. SHM outputs will not have an info.
    auto info = reinterpret_cast<AllocPayload::OutputInfo*>(userp);

    size_t element_count = 1;
    uint32_t batch_size = 0;

    // If returning output as classification then need to set the
    // datatype and shape based on classification requirements.
    if ((info != nullptr) && (info->class_cnt_ > 0)) {
      // For classification need to determine the batch size, if any,
      // because need to use that to break up the response for each
      // batch entry.
      uint32_t batch_flags;
      RETURN_IF_ERR(TRITONSERVER_ServerModelBatchProperties(
          server_, model_name, model_version, &batch_flags,
          nullptr /* voidp */));
      if ((dim_count > 0) &&
          ((batch_flags & TRITONSERVER_BATCH_FIRST_DIM) != 0)) {
        batch_size = shape[0];
      }

      // Determine the batch1 byte size of the output tensor... needed
      // when the response tensor batch-size > 1 so that we know how
      // to stride though the tensor data.
      size_t batch1_element_count = 1;
      for (size_t sidx = ((batch_size == 0) ? 0 : 1); sidx < dim_count;
           sidx++) {
        batch1_element_count *= shape[sidx];
      }

      const size_t batch1_byte_size =
          batch1_element_count * TRITONSERVER_DataTypeByteSize(datatype);

      // Create the classification contents
      std::string serialized;

      size_t class_offset = 0;
      for (uint32_t bs = 0; bs < std::max((uint32_t)1, batch_size); ++bs) {
        std::vector<std::string> class_strs;
        RETURN_IF_ERR(TopkClassifications(
            response, idx, reinterpret_cast<const char*>(base) + class_offset,
            ((class_offset + batch1_byte_size) > byte_size) ? 0
                                                            : batch1_byte_size,
            datatype, info->class_cnt_, &class_strs));

        // Serialize for binary representation...
        for (const auto& str : class_strs) {
          uint32_t len = str.size();
          serialized.append(reinterpret_cast<const char*>(&len), sizeof(len));
          if (len > 0) {
            serialized.append(str);
          }
        }

        class_offset += batch1_byte_size;
      }

      // Replace existing output with serialized classification output.
      const char* datatype_str =
          TRITONSERVER_DataTypeString(TRITONSERVER_TYPE_BYTES);
      RETURN_IF_ERR(output_json.AddStringRef("datatype", datatype_str));

      triton::common::TritonJson::Value shape_json(
          response_json, triton::common::TritonJson::ValueType::ARRAY);
      if (batch_size > 0) {
        RETURN_IF_ERR(shape_json.AppendUInt(batch_size));
        element_count *= batch_size;
      }
      size_t actual_class_count =
          std::min((size_t)info->class_cnt_, batch1_element_count);
      element_count *= actual_class_count;
      RETURN_IF_ERR(shape_json.AppendUInt(actual_class_count));
      RETURN_IF_ERR(output_json.Add("shape", std::move(shape_json)));

      evbuffer_free(info->evbuffer_);
      info->evbuffer_ = nullptr;

      void* buffer;
      byte_size = serialized.size();
      RETURN_IF_ERR(AllocEVBuffer(byte_size, &info->evbuffer_, &buffer));
      memcpy(buffer, serialized.c_str(), byte_size);
      base = reinterpret_cast<const void*>(buffer);
      datatype = TRITONSERVER_TYPE_BYTES;
    } else {
      const char* datatype_str = TRITONSERVER_DataTypeString(datatype);
      RETURN_IF_ERR(output_json.AddStringRef("datatype", datatype_str));

      triton::common::TritonJson::Value shape_json(
          response_json, triton::common::TritonJson::ValueType::ARRAY);
      for (size_t j = 0; j < dim_count; j++) {
        RETURN_IF_ERR(shape_json.AppendUInt(shape[j]));
        element_count *= shape[j];
      }

      RETURN_IF_ERR(output_json.Add("shape", std::move(shape_json)));
    }

    // Add JSON data, or collect binary data. If 'info' is nullptr
    // then using shared memory so don't need this step.
    if (info != nullptr) {
      if (info->kind_ == AllocPayload::OutputInfo::BINARY) {
        triton::common::TritonJson::Value parameters_json;
        if (!output_json.Find("parameters", &parameters_json)) {
          parameters_json = triton::common::TritonJson::Value(
              response_json, triton::common::TritonJson::ValueType::OBJECT);
          RETURN_IF_ERR(parameters_json.AddUInt("binary_data_size", byte_size));
          RETURN_IF_ERR(
              output_json.Add("parameters", std::move(parameters_json)));
        } else {
          RETURN_IF_ERR(parameters_json.AddUInt("binary_data_size", byte_size));
        }
        if (byte_size > 0) {
          ordered_buffers.push_back(info->evbuffer_);
        }
      } else {
        triton::common::TritonJson::Value data_json(
            response_json, triton::common::TritonJson::ValueType::ARRAY);
        RETURN_IF_ERR(WriteDataToJson(
            &data_json, cname, datatype, base, byte_size, element_count));
        RETURN_IF_ERR(output_json.Add("data", std::move(data_json)));
      }
    }

    RETURN_IF_ERR(response_outputs.Append(std::move(output_json)));
  }

  RETURN_IF_ERR(response_json.Add("outputs", std::move(response_outputs)));

  evbuffer* response_placeholder = evbuffer_new();
  // Write json metadata into response evbuffer
  triton::common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERR(response_json.Write(&buffer));
  evbuffer_add(response_placeholder, buffer.Base(), buffer.Size());

  // If there is binary data write it next in the appropriate
  // order... also need the HTTP header when returning binary data.
  if (!ordered_buffers.empty()) {
    for (evbuffer* b : ordered_buffers) {
      evbuffer_add_buffer(response_placeholder, b);
    }
  }

  evbuffer* response_body = response_placeholder;
  switch (response_compression_type_) {
    case DataCompressor::Type::DEFLATE:
    case DataCompressor::Type::GZIP: {
      auto compressed_buffer = evbuffer_new();
      auto err = DataCompressor::CompressData(
          response_compression_type_, response_placeholder, compressed_buffer);
      if (err == nullptr) {
        response_body = compressed_buffer;
        evbuffer_free(response_placeholder);
      } else {
        // just log the compression error and return the uncompressed data
        LOG_VERBOSE(1) << "unable to compress response: "
                       << TRITONSERVER_ErrorMessage(err);
        TRITONSERVER_ErrorDelete(err);
        evbuffer_free(compressed_buffer);
        response_compression_type_ = DataCompressor::Type::IDENTITY;
      }
      break;
    }
    case DataCompressor::Type::IDENTITY:
    case DataCompressor::Type::UNKNOWN:
      // Do nothing for other cases
      break;
  }
  SetResponseHeader(!ordered_buffers.empty(), buffer.Size());
  evbuffer_add_buffer(req_->buffer_out, response_body);
  // Destroy the evbuffer object as the data has been moved
  // to HTTP response buffer
  evbuffer_free(response_body);

  return nullptr;  // success
}

void
HTTPAPIServer::InferRequestClass::SetResponseHeader(
    bool has_binary_data, size_t header_length)
{
  if (has_binary_data) {
    evhtp_headers_add_header(
        req_->headers_out,
        evhtp_header_new(kContentTypeHeader, "application/octet-stream", 1, 1));
    evhtp_headers_add_header(
        req_->headers_out, evhtp_header_new(
                               kInferHeaderContentLengthHTTPHeader,
                               std::to_string(header_length).c_str(), 1, 1));
  } else {
    evhtp_headers_add_header(
        req_->headers_out,
        evhtp_header_new(kContentTypeHeader, "application/json", 1, 1));
  }

  switch (response_compression_type_) {
    case DataCompressor::Type::DEFLATE:
      evhtp_headers_add_header(
          req_->headers_out,
          evhtp_header_new(kContentEncodingHTTPHeader, "deflate", 1, 1));
      break;
    case DataCompressor::Type::GZIP:
      evhtp_headers_add_header(
          req_->headers_out,
          evhtp_header_new(kContentEncodingHTTPHeader, "gzip", 1, 1));
      break;
    case DataCompressor::Type::IDENTITY:
    case DataCompressor::Type::UNKNOWN:
      break;
  }
}

uint32_t
HTTPAPIServer::InferRequestClass::IncrementResponseCount()
{
  return response_count_++;
}


void
HTTPAPIServer::Handle(evhtp_request_t* req)
{
  LOG_VERBOSE(1) << "HTTP request: " << req->method << " "
                 << req->uri->path->full;

  if (std::string(req->uri->path->full) == "/v2/models/stats") {
    // model statistics
    HandleModelStats(req);
    return;
  }

  std::string model_name, version, kind;
  if (RE2::FullMatch(
          std::string(req->uri->path->full), model_regex_, &model_name,
          &version, &kind)) {
    if (kind == "ready") {
      // model ready
      HandleModelReady(req, model_name, version);
      return;
    } else if (kind == "infer") {
      // model infer
      HandleInfer(req, model_name, version);
      return;
    } else if (kind == "config") {
      // model configuration
      HandleModelConfig(req, model_name, version);
      return;
    } else if (kind == "stats") {
      // model statistics
      HandleModelStats(req, model_name, version);
      return;
    } else if (kind == "") {
      // model metadata
      HandleModelMetadata(req, model_name, version);
      return;
    }
  }

  std::string region, action, rest, repo_name;
  if (std::string(req->uri->path->full) == "/v2") {
    // server metadata
    HandleServerMetadata(req);
    return;
  } else if (RE2::FullMatch(
                 std::string(req->uri->path->full), server_regex_, &rest)) {
    // server health
    HandleServerHealth(req, rest);
    return;
  } else if (RE2::FullMatch(
                 std::string(req->uri->path->full), systemsharedmemory_regex_,
                 &region, &action)) {
    // system shared memory
    HandleSystemSharedMemory(req, region, action);
    return;
  } else if (RE2::FullMatch(
                 std::string(req->uri->path->full), cudasharedmemory_regex_,
                 &region, &action)) {
    // cuda shared memory
    HandleCudaSharedMemory(req, region, action);
    return;
  } else if (RE2::FullMatch(
                 std::string(req->uri->path->full), modelcontrol_regex_,
                 &repo_name, &kind, &model_name, &action)) {
    // model repository
    if (kind == "index") {
      HandleRepositoryIndex(req, repo_name);
      return;
    } else if (kind.find("models", 0) == 0) {
      HandleRepositoryControl(req, repo_name, model_name, action);
      return;
    }
  }

  LOG_VERBOSE(1) << "HTTP error: " << req->method << " " << req->uri->path->full
                 << " - " << static_cast<int>(EVHTP_RES_BADREQ);

  evhtp_send_reply(req, EVHTP_RES_BADREQ);
}

TRITONSERVER_Error*
HTTPAPIServer::Create(
    const std::shared_ptr<TRITONSERVER_Server>& server,
    nvidia::inferenceserver::TraceManager* trace_manager,
    const std::shared_ptr<SharedMemoryManager>& shm_manager, const int32_t port,
    const int thread_cnt, std::unique_ptr<HTTPServer>* http_server)
{
  http_server->reset(
      new HTTPAPIServer(server, trace_manager, shm_manager, port, thread_cnt));

  const std::string addr = "0.0.0.0:" + std::to_string(port);
  LOG_INFO << "Started HTTPService at " << addr;

  return nullptr;
}

}}  // namespace nvidia::inferenceserver
