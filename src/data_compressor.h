// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <event2/buffer.h>
#include <zlib.h>
#include "common.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace server {

//
// DataCompressor
//
class DataCompressor {
 public:
  enum class Type { UNKNOWN, IDENTITY, GZIP, DEFLATE };

  // Specialization where the source and destination buffer are stored as
  // evbuffer
  static TRITONSERVER_Error* CompressData(
      const Type type, evbuffer* source, evbuffer* compressed_data)
  {
    size_t expected_compressed_size = evbuffer_get_length(source);
    // nothing to be compressed
    if (expected_compressed_size == 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG, "nothing to be compressed");
    }

    z_stream stream;
    stream.zalloc = Z_NULL;
    stream.zfree = Z_NULL;
    stream.opaque = Z_NULL;
    switch (type) {
      case Type::UNKNOWN:
      case Type::IDENTITY: {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG, "nothing to be compressed");
      }
      case Type::GZIP:
        if (deflateInit2(
                &stream, Z_DEFAULT_COMPRESSION /* level */,
                Z_DEFLATED /* method */, 15 | 16 /* windowBits */,
                8 /* memLevel */, Z_DEFAULT_STRATEGY /* strategy */) != Z_OK) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              "failed to initialize state for gzip data compression");
        }
        break;
      case Type::DEFLATE: {
        if (deflateInit(&stream, Z_DEFAULT_COMPRESSION /* level */) != Z_OK) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              "failed to initialize state for deflate data compression");
        }
        break;
      }
    }
    // ensure the internal state are cleaned up on function return
    std::unique_ptr<z_stream, decltype(&deflateEnd)> managed_stream(
        &stream, deflateEnd);

    // Get the addr and size of each chunk of memory in 'source'
    struct evbuffer_iovec* buffer_array = nullptr;
    int buffer_count = evbuffer_peek(source, -1, NULL, NULL, 0);
    if (buffer_count > 0) {
      buffer_array = static_cast<struct evbuffer_iovec*>(
          alloca(sizeof(struct evbuffer_iovec) * buffer_count));
      if (evbuffer_peek(source, -1, NULL, buffer_array, buffer_count) !=
          buffer_count) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            "unexpected error getting buffers to be compressed");
      }
    }
    // Reserve the same size as source for compressed data, it is less likely
    // that a negative compression happens.
    struct evbuffer_iovec current_reserved_space;
    RETURN_MSG_IF_ERR(
        AllocEVBuffer(
            expected_compressed_size, compressed_data, &current_reserved_space),
        "unexpected error allocating output buffer for compression: ");
    stream.next_out =
        reinterpret_cast<unsigned char*>(current_reserved_space.iov_base);
    stream.avail_out = expected_compressed_size;

    // Compress until end of 'source'
    for (int idx = 0; idx < buffer_count; ++idx) {
      stream.next_in =
          reinterpret_cast<unsigned char*>(buffer_array[idx].iov_base);
      stream.avail_in = buffer_array[idx].iov_len;

      // run deflate() on input until source has been read in
      do {
        // Need additional buffer
        if (stream.avail_out == 0) {
          RETURN_MSG_IF_ERR(
              CommitEVBuffer(
                  compressed_data, &current_reserved_space,
                  expected_compressed_size),
              "unexpected error comitting output buffer for compression: ");
          RETURN_MSG_IF_ERR(
              AllocEVBuffer(
                  expected_compressed_size, compressed_data,
                  &current_reserved_space),
              "unexpected error allocating output buffer for compression: ");
          stream.next_out =
              reinterpret_cast<unsigned char*>(current_reserved_space.iov_base);
          stream.avail_out = expected_compressed_size;
        }
        auto flush = ((idx + 1) != buffer_count) ? Z_NO_FLUSH : Z_FINISH;
        auto ret = deflate(&stream, flush);
        if (ret == Z_STREAM_ERROR) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              "encountered inconsistent stream state during compression");
        }
      } while (stream.avail_out == 0);
    }
    // Make sure the last buffer is committed
    if (current_reserved_space.iov_base != nullptr) {
      RETURN_MSG_IF_ERR(
          CommitEVBuffer(
              compressed_data, &current_reserved_space,
              expected_compressed_size - stream.avail_out),
          "unexpected error comitting output buffer for compression: ");
    }
    return nullptr;  // success
  }

  static TRITONSERVER_Error* DecompressData(
      const Type type, evbuffer* source, evbuffer* decompressed_data)
  {
    size_t source_byte_size = evbuffer_get_length(source);
    // nothing to be decompressed
    if (evbuffer_get_length(source) == 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG, "nothing to be decompressed");
    }
    // Set reasonable size for each output buffer to be allocated
    size_t output_buffer_size = (source_byte_size > (1 << 20 /* 1MB */))
                                    ? source_byte_size
                                    : (1 << 20 /* 1MB */);

    switch (type) {
      case Type::UNKNOWN:
      case Type::IDENTITY: {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG, "nothing to be decompressed");
      }
      case Type::GZIP:
      case Type::DEFLATE:
        // zlib can automatically detect compression type
        {
          z_stream stream;
          stream.zalloc = Z_NULL;
          stream.zfree = Z_NULL;
          stream.opaque = Z_NULL;
          stream.avail_in = 0;
          stream.next_in = Z_NULL;

          if (inflateInit2(&stream, 15 | 32) != Z_OK) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                "failed to initialize state for data decompression");
          }
          // ensure the internal state are cleaned up on function return
          std::unique_ptr<z_stream, decltype(&inflateEnd)> managed_stream(
              &stream, inflateEnd);

          // Get the addr and size of each chunk of memory in 'source'
          struct evbuffer_iovec* buffer_array = nullptr;
          int buffer_count = evbuffer_peek(source, -1, NULL, NULL, 0);
          if (buffer_count > 0) {
            buffer_array = static_cast<struct evbuffer_iovec*>(
                alloca(sizeof(struct evbuffer_iovec) * buffer_count));
            if (evbuffer_peek(source, -1, NULL, buffer_array, buffer_count) !=
                buffer_count) {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INTERNAL,
                  "unexpected error getting buffers to be decompressed");
            }
          }
          // Reserve the same size as source for compressed data, it is less
          // likely that a negative compression happens.
          struct evbuffer_iovec current_reserved_space;
          RETURN_MSG_IF_ERR(
              AllocEVBuffer(
                  output_buffer_size, decompressed_data,
                  &current_reserved_space),
              "unexpected error allocating output buffer for decompression: ");
          stream.next_out =
              reinterpret_cast<unsigned char*>(current_reserved_space.iov_base);
          stream.avail_out = output_buffer_size;

          // Compress until end of 'source'
          for (int idx = 0; idx < buffer_count; ++idx) {
            stream.next_in =
                reinterpret_cast<unsigned char*>(buffer_array[idx].iov_base);
            stream.avail_in = buffer_array[idx].iov_len;

            // run inflate() on input until source has been read in
            do {
              // Need additional buffer
              if (stream.avail_out == 0) {
                RETURN_MSG_IF_ERR(
                    CommitEVBuffer(
                        decompressed_data, &current_reserved_space,
                        output_buffer_size),
                    "unexpected error comitting output buffer for "
                    "decompression: ");
                RETURN_MSG_IF_ERR(
                    AllocEVBuffer(
                        output_buffer_size, decompressed_data,
                        &current_reserved_space),
                    "unexpected error allocating output buffer for "
                    "decompression: ");
                stream.next_out = reinterpret_cast<unsigned char*>(
                    current_reserved_space.iov_base);
                stream.avail_out = output_buffer_size;
              }
              auto ret = inflate(&stream, Z_NO_FLUSH);
              if (ret == Z_STREAM_ERROR) {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INTERNAL,
                    "encountered inconsistent stream state during "
                    "decompression");
              }
            } while (stream.avail_out == 0);
          }
          // Make sure the last buffer is committed
          if (current_reserved_space.iov_base != nullptr) {
            RETURN_MSG_IF_ERR(
                CommitEVBuffer(
                    decompressed_data, &current_reserved_space,
                    output_buffer_size - stream.avail_out),
                "unexpected error comitting output buffer for compression: ");
          }
          break;
        }
    }
    return nullptr;  // success
  }

 private:
  static TRITONSERVER_Error* AllocEVBuffer(
      const size_t byte_size, evbuffer* evb,
      struct evbuffer_iovec* current_reserved_space)
  {
    // Reserve requested space in evbuffer...
    if ((evbuffer_reserve_space(evb, byte_size, current_reserved_space, 1) !=
         1) ||
        (current_reserved_space->iov_len < byte_size)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          std::string(
              "failed to reserve " + std::to_string(byte_size) +
              " bytes in evbuffer")
              .c_str());
    }
    return nullptr;  // success
  }

  static TRITONSERVER_Error* CommitEVBuffer(
      evbuffer* evb, struct evbuffer_iovec* current_reserved_space,
      const size_t filled_byte_size)
  {
    current_reserved_space->iov_len = filled_byte_size;
    if (evbuffer_commit_space(evb, current_reserved_space, 1) != 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "failed to commit allocated evbuffer");
    }
    current_reserved_space->iov_base = nullptr;
    return nullptr;  // success
  }
};

}}  // namespace triton::server
