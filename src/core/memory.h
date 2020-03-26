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

#include "src/core/constants.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

//
// Memory used to access data in inference requests
//
class Memory {
 public:
  // Get the 'idx'-th data block in the buffer. Using index to avoid
  // maintaining internal state such that one buffer can be shared
  // across multiple providers.
  // 'idx' zero base index. Valid indices are continuous.
  // 'byte_size' returns the byte size of the chunk of bytes.
  // 'memory_type' returns the memory type of the chunk of bytes.
  // 'memory_type_id' returns the memory type id of the chunk of bytes.
  // Return the pointer to the data block. Returns nullptr if 'idx' is
  // out of range
  virtual const char* BufferAt(
      size_t idx, size_t* byte_size, TRTSERVER_Memory_Type* memory_type,
      int64_t* memory_type_id) const = 0;

  // Get the number of contiguous buffers composing the memory.
  size_t BufferCount() const { return buffer_count_; }

  // Return the total byte size of the data buffer
  size_t TotalByteSize() const { return total_byte_size_; }

 protected:
  Memory() : total_byte_size_(0), buffer_count_(0) {}
  size_t total_byte_size_;
  size_t buffer_count_;
};

//
// MemoryReference
//
class MemoryReference : public Memory {
 public:
  // Create a read-only data buffer as a reference to other data buffer
  MemoryReference();

  //\see Memory::BufferAt()
  const char* BufferAt(
      size_t idx, size_t* byte_size, TRTSERVER_Memory_Type* memory_type,
      int64_t* memory_type_id) const override;

  // Add a 'buffer' with 'byte_size' as part of this data buffer
  // Return the index of the buffer
  size_t AddBuffer(
      const char* buffer, size_t byte_size, TRTSERVER_Memory_Type memory_type,
      int64_t memory_type_id);

 private:
  struct Block {
    Block(
        const char* buffer, size_t byte_size, TRTSERVER_Memory_Type memory_type,
        int64_t memory_type_id)
        : buffer_(buffer), byte_size_(byte_size), memory_type_(memory_type),
          memory_type_id_(memory_type_id)
    {
    }
    const char* buffer_;
    size_t byte_size_;
    TRTSERVER_Memory_Type memory_type_;
    int64_t memory_type_id_;
  };
  std::vector<Block> buffer_;
};

//
// MutableMemory
//
class MutableMemory : public Memory {
 public:
  // Create a mutable data buffer referencing to other data buffer.
  MutableMemory(
      char* buffer, size_t byte_size, TRTSERVER_Memory_Type memory_type,
      int64_t memory_type_id);

  virtual ~MutableMemory() {}

  //\see Memory::BufferAt()
  const char* BufferAt(
      size_t idx, size_t* byte_size, TRTSERVER_Memory_Type* memory_type,
      int64_t* memory_type_id) const override;

  // Return a pointer to the base address of the mutable buffer. If
  // non-null 'memory_type' returns the memory type of the chunk of
  // bytes. If non-null 'memory_type_id' returns the memory type id of
  // the chunk of bytes.
  char* MutableBuffer(
      TRTSERVER_Memory_Type* memory_type = nullptr,
      int64_t* memory_type_id = nullptr);

  DISALLOW_COPY_AND_ASSIGN(MutableMemory);

 protected:
  MutableMemory() : Memory() {}

  char* buffer_;
  TRTSERVER_Memory_Type memory_type_;
  int64_t memory_type_id_;
};

//
// AllocatedMemory
//
class AllocatedMemory : public MutableMemory {
 public:
  // Create a continuous data buffer with 'byte_size', 'memory_type' and
  // 'memory_type_id'. Note that the buffer may be created on different memeory
  // type and memory type id if the original request type and id can not be
  // satisfied, thus the function caller should always check the actual memory
  // type and memory type id before use.
  AllocatedMemory(
      size_t byte_size, TRTSERVER_Memory_Type memory_type,
      int64_t memory_type_id);

  ~AllocatedMemory() override;
};

}}  // namespace nvidia::inferenceserver
