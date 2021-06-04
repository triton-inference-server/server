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

#include <memory>
#include <string>
#include "src/core/constants.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

// SharedLibrary
//
// Utility functions for shared libraries. Because some operations
// require serialization, this object cannot be directly constructed
// and must instead be accessed using Acquire().
class SharedLibrary {
 public:
  // Acquire a SharedLibrary object exclusively. Any other attempts to
  // concurrently acquire a SharedLibrary object will block.
  // object. Ownership is released by destroying the SharedLibrary
  // object.
  static Status Acquire(std::unique_ptr<SharedLibrary>* slib);

  ~SharedLibrary();

  // Configuration so that dependent libraries will be searched for in
  // 'path' during OpenLibraryHandle.
  Status SetLibraryDirectory(const std::string& path);

  // Reset any configuration done by SetLibraryDirectory.
  Status ResetLibraryDirectory();

  // Open shared library and return generic handle.
  Status OpenLibraryHandle(const std::string& path, void** handle);

  // Close shared library.
  Status CloseLibraryHandle(void* handle);

  // Get a generic pointer for an entrypoint into a shared library.
  Status GetEntrypoint(
      void* handle, const std::string& name, const bool optional, void** befn);

 private:
  DISALLOW_COPY_AND_ASSIGN(SharedLibrary);
  explicit SharedLibrary() = default;
};

}}  // namespace nvidia::inferenceserver
