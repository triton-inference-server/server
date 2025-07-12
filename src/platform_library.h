// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>
#ifdef __linux__
#include <dlfcn.h>
#elif defined(__APPLE__)
#include <dlfcn.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace triton { namespace core {

// Platform-specific library handling utilities

// Get the platform-specific shared library extension
inline const char* GetSharedLibraryExtension()
{
#ifdef __APPLE__
  return ".dylib";
#elif defined(_WIN32)
  return ".dll";
#else
  return ".so";
#endif
}

// Get the platform-specific library name prefix
inline const char* GetSharedLibraryPrefix()
{
#ifdef _WIN32
  return "";
#else
  return "lib";
#endif
}

// Convert a base library name to platform-specific format
// e.g., "tritoncache" -> "libtritoncache.so" (Linux)
//                     -> "libtritoncache.dylib" (macOS)
//                     -> "tritoncache.dll" (Windows)
inline std::string GetPlatformLibraryName(const std::string& base_name)
{
  return GetSharedLibraryPrefix() + base_name + GetSharedLibraryExtension();
}

// Get platform-specific dlopen flags
inline int GetPlatformDlopenFlags()
{
#ifdef __APPLE__
  // macOS prefers RTLD_LOCAL to avoid symbol conflicts
  return RTLD_NOW | RTLD_LOCAL;
#else
  // Linux typically uses RTLD_GLOBAL for plugin systems
  return RTLD_NOW | RTLD_GLOBAL;
#endif
}

// Platform-specific library path handling
inline std::string GetPlatformLibraryPath(const std::string& dir, const std::string& library_name)
{
  std::string path = dir;
  if (!path.empty() && path.back() != '/') {
    path += '/';
  }
  
#ifdef __APPLE__
  // macOS may use @rpath, @loader_path, or @executable_path
  // For now, we'll use standard paths, but this can be extended
  // to handle special macOS paths if needed
#endif
  
  path += library_name;
  return path;
}

}}  // namespace triton::core