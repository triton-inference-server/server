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

#include "src/core/shared_library.h"

#include "mutex"
#include "src/core/filesystem.h"
#include "src/core/logging.h"

#ifdef _WIN32
// suppress the min and max definitions in Windef.h.
#define NOMINMAX
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace nvidia { namespace inferenceserver {

static std::mutex mu_;

Status
SharedLibrary::Acquire(std::unique_ptr<SharedLibrary>* slib)
{
  mu_.lock();
  slib->reset(new SharedLibrary());
  return Status::Success;
}

SharedLibrary::~SharedLibrary()
{
  mu_.unlock();
}

Status
SharedLibrary::SetLibraryDirectory(const std::string& path)
{
#ifdef _WIN32
  LOG_VERBOSE(1) << "SetLibraryDirectory: path = " << path;
  if (!SetDllDirectory(path.c_str())) {
    LPSTR err_buffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&err_buffer, 0, NULL);
    std::string errstr(err_buffer, size);
    LocalFree(err_buffer);

    return Status(
        Status::Code::NOT_FOUND,
        "unable to set dll path " + path + ": " + errstr);
  }
#endif

  return Status::Success;
}

Status
SharedLibrary::ResetLibraryDirectory()
{
#ifdef _WIN32
  LOG_VERBOSE(1) << "ResetLibraryDirectory";
  if (!SetDllDirectory(NULL)) {
    LPSTR err_buffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&err_buffer, 0, NULL);
    std::string errstr(err_buffer, size);
    LocalFree(err_buffer);

    return Status(
        Status::Code::NOT_FOUND, "unable to reset dll path: " + errstr);
  }
#endif

  return Status::Success;
}

Status
SharedLibrary::OpenLibraryHandle(const std::string& path, void** handle)
{
  LOG_VERBOSE(1) << "OpenLibraryHandle: " << path;

#ifdef _WIN32
  // Need to put shared library directory on the DLL path so that any
  // dependencies of the shared library are found
  const std::string library_dir = DirName(path);
  RETURN_IF_ERROR(SetLibraryDirectory(library_dir));

  // HMODULE is typedef of void*
  // https://docs.microsoft.com/en-us/windows/win32/winprog/windows-data-types
  LOG_VERBOSE(1) << "OpenLibraryHandle: path = " << path;
  *handle = LoadLibrary(path.c_str());

  // Remove the dll path added above... do this unconditionally before
  // check for failure in dll load.
  RETURN_IF_ERROR(ResetLibraryDirectory());

  if (*handle == nullptr) {
    LPSTR err_buffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&err_buffer, 0, NULL);
    std::string errstr(err_buffer, size);
    LocalFree(err_buffer);

    return Status(
        Status::Code::NOT_FOUND, "unable to load shared library: " + errstr);
  }
#else
  *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (*handle == nullptr) {
    return Status(
        Status::Code::NOT_FOUND,
        "unable to load shared library: " + std::string(dlerror()));
  }
#endif

  return Status::Success;
}

Status
SharedLibrary::CloseLibraryHandle(void* handle)
{
  if (handle != nullptr) {
#ifdef _WIN32
    if (FreeLibrary((HMODULE)handle) == 0) {
      LPSTR err_buffer = nullptr;
      size_t size = FormatMessageA(
          FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
              FORMAT_MESSAGE_IGNORE_INSERTS,
          NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
          (LPSTR)&err_buffer, 0, NULL);
      std::string errstr(err_buffer, size);
      LocalFree(err_buffer);
      return Status(
          Status::Code::INTERNAL, "unable to unload shared library: " + errstr);
    }
#else
    if (dlclose(handle) != 0) {
      return Status(
          Status::Code::INTERNAL,
          "unable to unload shared library: " + std::string(dlerror()));
    }
#endif
  }

  return Status::Success;
}

Status
SharedLibrary::GetEntrypoint(
    void* handle, const std::string& name, const bool optional, void** befn)
{
  *befn = nullptr;

#ifdef _WIN32
  void* fn = GetProcAddress((HMODULE)handle, name.c_str());
  if ((fn == nullptr) && !optional) {
    LPSTR err_buffer = nullptr;
    size_t size = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&err_buffer, 0, NULL);
    std::string errstr(err_buffer, size);
    LocalFree(err_buffer);
    return Status(
        Status::Code::NOT_FOUND,
        "unable to find '" + name +
            "' entrypoint in custom library: " + errstr);
  }
#else
  dlerror();
  void* fn = dlsym(handle, name.c_str());
  const char* dlsym_error = dlerror();
  if (dlsym_error != nullptr) {
    if (optional) {
      return Status::Success;
    }

    std::string errstr(dlsym_error);  // need copy as dlclose overwrites
    return Status(
        Status::Code::NOT_FOUND, "unable to find required entrypoint '" + name +
                                     "' in shared library: " + errstr);
  }

  if (fn == nullptr) {
    if (optional) {
      return Status::Success;
    }

    return Status(
        Status::Code::NOT_FOUND,
        "unable to find required entrypoint '" + name + "' in shared library");
  }
#endif

  *befn = fn;
  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
