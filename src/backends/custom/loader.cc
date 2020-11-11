// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "src/backends/custom/loader.h"

#ifdef _WIN32
// suppress the min and max definitions in Windef.h.
#define NOMINMAX
#include <Windows.h>
#else
#include <dlfcn.h>
#endif
#include "src/core/logging.h"

namespace nvidia { namespace inferenceserver {

namespace {

Status
OpenLibraryHandle(const std::string& path, void** handle)
{
#ifdef _WIN32
  // HMODULE is typedef of void*
  // https://docs.microsoft.com/en-us/windows/win32/winprog/windows-data-types
  *handle = LoadLibrary(path.c_str());
  if (*handle == nullptr) {
    LPSTR err_buffer = nullptr;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&err_buffer, 0, NULL);
    std::string errstr(err_buffer, size);
    LocalFree(err_buffer);

    return Status(
        Status::Code::NOT_FOUND,
        "unable to load custom library: " + errstr);
  }
#else
  *handle = dlopen(path.c_str(), RTLD_LAZY);
  if (*handle == nullptr) {
    return Status(
        Status::Code::NOT_FOUND,
        "unable to load custom library: " + std::string(dlerror()));
  }
#endif
  return Status::Success;
}

void
CloseLibraryHandle(void* handle)
{
  if (handle != nullptr) {
#ifdef _WIN32
    if (FreeLibrary((HMODULE)handle) == 0) {
      LPSTR err_buffer = nullptr;
      size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                  NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&err_buffer, 0, NULL);
      std::string errstr(err_buffer, size);
      LocalFree(err_buffer);
      LOG_ERROR << "unable to unload custom library: " << errstr;
    }
#else
    if (dlclose(handle) != 0) {
      LOG_ERROR << "unable to unload custom library: " << dlerror();
    }
#endif
  }
}

Status
GetEntrypoint(void* handle, const std::string& name, void** fn)
{
#ifdef _WIN32
  *fn = GetProcAddress((HMODULE)handle, name.c_str());
  if (*fn == nullptr) {
    LPSTR err_buffer = nullptr;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&err_buffer, 0, NULL);
    std::string errstr(err_buffer, size);
    LocalFree(err_buffer);
    return Status(
        Status::Code::NOT_FOUND,
        "unable to find '" + name +
            "' entrypoint in custom library: " + errstr);
  }
#else
  dlerror();
  *fn = dlsym(handle, name.c_str());
  const char* dlsym_error = dlerror();
  if (dlsym_error != nullptr) {
    std::string errstr(dlsym_error);  // need copy as dlclose overwrites
    return Status(
        Status::Code::NOT_FOUND,
        "unable to find '" + name +
            "' entrypoint in custom library: " + errstr);
  }

  if (*fn == nullptr) {
    return Status(
        Status::Code::NOT_FOUND,
        "unable to find '" + name + "' entrypoint in custom library");
  }
#endif

  return Status::Success;
}

}  // namespace

Status
LoadCustom(
    const std::string& path, void** dlhandle,
    CustomInitializeFn_t* InitializeFn, CustomFinalizeFn_t* FinalizeFn,
    CustomErrorStringFn_t* ErrorStringFn, CustomExecuteFn_t* ExecuteFn,
    CustomExecuteV2Fn_t* ExecuteV2Fn, int* custom_version)
{
  *dlhandle = nullptr;
  *InitializeFn = nullptr;
  *FinalizeFn = nullptr;
  *ErrorStringFn = nullptr;
  *ExecuteFn = nullptr;
  *ExecuteV2Fn = nullptr;
  *custom_version = 0;

  void* handle = nullptr;
  // Load the custom library
  RETURN_IF_ERROR(OpenLibraryHandle(path, &handle));

  Status status;

  // Get shared library entrypoints.
  void* init_fn;
  status = GetEntrypoint(handle, "CustomInitialize", &init_fn);
  if (!status.IsOk()) {
    CloseLibraryHandle(handle);
    return status;
  }

  void* fini_fn;
  status = GetEntrypoint(handle, "CustomFinalize", &fini_fn);
  if (!status.IsOk()) {
    CloseLibraryHandle(handle);
    return status;
  }

  void* errstr_fn;
  status = GetEntrypoint(handle, "CustomErrorString", &errstr_fn);
  if (!status.IsOk()) {
    CloseLibraryHandle(handle);
    return status;
  }

  void* ver_fn;
  status = GetEntrypoint(handle, "CustomVersion", &ver_fn);
  if (!status.IsOk()) {
    *custom_version = 1;
  } else {
    *custom_version = ((CustomVersionFn_t)ver_fn)();
  }

  // Load version dependent symbols
  void* exec_fn;
  switch (*custom_version) {
    case 1:
      status = GetEntrypoint(handle, "CustomExecute", &exec_fn);
      break;
    case 2:
      status = GetEntrypoint(handle, "CustomExecuteV2", &exec_fn);
      break;
    default:
      status = Status(
          Status::Code::INVALID_ARG,
          "unable to load custom library: invalid custom version " +
              std::to_string(*custom_version) + " is provided");
      break;
  }
  if (!status.IsOk()) {
    CloseLibraryHandle(handle);
    return status;
  }

  *dlhandle = handle;
  *InitializeFn = (CustomInitializeFn_t)init_fn;
  *FinalizeFn = (CustomFinalizeFn_t)fini_fn;
  *ErrorStringFn = (CustomErrorStringFn_t)errstr_fn;

  if (*custom_version == 1) {
    *ExecuteFn = (CustomExecuteFn_t)exec_fn;
  } else {
    *ExecuteV2Fn = (CustomExecuteV2Fn_t)exec_fn;
  }

  return Status::Success;
}

void
UnloadCustom(void* handle)
{
  CloseLibraryHandle(handle);
}

}}  // namespace nvidia::inferenceserver
