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

#include <dlfcn.h>
#include "src/core/logging.h"

namespace nvidia { namespace inferenceserver {

namespace {

Status
GetEntrypoint(void* handle, const std::string& name, void** fn)
{
  dlerror();
  *fn = dlsym(handle, name.c_str());
  const char* dlsym_error = dlerror();
  if (dlsym_error != nullptr) {
    std::string errstr(dlsym_error);  // need copy as dlclose overwrites
    return Status(
        RequestStatusCode::NOT_FOUND,
        "unable to find '" + name +
            "' entrypoint in custom library: " + errstr);
  }

  if (*fn == nullptr) {
    return Status(
        RequestStatusCode::NOT_FOUND,
        "unable to find '" + name + "' entrypoint in custom library");
  }

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

  // Load the custom library
  void* handle = dlopen(path.c_str(), RTLD_LAZY);
  if (handle == nullptr) {
    return Status(
        RequestStatusCode::NOT_FOUND,
        "unable to load custom library: " + std::string(dlerror()));
  }

  Status status;

  // Get shared library entrypoints.
  void* init_fn;
  status = GetEntrypoint(handle, "CustomInitialize", &init_fn);
  if (!status.IsOk()) {
    dlclose(handle);
    return status;
  }

  void* fini_fn;
  status = GetEntrypoint(handle, "CustomFinalize", &fini_fn);
  if (!status.IsOk()) {
    dlclose(handle);
    return status;
  }

  void* errstr_fn;
  status = GetEntrypoint(handle, "CustomErrorString", &errstr_fn);
  if (!status.IsOk()) {
    dlclose(handle);
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
          RequestStatusCode::INVALID_ARG,
          "unable to load custom library: invalid custom version " +
              std::to_string(*custom_version) + " is provided");
      break;
  }
  if (!status.IsOk()) {
    dlclose(handle);
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
  if (handle != nullptr) {
    if (dlclose(handle) != 0) {
      LOG_ERROR << "unable to unload custom library: " << dlerror();
    }
  }
}

}}  // namespace nvidia::inferenceserver
