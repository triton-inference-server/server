// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "src/backends/backend/triton_backend_manager.h"

#include <dlfcn.h>
#include "src/core/logging.h"

namespace nvidia { namespace inferenceserver {

namespace {

Status
GetEntrypoint(
    void* handle, const std::string& name, const bool optional, void** befn)
{
  *befn = nullptr;

  dlerror();
  void* fn = dlsym(handle, name.c_str());
  const char* dlsym_error = dlerror();
  if (dlsym_error != nullptr) {
    if (optional) {
      return Status::Success;
    }

    std::string errstr(dlsym_error);  // need copy as dlclose overwrites
    dlclose(handle);
    return Status(
        Status::Code::NOT_FOUND, "unable to find required entrypoint '" + name +
                                     "' in backend library: " + errstr);
  }

  if (fn == nullptr) {
    if (optional) {
      return Status::Success;
    }

    dlclose(handle);
    return Status(
        Status::Code::NOT_FOUND,
        "unable to find required entrypoint '" + name + "' in backend library");
  }

  *befn = fn;
  return Status::Success;
}

}  // namespace

//
// TritonBackend
//
Status
TritonBackend::Create(
    const std::string& name, const std::string& path,
    std::shared_ptr<TritonBackend>* backend)
{
  auto local_backend =
      std::shared_ptr<TritonBackend>(new TritonBackend(name, path));

  // Load the library and initialize all the entrypoints
  RETURN_IF_ERROR(local_backend->LoadBackendLibrary());

  // Backend initialization is optional... The TRITONBACKEND_Backend
  // object is this TritonBackend object.
  if (local_backend->backend_init_fn_ != nullptr) {
    RETURN_IF_TRITONSERVER_ERROR(local_backend->backend_init_fn_(
        reinterpret_cast<TRITONBACKEND_Backend*>(local_backend.get())));
  }

  *backend = std::move(local_backend);
  return Status::Success;
}

TritonBackend::TritonBackend(const std::string& name, const std::string& path)
    : name_(name), path_(path), state_(nullptr)
{
  ClearHandles();
}

TritonBackend::~TritonBackend()
{
  // Backend finalization is optional... The TRITONBACKEND_Backend
  // object is this TritonBackend object.
  if (backend_fini_fn_ != nullptr) {
    LOG_TRITONSERVER_ERROR(
        backend_fini_fn_(reinterpret_cast<TRITONBACKEND_Backend*>(this)),
        "failed finalizing backend");
  }

  LOG_STATUS_ERROR(UnloadBackendLibrary(), "failed unloading backend");
}

void
TritonBackend::ClearHandles()
{
  dlhandle_ = nullptr;
  backend_init_fn_ = nullptr;
  backend_fini_fn_ = nullptr;
  model_init_fn_ = nullptr;
  model_fini_fn_ = nullptr;
  model_exec_fn_ = nullptr;
}

Status
TritonBackend::LoadBackendLibrary()
{
  void* handle = dlopen(path_.c_str(), RTLD_LAZY);
  if (handle == nullptr) {
    return Status(
        Status::Code::NOT_FOUND,
        "unable to load backend library: " + std::string(dlerror()));
  }

  TritonBackendInitFn_t bifn;
  TritonBackendFiniFn_t bffn;
  TritonModelInitFn_t mifn;
  TritonModelFiniFn_t mffn;
  TritonModelExecFn_t mefn;

  // Backend initialize and finalize functions, optional
  RETURN_IF_ERROR(GetEntrypoint(
      handle, "TRITONBACKEND_Initialize", true /* optional */,
      reinterpret_cast<void**>(&bifn)));
  RETURN_IF_ERROR(GetEntrypoint(
      handle, "TRITONBACKEND_Finalize", true /* optional */,
      reinterpret_cast<void**>(&bffn)));

  // Model initialize and finalize functions, optional
  RETURN_IF_ERROR(GetEntrypoint(
      handle, "TRITONBACKEND_ModelInitialize", true /* optional */,
      reinterpret_cast<void**>(&mifn)));
  RETURN_IF_ERROR(GetEntrypoint(
      handle, "TRITONBACKEND_ModelFinalize", true /* optional */,
      reinterpret_cast<void**>(&mffn)));

  // Model execute function, required
  RETURN_IF_ERROR(GetEntrypoint(
      handle, "TRITONBACKEND_ModelExecute", false /* optional */,
      reinterpret_cast<void**>(&mefn)));

  dlhandle_ = handle;
  backend_init_fn_ = bifn;
  backend_fini_fn_ = bffn;
  model_init_fn_ = mifn;
  model_fini_fn_ = mffn;
  model_exec_fn_ = mefn;

  return Status::Success;
}

Status
TritonBackend::UnloadBackendLibrary()
{
  if ((dlhandle_ != nullptr) && (dlclose(dlhandle_) != 0)) {
    return Status(
        Status::Code::INTERNAL,
        "unable to unload backend library: " + std::string(dlerror()));
  }

  ClearHandles();

  return Status::Success;
}

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_BackendName(TRITONBACKEND_Backend* backend, const char** name)
{
  TritonBackend* tb = reinterpret_cast<TritonBackend*>(backend);
  *name = tb->Name().c_str();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_BackendApiVersion(
    TRITONBACKEND_Backend* backend, uint32_t* api_version)
{
  *api_version = TRITONBACKEND_API_VERSION;
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_BackendState(TRITONBACKEND_Backend* backend, void** state)
{
  TritonBackend* tb = reinterpret_cast<TritonBackend*>(backend);
  *state = tb->State();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_BackendSetState(TRITONBACKEND_Backend* backend, void* state)
{
  TritonBackend* tb = reinterpret_cast<TritonBackend*>(backend);
  tb->SetState(state);
  return nullptr;  // success
}

}  // extern C

//
// TritonBackendManager
//
Status
TritonBackendManager::CreateBackend(
    const std::string& name, const std::string& path,
    std::shared_ptr<TritonBackend>* backend)
{
  static TritonBackendManager singleton_manager;

  std::lock_guard<std::mutex> lock(singleton_manager.mu_);

  const auto& itr = singleton_manager.backend_map_.find(path);
  if (itr != singleton_manager.backend_map_.end()) {
    // Found in map. If the weak_ptr is still valid that means that
    // there are other models using the backend and we just reuse that
    // same backend. If the weak_ptr is not valid then backend has
    // been unloaded so we need to remove the weak_ptr from the map
    // and create the backend again.
    *backend = itr->second.lock();
    if (*backend != nullptr) {
      return Status::Success;
    }

    singleton_manager.backend_map_.erase(itr);
  }

  RETURN_IF_ERROR(TritonBackend::Create(name, path, backend));
  singleton_manager.backend_map_.insert({path, *backend});

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
