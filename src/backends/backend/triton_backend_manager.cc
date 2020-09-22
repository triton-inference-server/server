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
#include "src/backends/backend/triton_memory_manager.h"
#include "src/core/logging.h"
#include "src/core/server_message.h"

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
    const std::string& name, const std::string& dir, const std::string& libpath,
    const BackendCmdlineConfig& backend_cmdline_config,
    std::shared_ptr<TritonBackend>* backend)
{
  // Create the JSON representation of the backend configuration.
  triton::common::TritonJson::Value backend_config_json(
      triton::common::TritonJson::ValueType::OBJECT);
  if (!backend_cmdline_config.empty()) {
    triton::common::TritonJson::Value cmdline_json(
        backend_config_json, triton::common::TritonJson::ValueType::OBJECT);
    for (const auto& pr : backend_cmdline_config) {
      RETURN_IF_ERROR(cmdline_json.AddString(pr.first.c_str(), pr.second));
    }

    RETURN_IF_ERROR(
        backend_config_json.Add("cmdline", std::move(cmdline_json)));
  }

  TritonServerMessage backend_config(backend_config_json);

  auto local_backend = std::shared_ptr<TritonBackend>(
      new TritonBackend(name, dir, libpath, backend_config));

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

TritonBackend::TritonBackend(
    const std::string& name, const std::string& dir, const std::string& libpath,
    const TritonServerMessage& backend_config)
    : name_(name), dir_(dir), libpath_(libpath),
      backend_config_(backend_config),
      exec_policy_(TRITONBACKEND_EXECUTION_BLOCKING), state_(nullptr)
{
  ClearHandles();
}

TritonBackend::~TritonBackend()
{
  LOG_VERBOSE(1) << "unloading backend '" << name_ << "'";

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
  inst_init_fn_ = nullptr;
  inst_fini_fn_ = nullptr;
  inst_exec_fn_ = nullptr;
}

Status
TritonBackend::LoadBackendLibrary()
{
  void* handle = dlopen(libpath_.c_str(), RTLD_LAZY);
  if (handle == nullptr) {
    return Status(
        Status::Code::NOT_FOUND,
        "unable to load backend library: " + std::string(dlerror()));
  }

  TritonBackendInitFn_t bifn;
  TritonBackendFiniFn_t bffn;
  TritonModelInitFn_t mifn;
  TritonModelFiniFn_t mffn;
  TritonModelInstanceInitFn_t iifn;
  TritonModelInstanceFiniFn_t iffn;
  TritonModelInstanceExecFn_t iefn;

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

  // Model instance initialize and finalize functions, optional
  RETURN_IF_ERROR(GetEntrypoint(
      handle, "TRITONBACKEND_ModelInstanceInitialize", true /* optional */,
      reinterpret_cast<void**>(&iifn)));
  RETURN_IF_ERROR(GetEntrypoint(
      handle, "TRITONBACKEND_ModelInstanceFinalize", true /* optional */,
      reinterpret_cast<void**>(&iffn)));

  // Model instance execute function, required
  RETURN_IF_ERROR(GetEntrypoint(
      handle, "TRITONBACKEND_ModelInstanceExecute", false /* optional */,
      reinterpret_cast<void**>(&iefn)));

  dlhandle_ = handle;
  backend_init_fn_ = bifn;
  backend_fini_fn_ = bffn;
  model_init_fn_ = mifn;
  model_fini_fn_ = mffn;
  inst_init_fn_ = iifn;
  inst_fini_fn_ = iffn;
  inst_exec_fn_ = iefn;

  return Status::Success;
}

Status
TritonBackend::UnloadBackendLibrary()
{
  bool enable_unload = true;

  // For memory leak debugging do not unload the shared library so
  // that it is available for stack trace generation when the server
  // exits.
  const char* dstr = getenv("TRITONSERVER_DISABLE_BACKEND_UNLOAD");
  if (dstr != nullptr) {
    enable_unload = (atoi(dstr) == 0);
    LOG_VERBOSE(1) << "Disable shared-library unload for backend '" << Name()
                   << "'";
  }

  if (enable_unload) {
    if ((dlhandle_ != nullptr) && (dlclose(dlhandle_) != 0)) {
      return Status(
          Status::Code::INTERNAL,
          "unable to unload backend library: " + std::string(dlerror()));
    }
  }

  ClearHandles();

  return Status::Success;
}

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_ApiVersion(uint32_t* major, uint32_t* minor)
{
  *major = TRITONBACKEND_API_VERSION_MAJOR;
  *minor = TRITONBACKEND_API_VERSION_MINOR;
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_BackendName(TRITONBACKEND_Backend* backend, const char** name)
{
  TritonBackend* tb = reinterpret_cast<TritonBackend*>(backend);
  *name = tb->Name().c_str();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_BackendConfig(
    TRITONBACKEND_Backend* backend, TRITONSERVER_Message** backend_config)
{
  TritonBackend* tb = reinterpret_cast<TritonBackend*>(backend);
  *backend_config = const_cast<TRITONSERVER_Message*>(
      reinterpret_cast<const TRITONSERVER_Message*>(&tb->BackendConfig()));
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_BackendExecutionPolicy(
    TRITONBACKEND_Backend* backend, TRITONBACKEND_ExecutionPolicy* policy)
{
  TritonBackend* tb = reinterpret_cast<TritonBackend*>(backend);
  *policy = tb->ExecutionPolicy();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_BackendSetExecutionPolicy(
    TRITONBACKEND_Backend* backend, TRITONBACKEND_ExecutionPolicy policy)
{
  TritonBackend* tb = reinterpret_cast<TritonBackend*>(backend);
  tb->SetExecutionPolicy(policy);
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_BackendArtifacts(
    TRITONBACKEND_Backend* backend, TRITONBACKEND_ArtifactType* artifact_type,
    const char** location)
{
  TritonBackend* tb = reinterpret_cast<TritonBackend*>(backend);
  *artifact_type = TRITONBACKEND_ARTIFACT_FILESYSTEM;
  *location = tb->Directory().c_str();
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_BackendMemoryManager(
    TRITONBACKEND_Backend* backend, TRITONBACKEND_MemoryManager** manager)
{
  static TritonMemoryManager gMemoryManager;
  *manager = reinterpret_cast<TRITONBACKEND_MemoryManager*>(&gMemoryManager);
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

TritonBackendManager&
TritonBackendManager::Instance()
{
  static TritonBackendManager triton_backend_manager{};
  return triton_backend_manager;
}

const std::unordered_map<std::string, std::weak_ptr<TritonBackend>>&
TritonBackendManager::BackendMap()
{
  return backend_map_;
}

Status
TritonBackendManager::CreateBackend(
    const std::string& name, const std::string& dir, const std::string& libpath,
    const BackendCmdlineConfig& backend_cmdline_config,
    std::shared_ptr<TritonBackend>* backend)
{
  TritonBackendManager& singleton_manager = TritonBackendManager::Instance();

  std::lock_guard<std::mutex> lock(singleton_manager.Mutex());

  const auto& itr = singleton_manager.backend_map_.find(libpath);
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

  RETURN_IF_ERROR(TritonBackend::Create(
      name, dir, libpath, backend_cmdline_config, backend));
  singleton_manager.backend_map_.insert({libpath, *backend});

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
