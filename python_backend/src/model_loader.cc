// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "model_loader.h"

#ifdef TRITON_PB_STUB
#include "pb_stub.h"
#endif

namespace triton { namespace backend { namespace python {

void
ModelLoader::SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool)
{
  AllocatedSharedMemory<ModelLoaderRequestShm> model_loader_req_shm =
      shm_pool->Construct<ModelLoaderRequestShm>();
  model_loader_req_shm_ptr_ = model_loader_req_shm.data_.get();

  std::unique_ptr<PbString> name_shm = PbString::Create(shm_pool, name_);
  std::unique_ptr<PbString> version_shm = PbString::Create(shm_pool, version_);
  std::unique_ptr<PbString> config_shm = PbString::Create(shm_pool, config_);
  std::unique_ptr<PbMap> files_shm = PbMap::Create(shm_pool, files_);

  model_loader_req_shm_ptr_->name_shm_handle = name_shm->ShmHandle();
  model_loader_req_shm_ptr_->version_shm_handle = version_shm->ShmHandle();
  model_loader_req_shm_ptr_->config_shm_handle = config_shm->ShmHandle();
  model_loader_req_shm_ptr_->files_shm_handle = files_shm->ShmHandle();
  model_loader_req_shm_ptr_->unload_dependents = unload_dependents_;

  // Save the references to shared memory.
  model_loader_req_shm_ = std::move(model_loader_req_shm);
  name_shm_ = std::move(name_shm);
  version_shm_ = std::move(version_shm);
  config_shm_ = std::move(config_shm);
  files_shm_ = std::move(files_shm);

  shm_handle_ = model_loader_req_shm_.handle_;
}

std::unique_ptr<ModelLoader>
ModelLoader::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t handle)
{
  AllocatedSharedMemory<ModelLoaderRequestShm> model_loader_req_shm =
      shm_pool->Load<ModelLoaderRequestShm>(handle);
  ModelLoaderRequestShm* model_loader_req_shm_ptr =
      model_loader_req_shm.data_.get();

  std::unique_ptr<PbString> name_shm = PbString::LoadFromSharedMemory(
      shm_pool, model_loader_req_shm_ptr->name_shm_handle);
  std::unique_ptr<PbString> version_shm = PbString::LoadFromSharedMemory(
      shm_pool, model_loader_req_shm_ptr->version_shm_handle);
  std::unique_ptr<PbString> config_shm = PbString::LoadFromSharedMemory(
      shm_pool, model_loader_req_shm_ptr->config_shm_handle);
  std::unique_ptr<PbMap> files_shm = PbMap::LoadFromSharedMemory(
      shm_pool, model_loader_req_shm_ptr->files_shm_handle);

  return std::unique_ptr<ModelLoader>(new ModelLoader(
      model_loader_req_shm, name_shm, version_shm, config_shm, files_shm));
}

ModelLoader::ModelLoader(
    AllocatedSharedMemory<ModelLoaderRequestShm>& model_loader_req_shm,
    std::unique_ptr<PbString>& name_shm, std::unique_ptr<PbString>& version_shm,
    std::unique_ptr<PbString>& config_shm, std::unique_ptr<PbMap>& files_shm)
    : model_loader_req_shm_(std::move(model_loader_req_shm)),
      name_shm_(std::move(name_shm)), version_shm_(std::move(version_shm)),
      config_shm_(std::move(config_shm)), files_shm_(std::move(files_shm))
{
  model_loader_req_shm_ptr_ = model_loader_req_shm_.data_.get();
  name_ = name_shm_->String();
  version_ = version_shm_->String();
  config_ = config_shm_->String();
  files_ = files_shm_->UnorderedMap();
  unload_dependents_ = model_loader_req_shm_ptr_->unload_dependents;
}
#ifdef TRITON_PB_STUB
void
ModelLoader::SendLoadModelRequest()
{
  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  SaveToSharedMemory(stub->ShmPool());
  AllocatedSharedMemory<ModelLoaderMessage> model_loader_msg_shm;

  try {
    stub->SendMessage<ModelLoaderMessage>(
        model_loader_msg_shm, PYTHONSTUB_LoadModelRequest, shm_handle_);
  }
  catch (const PythonBackendException& pb_exception) {
    throw PythonBackendException(
        "Failed to load model: " + std::string(pb_exception.what()));
  }
}

void
ModelLoader::SendUnloadModelRequest()
{
  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  SaveToSharedMemory(stub->ShmPool());
  AllocatedSharedMemory<ModelLoaderMessage> model_loader_msg_shm;
  try {
    stub->SendMessage<ModelLoaderMessage>(
        model_loader_msg_shm, PYTHONSTUB_UnloadModelRequest, shm_handle_);
  }
  catch (const PythonBackendException& pb_exception) {
    throw PythonBackendException(
        "Failed to unload model: " + std::string(pb_exception.what()));
  }
}

bool
ModelLoader::SendModelReadinessRequest()
{
  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  SaveToSharedMemory(stub->ShmPool());
  ModelLoaderMessage* model_loader_msg = nullptr;
  AllocatedSharedMemory<ModelLoaderMessage> model_loader_msg_shm;
  try {
    stub->SendMessage<ModelLoaderMessage>(
        model_loader_msg_shm, PYTHONSTUB_ModelReadinessRequest, shm_handle_);
  }
  catch (const PythonBackendException& pb_exception) {
    throw PythonBackendException(
        "Failed to check model readiness: " + std::string(pb_exception.what()));
  }

  model_loader_msg = model_loader_msg_shm.data_.get();
  return model_loader_msg->is_model_ready;
}

void
LoadModel(
    const std::string& name, const std::string& config, const py::object& files)
{
  std::unordered_map<std::string, std::string> files_map;

  if (!files.is_none()) {
    if (!py::isinstance<py::dict>(files)) {
      throw PythonBackendException(
          "failed to load model '" + name +
          "', files should be a dictionary of file paths and file contents");
    }

    py::dict files_dict = py::cast<py::dict>(files);
    for (const auto& item : files_dict) {
      std::string key = py::cast<std::string>(item.first);
      py::bytes value = py::cast<py::bytes>(item.second);
      std::string content(value);
      files_map[key] = content;
    }
  }

  ModelLoader model_loader(name, config, files_map);
  model_loader.SendLoadModelRequest();
}

void
UnloadModel(const std::string& name, const bool unload_dependents)
{
  ModelLoader model_loader(name, unload_dependents);
  model_loader.SendUnloadModelRequest();
}

bool
IsModelReady(const std::string& name, const std::string& version)
{
  ModelLoader model_loader(name, version);
  return model_loader.SendModelReadinessRequest();
}
#else
void
ModelLoader::LoadModel(TRITONSERVER_Server* server)
{
  std::string path = "";
  std::string file_content = "";
  std::vector<const TRITONSERVER_Parameter*> const_params;
  if (!config_.empty()) {
    const_params.emplace_back(TRITONSERVER_ParameterNew(
        "config", TRITONSERVER_PARAMETER_STRING, config_.c_str()));
  }
  if (!files_.empty()) {
    for (auto& file : files_) {
      path = file.first;
      file_content = file.second;
      const_params.emplace_back(TRITONSERVER_ParameterBytesNew(
          path.c_str(), file_content.data(), file_content.size()));
    }
  }

  THROW_IF_TRITON_ERROR(TRITONSERVER_ServerLoadModelWithParameters(
      server, name_.c_str(), const_params.data(), const_params.size()));

  for (const auto param : const_params) {
    TRITONSERVER_ParameterDelete(const_cast<TRITONSERVER_Parameter*>(param));
  }
}

void
ModelLoader::UnloadModel(TRITONSERVER_Server* server)
{
  if (unload_dependents_) {
    THROW_IF_TRITON_ERROR(
        TRITONSERVER_ServerUnloadModelAndDependents(server, name_.c_str()));
  } else {
    THROW_IF_TRITON_ERROR(
        TRITONSERVER_ServerUnloadModel(server, name_.c_str()));
  }
}

bool
ModelLoader::IsModelReady(TRITONSERVER_Server* server)
{
  bool is_ready = false;
  int64_t model_version = GetModelVersionFromString(version_);
  THROW_IF_TRITON_ERROR(TRITONSERVER_ServerModelIsReady(
      server, name_.c_str(), model_version, &is_ready));
  return is_ready;
}

int64_t
ModelLoader::GetModelVersionFromString(const std::string& version_string)
{
  int64_t version = -1;
  if (!version_string.empty()) {
    try {
      version = std::stol(version_string);
    }
    catch (std::exception& e) {
      throw PythonBackendException(
          "failed to get model version from specified version string '" +
          version_string + "' (details: " + e.what() +
          "), version should be an integral value > 0");
    }

    if (version < 0) {
      throw PythonBackendException(
          "failed to get model version from specified version string '" +
          version_string + "', version should be an integral value > 0");
    }
  }
  return version;
}
#endif
}}}  // namespace triton::backend::python
