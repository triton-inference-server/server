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

#pragma once

#include <string>
#include <unordered_map>

#include "ipc_message.h"
#include "pb_map.h"
#include "pb_string.h"
#include "pb_utils.h"

#ifdef TRITON_PB_STUB
#include <pybind11/embed.h>
namespace py = pybind11;
#else
#include "triton/core/tritonserver.h"
#endif

namespace triton { namespace backend { namespace python {

// The 'ModelLoaderRequestShm' struct is utilized by the 'ModelLoader' class for
// saving the essential data to shared memory and for loading the data from
// shared memory in order to reconstruct the 'ModelLoader' object.
struct ModelLoaderRequestShm {
  // The shared memory handle of the model name in PbString format.
  bi::managed_external_buffer::handle_t name_shm_handle;
  // The shared memory handle of the model version in PbString format.
  bi::managed_external_buffer::handle_t version_shm_handle;
  // The flag to unload the dependent models.
  bool unload_dependents;
  // The shared memory handle of the config in PbString format.
  bi::managed_external_buffer::handle_t config_shm_handle;
  // The shared memory handle of the files in PbMap format.
  bi::managed_external_buffer::handle_t files_shm_handle;
};

class ModelLoader {
 public:
  ModelLoader(
      const std::string& name, const std::string& config,
      const std::unordered_map<std::string, std::string>& files)
      : name_(name), version_(""), config_(config), files_(files),
        unload_dependents_(false)
  {
  }

  ModelLoader(const std::string& name, const bool unload_dependents)
      : name_(name), version_(""), config_(""), files_({}),
        unload_dependents_(unload_dependents)
  {
  }

  ModelLoader(const std::string& name, const std::string& version)
      : name_(name), version_(version), config_(""), files_({}),
        unload_dependents_(false)
  {
  }

  /// Save ModelLoader object to shared memory.
  /// \param shm_pool Shared memory pool to save the ModelLoader object.
  void SaveToSharedMemory(std::unique_ptr<SharedMemoryManager>& shm_pool);

  /// Create a ModelLoader object from shared memory.
  /// \param shm_pool Shared memory pool
  /// \param handle Shared memory handle of the ModelLoader.
  /// \return Returns the ModelLoaders in the specified request_handle
  /// location.
  static std::unique_ptr<ModelLoader> LoadFromSharedMemory(
      std::unique_ptr<SharedMemoryManager>& shm_pool,
      bi::managed_external_buffer::handle_t handle);
#ifdef TRITON_PB_STUB
  /// Send a request to load the model.
  void SendLoadModelRequest();

  /// Send a request to unload the model.
  void SendUnloadModelRequest();

  /// Send a request to check if the model is ready.
  bool SendModelReadinessRequest();
#else
  /// Use Triton C API to load the model.
  /// \param server The Triton server object.
  void LoadModel(TRITONSERVER_Server* server);

  /// Use Triton C API to unload the model.
  /// \param server The Triton server object.
  void UnloadModel(TRITONSERVER_Server* server);

  /// Use Triton C API to check if the model is ready.
  /// \param server The Triton server object.
  /// \return Returns true if the model is ready.
  bool IsModelReady(TRITONSERVER_Server* server);

  /// Get the model version from the version string.
  /// \param version_string The version string.
  /// \return Returns the model version in uint64_t.
  int64_t GetModelVersionFromString(const std::string& version_string);
#endif
  /// Disallow copying the ModelLoader object.
  DISALLOW_COPY_AND_ASSIGN(ModelLoader);

 private:
  // The private constructor for creating a Metric object from shared memory.
  ModelLoader(
      AllocatedSharedMemory<ModelLoaderRequestShm>& model_loader_req_shm,
      std::unique_ptr<PbString>& name_shm,
      std::unique_ptr<PbString>& version_shm,
      std::unique_ptr<PbString>& config_shm, std::unique_ptr<PbMap>& files_shm);

  // The name of the model.
  std::string name_;
  // The version of the model.
  std::string version_;
  // The configuration of the model.
  std::string config_;
  // The files of the model.
  std::unordered_map<std::string, std::string> files_;
  // The flag to unload the dependent models.
  bool unload_dependents_;

  // // Shared Memory Data Structures
  AllocatedSharedMemory<ModelLoaderRequestShm> model_loader_req_shm_;
  ModelLoaderRequestShm* model_loader_req_shm_ptr_;
  bi::managed_external_buffer::handle_t shm_handle_;
  std::unique_ptr<PbString> name_shm_;
  std::unique_ptr<PbString> version_shm_;
  std::unique_ptr<PbString> config_shm_;
  std::unique_ptr<PbMap> files_shm_;
};

#ifdef TRITON_PB_STUB
// The binding functions for the Python stub.
void LoadModel(
    const std::string& name, const std::string& config,
    const py::object& files = py::none());
void UnloadModel(const std::string& name, const bool unload_dependents);
bool IsModelReady(const std::string& name, const std::string& version);
#endif

}}};  // namespace triton::backend::python
