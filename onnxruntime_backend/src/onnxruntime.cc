// Copyright 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stdint.h>

#include <mutex>
#include <vector>

#include "onnxruntime_loader.h"
#include "onnxruntime_utils.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/backend/device_memory_tracker.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

//
// ONNX Runtime Backend that implements the TRITONBACKEND API.
//

namespace triton { namespace backend { namespace onnxruntime {

/// Deleter for OrtSession.
struct SessionDeleter {
  void operator()(OrtSession* f) { OnnxLoader::UnloadSession(f); }
};

// BackendConfiguration
struct BackendConfiguration {
  static const BackendConfiguration& RetrieveFrom(
      TRITONBACKEND_Backend* backend)
  {
    void* state = nullptr;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        TRITONBACKEND_BackendState(backend, &state));
    return *reinterpret_cast<BackendConfiguration*>(state);
  }

  static const BackendConfiguration& RetrieveFrom(TRITONBACKEND_Model* model)
  {
    TRITONBACKEND_Backend* backend = nullptr;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        TRITONBACKEND_ModelBackend(model, &backend));
    return RetrieveFrom(backend);
  }

  static const BackendConfiguration& RetrieveFrom(
      TRITONBACKEND_ModelInstance* instance)
  {
    TRITONBACKEND_Model* model = nullptr;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        TRITONBACKEND_ModelInstanceModel(instance, &model));
    return RetrieveFrom(model);
  }

  bool enable_memory_tracker_{false};
  int default_max_batch_size_{0};
};

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  // Load an ONNX model using 'artifact_name' as the name for the ONNX
  // file/directory. If 'instance_group_kind' is not
  // TRITONSERVER_INSTANCEGROUPKIND_AUTO then use it and
  // 'instance_group_device_id' to initialize the appropriate
  // execution providers. Return in 'model_path' the full path to the
  // onnx file, return in 'session' and 'allocator' the ORT session
  // and allocator.
  TRITONSERVER_Error* LoadModel(
      const std::string& artifact_name,
      const TRITONSERVER_InstanceGroupKind instance_group_kind,
      const int32_t instance_group_device_id, std::string* model_path,
      OrtSession** session, OrtAllocator** default_allocator,
      cudaStream_t stream);

  const std::map<std::string, std::pair<int64_t, int64_t>>& ModelOutputs()
  {
    return model_outputs_;
  }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  TRITONSERVER_Error* AutoCompleteConfig();
  TRITONSERVER_Error* AutoCompleteMaxBatch(
      const OnnxTensorInfoMap& input_tensor_infos,
      const OnnxTensorInfoMap& output_tensor_infos);
  TRITONSERVER_Error* AutoCompleteIO(
      const char* key, const OnnxTensorInfoMap& io_infos);

  // Session options used when creating a ORT session.
  std::unique_ptr<OrtSessionOptions, SessionOptionsDeleter> session_options_;

  // model_outputs is a map that contains unique outputs that the model must
  // provide. In the model configuration, the output in the state configuration
  // can have intersection with the outputs section of the model. If an output
  // is specified both in the output section and state section, it indicates
  // that the backend must return the output state to the client too.
  std::map<std::string, std::pair<int64_t, int64_t>> model_outputs_;
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  // Auto-complete the configuration if requested...
  bool auto_complete_config = false;
  RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(
      triton_model, &auto_complete_config));
  if (auto_complete_config) {
    RETURN_IF_ERROR((*state)->AutoCompleteConfig());
    RETURN_IF_ERROR((*state)->SetModelConfig());
  }

  auto& model_outputs = (*state)->model_outputs_;

  // Parse the output states in the model configuration
  triton::common::TritonJson::Value sequence_batching;
  if ((*state)->ModelConfig().Find("sequence_batching", &sequence_batching)) {
    triton::common::TritonJson::Value states;
    if (sequence_batching.Find("state", &states)) {
      for (size_t i = 0; i < states.ArraySize(); i++) {
        triton::common::TritonJson::Value state;
        RETURN_IF_ERROR(states.IndexAsObject(i, &state));
        std::string output_state_name;
        RETURN_IF_ERROR(
            state.MemberAsString("output_name", &output_state_name));
        auto it = model_outputs.find(output_state_name);
        if (it == model_outputs.end()) {
          model_outputs.insert({output_state_name, std::make_pair(-1, i)});
        } else {
          it->second.second = i;
        }
      }
    }
  }

  // Parse the output names in the model configuration
  triton::common::TritonJson::Value outputs;
  RETURN_IF_ERROR((*state)->ModelConfig().MemberAsArray("output", &outputs));
  for (size_t i = 0; i < outputs.ArraySize(); i++) {
    triton::common::TritonJson::Value output;
    RETURN_IF_ERROR(outputs.IndexAsObject(i, &output));

    std::string output_name_str;

    RETURN_IF_ERROR(output.MemberAsString("name", &output_name_str));
    auto it = model_outputs.find(output_name_str);
    if (it == model_outputs.end()) {
      model_outputs.insert({output_name_str, {i, -1}});
    } else {
      it->second.first = i;
    }
  }


  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model, true /* allow_optional */)
{
  // Create session options that will be cloned and used for each
  // instance when creating that instance's session.
  OrtSessionOptions* soptions;
  THROW_IF_BACKEND_MODEL_ORT_ERROR(ort_api->CreateSessionOptions(&soptions));
  session_options_.reset(soptions);

  GraphOptimizationLevel optimization_level =
      GraphOptimizationLevel::ORT_ENABLE_ALL;
  {
    triton::common::TritonJson::Value optimization;
    if (ModelConfig().Find("optimization", &optimization)) {
      triton::common::TritonJson::Value graph;
      if (optimization.Find("graph", &graph)) {
        int64_t graph_level = 0;
        THROW_IF_BACKEND_MODEL_ERROR(graph.MemberAsInt("level", &graph_level));
        if (graph_level == -1) {
          optimization_level = GraphOptimizationLevel::ORT_ENABLE_BASIC;
        } else if (graph_level == 1) {
          optimization_level = GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
        } else if (graph_level == 2) {
          optimization_level = GraphOptimizationLevel::ORT_DISABLE_ALL;
        }
      }
    }
  }
  THROW_IF_BACKEND_MODEL_ORT_ERROR(
      ort_api->SetSessionGraphOptimizationLevel(soptions, optimization_level));

  {
    // Controls whether you want to execute operators in your graph sequentially
    // or in parallel. Usually when the model has many branches, setting this
    // option to ExecutionMode::ORT_PARALLEL will give you better performance.
    int execution_mode = 0;
    triton::common::TritonJson::Value params;
    if (ModelConfig().Find("parameters", &params)) {
      THROW_IF_BACKEND_MODEL_ERROR(TryParseModelStringParameter(
          params, "execution_mode", &execution_mode, 0));
    }

    // 0 and 1 are the only valid values.
    if (execution_mode != 0 && execution_mode != 1) {
      throw BackendModelException(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string(
               "Invalid configuration value provided. Expected values for "
               " execution_mode are 0 or 1 but got " +
               std::to_string(execution_mode) + " .")
               .c_str())));
    } else {
      THROW_IF_BACKEND_MODEL_ORT_ERROR(ort_api->SetSessionExecutionMode(
          soptions, execution_mode == 0 ? ExecutionMode::ORT_SEQUENTIAL
                                        : ExecutionMode::ORT_PARALLEL));
    }
  }

  // If global threadpool is enabled, disable per session threads.
  // If it is not enabled then try to read the configs for intra and inter
  // op num threads and set them for session.
  if (OnnxLoader::IsGlobalThreadPoolEnabled()) {
    THROW_IF_BACKEND_MODEL_ORT_ERROR(
        ort_api->DisablePerSessionThreads(soptions));
  } else {
    {
      // Sets the number of threads used to parallelize the execution within
      // nodes A value of 0 means ORT will pick a default
      int intra_op_thread_count = 0;
      triton::common::TritonJson::Value params;
      if (ModelConfig().Find("parameters", &params)) {
        THROW_IF_BACKEND_MODEL_ERROR(TryParseModelStringParameter(
            params, "intra_op_thread_count", &intra_op_thread_count, 0));
      }
      if (intra_op_thread_count > 0) {
        THROW_IF_BACKEND_MODEL_ORT_ERROR(
            ort_api->SetIntraOpNumThreads(soptions, intra_op_thread_count));
      }
    }

    {
      // Sets the number of threads used to parallelize the execution of the
      // graph (across nodes) If sequential execution is enabled this value is
      // ignored A value of 0 means ORT will pick a default
      int inter_op_thread_count = 0;
      triton::common::TritonJson::Value params;
      if (ModelConfig().Find("parameters", &params)) {
        THROW_IF_BACKEND_MODEL_ERROR(TryParseModelStringParameter(
            params, "inter_op_thread_count", &inter_op_thread_count, 0));
      }
      if (inter_op_thread_count > 0) {
        THROW_IF_BACKEND_MODEL_ORT_ERROR(
            ort_api->SetInterOpNumThreads(soptions, inter_op_thread_count));
      }
    }
  }

  // Enable/disable use_device_allocator_for_initializers
  {
    triton::common::TritonJson::Value params;
    if (ModelConfig().Find("parameters", &params)) {
      triton::common::TritonJson::Value json_value;
      const char* use_device_allocator_for_initializers_key =
          "session.use_device_allocator_for_initializers";
      if (params.Find(use_device_allocator_for_initializers_key, &json_value)) {
        std::string string_value;
        THROW_IF_BACKEND_MODEL_ERROR(
            json_value.MemberAsString("string_value", &string_value));

        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string("Configuring '") +
             use_device_allocator_for_initializers_key + "' to '" +
             string_value + "' for '" + Name() + "'")
                .c_str());
        THROW_IF_BACKEND_MODEL_ORT_ERROR(ort_api->AddSessionConfigEntry(
            soptions, use_device_allocator_for_initializers_key,
            string_value.c_str()));
      }
    }
  }

  // memory configs
  // enable/disable mem arena
  {
    triton::common::TritonJson::Value params;
    if (ModelConfig().Find("parameters", &params)) {
      triton::common::TritonJson::Value json_value;
      if (params.Find("enable_mem_arena", &json_value)) {
        std::string string_value;
        THROW_IF_BACKEND_MODEL_ERROR(
            json_value.MemberAsString("string_value", &string_value));
        bool enable_cpu_mem_arena = false;
        THROW_IF_BACKEND_MODEL_ERROR(
            ParseBoolValue(string_value, &enable_cpu_mem_arena));

        OrtStatus* ort_status = nullptr;
        if (enable_cpu_mem_arena) {
          ort_status = ort_api->EnableCpuMemArena(soptions);
        } else {
          ort_status = ort_api->DisableCpuMemArena(soptions);
        }
        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string("Configuring enable_mem_arena to ") + string_value)
                .c_str());
        THROW_IF_BACKEND_MODEL_ORT_ERROR(ort_status);
      }
    }
  }

  // enable/disable mem pattern
  {
    triton::common::TritonJson::Value params;
    if (ModelConfig().Find("parameters", &params)) {
      triton::common::TritonJson::Value json_value;
      if (params.Find("enable_mem_pattern", &json_value)) {
        std::string string_value;
        THROW_IF_BACKEND_MODEL_ERROR(
            json_value.MemberAsString("string_value", &string_value));
        bool enable_mem_pattern = false;
        auto err = ParseBoolValue(string_value, &enable_mem_pattern);
        THROW_IF_BACKEND_MODEL_ERROR(err);

        OrtStatus* ort_status = nullptr;
        if (enable_mem_pattern) {
          ort_status = ort_api->EnableMemPattern(soptions);
        } else {
          ort_status = ort_api->DisableMemPattern(soptions);
        }
        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string("Configuring enable_mem_pattern to ") + string_value)
                .c_str());
        THROW_IF_BACKEND_MODEL_ORT_ERROR(ort_status);
      }
    }
  }

  // FIXME. Is it possible to share a single OrtSession across
  // multiple instances? If so then should move loading and validation
  // of the session to here instead of creating a session for each
  // instance in ModelStateInstance::Create().
}

TRITONSERVER_Error*
ModelState::LoadModel(
    const std::string& artifact_name,
    const TRITONSERVER_InstanceGroupKind instance_group_kind,
    const int32_t instance_group_device_id, std::string* model_path,
    OrtSession** session, OrtAllocator** default_allocator, cudaStream_t stream)
{
  // Find the ONNX file that describes the model itself. If the model
  // configuration doesn't have an explicit model file specified then
  // use the default name ("model.onnx").
  std::string cc_model_filename = artifact_name;
  if (cc_model_filename.empty()) {
    cc_model_filename = "model.onnx";
  }

  *model_path = JoinPath(
      {RepositoryPath(), std::to_string(Version()), cc_model_filename});

  // If the model path is a directory then the actual model is
  // <dir>/model.onnx.
  {
    bool is_dir;
    RETURN_IF_ERROR(IsDirectory(*model_path, &is_dir));
    if (is_dir) {
      *model_path = JoinPath({*model_path, "model.onnx"});
    }
  }

  {
    bool exists;
    RETURN_IF_ERROR(FileExists(*model_path, &exists));
    RETURN_ERROR_IF_FALSE(
        exists, TRITONSERVER_ERROR_UNAVAILABLE,
        std::string("unable to find '") + *model_path +
            "' for model instance '" + Name() + "'");
  }

  // Make a clone for the session options for this instance...
  OrtSessionOptions* soptions;
  RETURN_IF_ORT_ERROR(
      ort_api->CloneSessionOptions(session_options_.get(), &soptions));
  std::unique_ptr<OrtSessionOptions, SessionOptionsDeleter> soptions_wrapper(
      soptions);

  bool need_lock = false;

  // Add execution providers if they are requested.
  // Don't need to ensure uniqueness of the providers, ONNX Runtime
  // will check it.

  // GPU execution providers
#ifdef TRITON_ENABLE_GPU
  if ((instance_group_kind == TRITONSERVER_INSTANCEGROUPKIND_GPU) ||
      (instance_group_kind == TRITONSERVER_INSTANCEGROUPKIND_AUTO)) {
    std::map<std::string, std::string> cuda_options_map;
    triton::common::TritonJson::Value optimization;
    if (model_config_.Find("optimization", &optimization)) {
      triton::common::TritonJson::Value eas;
      if (optimization.Find("execution_accelerators", &eas)) {
        triton::common::TritonJson::Value gpu_eas;
        if (eas.Find("gpu_execution_accelerator", &gpu_eas)) {
          for (size_t ea_idx = 0; ea_idx < gpu_eas.ArraySize(); ea_idx++) {
            triton::common::TritonJson::Value ea;
            RETURN_IF_ERROR(gpu_eas.IndexAsObject(ea_idx, &ea));
            std::string name;
            RETURN_IF_ERROR(ea.MemberAsString("name", &name));
#ifdef TRITON_ENABLE_ONNXRUNTIME_TENSORRT
            if (name == kTensorRTExecutionAccelerator) {
              // create tensorrt options with default values
              OrtTensorRTProviderOptionsV2* trt_options;
              THROW_IF_BACKEND_MODEL_ORT_ERROR(
                  ort_api->CreateTensorRTProviderOptions(&trt_options));
              std::unique_ptr<
                  OrtTensorRTProviderOptionsV2,
                  decltype(ort_api->ReleaseTensorRTProviderOptions)>
                  rel_trt_options(
                      trt_options, ort_api->ReleaseTensorRTProviderOptions);
              std::string int8_calibration_table_name;
              std::string trt_engine_cache_path;
              // Validate and set parameters
              triton::common::TritonJson::Value params;
              if (ea.Find("parameters", &params)) {
                std::vector<std::string> param_keys, keys, values;
                RETURN_IF_ERROR(params.Members(&param_keys));
                for (const auto& param_key : param_keys) {
                  std::string value_string, key, value;
                  if (param_key == "precision_mode") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    if (value_string == "FP16") {
                      key = "trt_fp16_enable";
                      value = "1";
                    } else if (value_string == "INT8") {
                      key = "trt_int8_enable";
                      value = "1";
                    } else if (value_string != "FP32") {
                      RETURN_ERROR_IF_FALSE(
                          false, TRITONSERVER_ERROR_INVALID_ARG,
                          std::string("unsupported precision mode '") +
                              value_string + "' is requested");
                    }
                  } else if (param_key == "max_workspace_size_bytes") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    size_t max_workspace_size_bytes;
                    RETURN_IF_ERROR(ParseUnsignedLongLongValue(
                        value_string, &max_workspace_size_bytes));
                    key = "trt_max_workspace_size";
                    value = value_string;
                  } else if (param_key == "trt_max_partition_iterations") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    int trt_max_partition_iterations;
                    RETURN_IF_ERROR(ParseIntValue(
                        value_string, &trt_max_partition_iterations));
                    key = "trt_max_partition_iterations";
                    value = value_string;
                  } else if (param_key == "trt_min_subgraph_size") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    int trt_min_subgraph_size;
                    RETURN_IF_ERROR(
                        ParseIntValue(value_string, &trt_min_subgraph_size));
                    key = "trt_min_subgraph_size";
                    value = value_string;
                  } else if (param_key == "int8_calibration_table_name") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_int8_calibration_table_name";
                  } else if (param_key == "int8_use_native_calibration_table") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool use_native_calibration_table;
                    RETURN_IF_ERROR(ParseBoolValue(
                        value_string, &use_native_calibration_table));
                    key = "trt_int8_use_native_calibration_table";
                    value = value_string;
                  } else if (param_key == "trt_dla_enable") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_dla_enable;
                    RETURN_IF_ERROR(
                        ParseBoolValue(value_string, &trt_dla_enable));
                    key = "trt_dla_enable";
                    value = value_string;
                  } else if (param_key == "trt_dla_core") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    int trt_dla_core;
                    RETURN_IF_ERROR(ParseIntValue(value_string, &trt_dla_core));
                    key = "trt_dla_core";
                    value = value_string;
                  } else if (param_key == "trt_engine_cache_enable") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool enable_cache;
                    RETURN_IF_ERROR(
                        ParseBoolValue(value_string, &enable_cache));
                    key = "trt_engine_cache_enable";
                    value = value_string;
                  } else if (param_key == "trt_engine_cache_path") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_engine_cache_path";
                  } else if (param_key == "trt_engine_cache_prefix") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_engine_cache_prefix";
                  } else if (param_key == "trt_dump_subgraphs") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool dump_subgraphs;
                    RETURN_IF_ERROR(
                        ParseBoolValue(value_string, &dump_subgraphs));
                    key = "trt_dump_subgraphs";
                    value = value_string;
                  } else if (param_key == "trt_force_sequential_engine_build") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_force_sequential_engine_build;
                    RETURN_IF_ERROR(ParseBoolValue(
                        value_string, &trt_force_sequential_engine_build));
                    key = "trt_force_sequential_engine_build";
                    value = value_string;
                  } else if (param_key == "trt_context_memory_sharing_enable") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_context_memory_sharing_enable;
                    RETURN_IF_ERROR(ParseBoolValue(
                        value_string, &trt_context_memory_sharing_enable));
                    key = "trt_context_memory_sharing_enable";
                    value = value_string;
                  } else if (param_key == "trt_layer_norm_fp32_fallback") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_layer_norm_fp32_fallback;
                    RETURN_IF_ERROR(ParseBoolValue(
                        value_string, &trt_layer_norm_fp32_fallback));
                    key = "trt_layer_norm_fp32_fallback";
                    value = value_string;
                  } else if (param_key == "trt_timing_cache_enable") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_timing_cache_enable;
                    RETURN_IF_ERROR(
                        ParseBoolValue(value_string, &trt_timing_cache_enable));
                    key = "trt_timing_cache_enable";
                    value = value_string;
                  } else if (param_key == "trt_timing_cache_path") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_timing_cache_path";
                  } else if (param_key == "trt_force_timing_cache") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_force_timing_cache;
                    RETURN_IF_ERROR(
                        ParseBoolValue(value_string, &trt_force_timing_cache));
                    key = "trt_force_timing_cache";
                    value = value_string;
                  } else if (param_key == "trt_detailed_build_log") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_detailed_build_log;
                    RETURN_IF_ERROR(
                        ParseBoolValue(value_string, &trt_detailed_build_log));
                    key = "trt_detailed_build_log";
                    value = value_string;
                  } else if (param_key == "trt_build_heuristics_enable") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_build_heuristics_enable;
                    RETURN_IF_ERROR(ParseBoolValue(
                        value_string, &trt_build_heuristics_enable));
                    key = "trt_build_heuristics_enable";
                    value = value_string;
                  } else if (param_key == "trt_sparsity_enable") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_sparsity_enable;
                    RETURN_IF_ERROR(
                        ParseBoolValue(value_string, &trt_sparsity_enable));
                    key = "trt_sparsity_enable";
                    value = value_string;
                  } else if (param_key == "trt_builder_optimization_level") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    int trt_builder_optimization_level;
                    RETURN_IF_ERROR(ParseIntValue(
                        value_string, &trt_builder_optimization_level));
                    key = "trt_builder_optimization_level";
                    value = value_string;
                  } else if (param_key == "trt_auxiliary_streams") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    int trt_auxiliary_streams;
                    RETURN_IF_ERROR(
                        ParseIntValue(value_string, &trt_auxiliary_streams));
                    key = "trt_auxiliary_streams";
                    value = value_string;
                  } else if (param_key == "trt_tactic_sources") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_tactic_sources";
                  } else if (param_key == "trt_extra_plugin_lib_paths") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_extra_plugin_lib_paths";
                  } else if (param_key == "trt_profile_min_shapes") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_profile_min_shapes";
                  } else if (param_key == "trt_profile_max_shapes") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_profile_max_shapes";
                  } else if (param_key == "trt_profile_opt_shapes") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_profile_opt_shapes";
                  } else if (param_key == "trt_cuda_graph_enable") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_cuda_graph_enable;
                    RETURN_IF_ERROR(
                        ParseBoolValue(value_string, &trt_cuda_graph_enable));
                    key = "trt_cuda_graph_enable";
                    value = value_string;
                  } else if (param_key == "trt_dump_ep_context_model") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool trt_dump_ep_context_model;
                    RETURN_IF_ERROR(ParseBoolValue(
                        value_string, &trt_dump_ep_context_model));
                    key = "trt_dump_ep_context_model";
                    value = value_string;
                  } else if (param_key == "trt_ep_context_file_path") {
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                    key = "trt_ep_context_file_path";
                  } else if (param_key == "trt_ep_context_embed_mode") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    int trt_ep_context_embed_mode;
                    RETURN_IF_ERROR(ParseIntValue(
                        value_string, &trt_ep_context_embed_mode));
                    key = "trt_ep_context_embed_mode";
                    value = value_string;
                  } else {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INVALID_ARG,
                        std::string(
                            "unknown parameter '" + param_key +
                            "' is provided for TensorRT Execution "
                            "Accelerator")
                            .c_str());
                  }
                  if (!key.empty() && !value.empty()) {
                    keys.push_back(key);
                    values.push_back(value);
                  }
                }

                // assign correct GPU to EP
                keys.push_back(std::string("device_id"));
                values.push_back(std::to_string(instance_group_device_id));

                std::vector<const char*> c_keys, c_values;
                if (!keys.empty() && !values.empty()) {
                  for (size_t i = 0; i < keys.size(); ++i) {
                    c_keys.push_back(keys[i].c_str());
                    c_values.push_back(values[i].c_str());
                  }
                  RETURN_IF_ORT_ERROR(ort_api->UpdateTensorRTProviderOptions(
                      rel_trt_options.get(), c_keys.data(), c_values.data(),
                      keys.size()));
                }
              }

              RETURN_IF_ORT_ERROR(
                  ort_api->SessionOptionsAppendExecutionProvider_TensorRT_V2(
                      static_cast<OrtSessionOptions*>(soptions),
                      rel_trt_options.get()));
              LOG_MESSAGE(
                  TRITONSERVER_LOG_VERBOSE,
                  (std::string("TensorRT Execution Accelerator is set for '") +
                   Name() + "' on device " +
                   std::to_string(instance_group_device_id))
                      .c_str());
              continue;
            }
#endif  // TRITON_ENABLE_ONNXRUNTIME_TENSORRT

            if (name == "cuda") {
              // Parse CUDA EP configurations
              triton::common::TritonJson::Value params;
              if (ea.Find("parameters", &params)) {
                std::vector<std::string> param_keys;
                RETURN_IF_ERROR(params.Members(&param_keys));
                for (const auto& param_key : param_keys) {
                  std::string value_string, key, value;
                  // Special handling for boolean values
                  if (param_key == "do_copy_in_default_stream" ||
                      param_key == "use_ep_level_unified_stream") {
                    RETURN_IF_ERROR(params.MemberAsString(
                        param_key.c_str(), &value_string));
                    bool bool_value;
                    RETURN_IF_ERROR(ParseBoolValue(value_string, &bool_value));
                    key = param_key;
                    value = value_string;
                  } else {
                    key = param_key;
                    RETURN_IF_ERROR(
                        params.MemberAsString(param_key.c_str(), &value));
                  }
                  if (!key.empty() && !value.empty()) {
                    cuda_options_map[key] = value;
                  }
                }
              }
            } else {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("unknown Execution Accelerator '") + name +
                   "' is requested")
                      .c_str());
            }
          }
        }
      }
    }

    // Default GPU execution provider.
    // Using default values for everything other than device id and cuda
    // stream
    OrtCUDAProviderOptionsV2* cuda_options;
    RETURN_IF_ORT_ERROR(ort_api->CreateCUDAProviderOptions(&cuda_options));
    std::unique_ptr<
        OrtCUDAProviderOptionsV2, decltype(ort_api->ReleaseCUDAProviderOptions)>
        rel_cuda_options(cuda_options, ort_api->ReleaseCUDAProviderOptions);
    cuda_options_map["device_id"] = std::to_string(instance_group_device_id);
    cuda_options_map["has_user_compute_stream"] = stream != nullptr ? "1" : "0";
    RETURN_IF_ORT_ERROR(ort_api->UpdateCUDAProviderOptionsWithValue(
        rel_cuda_options.get(), "default_memory_arena_cfg", nullptr));
    {
      // Parse CUDA EP configurations directly from the parameters field.
      // This is deprecated with adding support for CUDA EP in the
      // gpu_execution_accelerator field. Keeping this for backward
      // compatibility.
      triton::common::TritonJson::Value params;
      if (model_config_.Find("parameters", &params)) {
        triton::common::TritonJson::Value json_value;
        if (params.Find("cudnn_conv_algo_search", &json_value)) {
          int cudnn_conv_algo_search = 0;
          RETURN_IF_ERROR(TryParseModelStringParameter(
              params, "cudnn_conv_algo_search", &cudnn_conv_algo_search, 0));
          std::string string_value;
          switch (cudnn_conv_algo_search) {
            case 0:
              string_value = "EXHAUSTIVE";
              break;
            case 1:
              string_value = "HEURISTIC";
              break;
            case 2:
              string_value = "DEFAULT";
              break;
            default:
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("unsupported cudnn_conv_algo_search value '") +
                   std::to_string(cudnn_conv_algo_search) + "' is requested")
                      .c_str());
          }
          cuda_options_map["cudnn_conv_algo_search"] = string_value;
        } else {
          cuda_options_map["cudnn_conv_algo_search"] = "EXHAUSTIVE";
        }

        if (params.Find("gpu_mem_limit", &json_value)) {
          std::string string_value;
          RETURN_IF_ERROR(
              json_value.MemberAsString("string_value", &string_value));
          cuda_options_map["gpu_mem_limit"] = string_value;
        } else {
          cuda_options_map["gpu_mem_limit"] =
              std::to_string(std::numeric_limits<size_t>::max());
        }

        if (params.Find("arena_extend_strategy", &json_value)) {
          int arena_extend_strategy = 0;
          RETURN_IF_ERROR(TryParseModelStringParameter(
              params, "arena_extend_strategy", &arena_extend_strategy, 0));
          std::string string_value;
          switch (arena_extend_strategy) {
            case 0:
              string_value = "kNextPowerOfTwo";
              break;
            case 1:
              string_value = "kSameAsRequested";
              break;
            default:
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string("unsupported arena_extend_strategy value '") +
                   std::to_string(arena_extend_strategy) + "' is requested")
                      .c_str());
          }
          cuda_options_map["arena_extend_strategy"] = string_value;
        } else {
          cuda_options_map["arena_extend_strategy"] = "kNextPowerOfTwo";
        }

        if (params.Find("do_copy_in_default_stream", &json_value)) {
          std::string string_value;
          RETURN_IF_ERROR(
              json_value.MemberAsString("string_value", &string_value));
          cuda_options_map["do_copy_in_default_stream"] = string_value;
        } else {
          cuda_options_map["do_copy_in_default_stream"] = "1";
        }
      }
    }

    std::vector<const char*> option_names, option_values;
    for (const auto& [key, value] : cuda_options_map) {
      option_names.push_back(key.c_str());
      option_values.push_back(value.c_str());
    }

    RETURN_IF_ORT_ERROR(ort_api->UpdateCUDAProviderOptions(
        rel_cuda_options.get(), option_names.data(), option_values.data(),
        option_values.size()));

    if (stream != nullptr) {
      RETURN_IF_ORT_ERROR(ort_api->UpdateCUDAProviderOptionsWithValue(
          rel_cuda_options.get(), "user_compute_stream", stream));
    }
    RETURN_IF_ORT_ERROR(ort_api->SessionOptionsAppendExecutionProvider_CUDA_V2(
        soptions, cuda_options));

    OrtAllocator* allocator;
    char* options;
    RETURN_IF_ORT_ERROR(ort_api->GetAllocatorWithDefaultOptions(&allocator));
    RETURN_IF_ORT_ERROR(ort_api->GetCUDAProviderOptionsAsString(
        rel_cuda_options.get(), allocator, &options));
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("CUDA Execution Accelerator is set for '") + Name() +
         "' on device " + std::to_string(instance_group_device_id) +
         std::string(" with options: ") + std::string(options))
            .c_str());
    RETURN_IF_ORT_ERROR(ort_api->AllocatorFree(allocator, options));
  }
#endif  // TRITON_ENABLE_GPU

  // CPU execution providers
  {
    triton::common::TritonJson::Value optimization;
    if (model_config_.Find("optimization", &optimization)) {
      triton::common::TritonJson::Value eas;
      if (optimization.Find("execution_accelerators", &eas)) {
        triton::common::TritonJson::Value cpu_eas;
        if (eas.Find("cpu_execution_accelerator", &cpu_eas)) {
          for (size_t ea_idx = 0; ea_idx < cpu_eas.ArraySize(); ea_idx++) {
            triton::common::TritonJson::Value ea;
            RETURN_IF_ERROR(cpu_eas.IndexAsObject(ea_idx, &ea));
            std::string name;
            RETURN_IF_ERROR(ea.MemberAsString("name", &name));
#ifdef TRITON_ENABLE_ONNXRUNTIME_OPENVINO
            if (name == kOpenVINOExecutionAccelerator) {
              need_lock = true;
              OrtOpenVINOProviderOptions openvino_options;
              openvino_options.device_type =
                  "CPU";  // device_type default is CPU

              RETURN_IF_ORT_ERROR(
                  ort_api->SessionOptionsAppendExecutionProvider_OpenVINO(
                      soptions, &openvino_options));

              LOG_MESSAGE(
                  TRITONSERVER_LOG_VERBOSE,
                  (std::string("OpenVINO Execution Accelerator is set for '") +
                   Name() + "' on CPU")
                      .c_str());
              continue;
            }
#endif  // TRITON_ENABLE_ONNXRUNTIME_OPENVINO
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("unknown Execution Accelerator '") + name +
                 "' is requested")
                    .c_str());
          }
        }
      }
    }
  }

  // Register all op libraries that contain custom operations.
  {
    triton::common::TritonJson::Value model_ops;
    if (model_config_.Find("model_operations", &model_ops)) {
      triton::common::TritonJson::Value op_library_filenames;
      if (model_ops.Find("op_library_filename", &op_library_filenames)) {
        for (size_t op_idx = 0; op_idx < op_library_filenames.ArraySize();
             op_idx++) {
          std::string op_filename;
          RETURN_IF_ERROR(
              op_library_filenames.IndexAsString(op_idx, &op_filename));
          void* library_handle = nullptr;
          RETURN_IF_ORT_ERROR(ort_api->RegisterCustomOpsLibrary(
              soptions, op_filename.c_str(), &library_handle));
        }
      }
    }
  }

  // ONNX session creation with OpenVINO is not thread-safe,
  // so multiple creations are serialized with a global lock.
  static std::mutex global_context_mu;
  std::unique_lock<std::mutex> glock(global_context_mu, std::defer_lock);
  if (need_lock) {
    glock.lock();
  }

  RETURN_IF_ERROR(OnnxLoader::LoadSession(
      true /* is_path */, *model_path, soptions, session));

  // get default cpu allocator
  RETURN_IF_ORT_ERROR(
      ort_api->GetAllocatorWithDefaultOptions(default_allocator));

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteConfig()
{
  // If the model configuration already specifies inputs and outputs
  // then don't perform any auto-completion.
  size_t input_cnt = 0;
  size_t output_cnt = 0;
  {
    triton::common::TritonJson::Value inputs;
    if (ModelConfig().Find("input", &inputs)) {
      input_cnt = inputs.ArraySize();
    }

    triton::common::TritonJson::Value config_batch_inputs;
    if (ModelConfig().Find("batch_input", &config_batch_inputs)) {
      input_cnt += config_batch_inputs.ArraySize();
    }

    triton::common::TritonJson::Value outputs;
    if (ModelConfig().Find("output", &outputs)) {
      output_cnt = outputs.ArraySize();
    }
  }

  if ((input_cnt > 0) && (output_cnt > 0)) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("skipping model configuration auto-complete for '") +
         Name() + "': inputs and outputs already specified")
            .c_str());
    return nullptr;  // success
  }

  std::string artifact_name;
  RETURN_IF_ERROR(
      ModelConfig().MemberAsString("default_model_filename", &artifact_name));

  // Must cleanup 'session'. 'allocator' is default allocator which
  // is managed by ONNX Runtime so don't need to free/release
  std::unique_ptr<OrtSession, SessionDeleter> session;
  OrtAllocator* default_allocator;
  std::string model_path;
  {
    TRITONSERVER_InstanceGroupKind kind = TRITONSERVER_INSTANCEGROUPKIND_CPU;

#ifdef TRITON_ENABLE_GPU
    triton::common::TritonJson::Value instance_group;
    ModelConfig().Find("instance_group", &instance_group);

    // Earlier in the model lifecycle, device checks for the instance group
    // have already occurred. If at least one instance group with
    // "kind" = "KIND_GPU" then allow model to use GPU else autocomplete to
    // "KIND_CPU"
    for (size_t i = 0; i < instance_group.ArraySize(); ++i) {
      triton::common::TritonJson::Value instance_obj;
      RETURN_IF_ERROR(instance_group.IndexAsObject(i, &instance_obj));

      triton::common::TritonJson::Value instance_group_kind;
      instance_obj.Find("kind", &instance_group_kind);
      std::string kind_str;
      RETURN_IF_ERROR(instance_group_kind.AsString(&kind_str));

      if (kind_str == "KIND_GPU") {
        kind = TRITONSERVER_INSTANCEGROUPKIND_GPU;
        break;
      }
    }
#endif  // TRITON_ENABLE_GPU

    OrtSession* sptr = nullptr;
    RETURN_IF_ERROR(LoadModel(
        artifact_name, kind, 0, &model_path, &sptr, &default_allocator,
        nullptr));
    session.reset(sptr);
  }
  OnnxTensorInfoMap input_tensor_infos;
  RETURN_IF_ERROR(
      InputInfos(session.get(), default_allocator, input_tensor_infos));
  OnnxTensorInfoMap output_tensor_infos;
  RETURN_IF_ERROR(
      OutputInfos(session.get(), default_allocator, output_tensor_infos));
  RETURN_IF_ERROR(
      AutoCompleteMaxBatch(input_tensor_infos, output_tensor_infos));
  if (input_cnt == 0) {
    RETURN_IF_ERROR(AutoCompleteIO("input", input_tensor_infos));
  }
  if (output_cnt == 0) {
    RETURN_IF_ERROR(AutoCompleteIO("output", output_tensor_infos));
  }

  if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
    triton::common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("post auto-complete:\n") + buffer.Contents()).c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteMaxBatch(
    const OnnxTensorInfoMap& input_tensor_infos,
    const OnnxTensorInfoMap& output_tensor_infos)
{
  // Determine if the model can potentially support batching. All
  // input and output tensors must have a variable first dimension.
  bool can_support_batching = true;
  for (const auto& io_info : input_tensor_infos) {
    const auto& dims = io_info.second.dims_;
    if ((dims.size() == 0) || (dims[0] != -1)) {
      can_support_batching = false;
    }
  }
  for (const auto& io_info : output_tensor_infos) {
    const auto& dims = io_info.second.dims_;
    if ((dims.size() == 0) || (dims[0] != -1)) {
      can_support_batching = false;
    }
  }

  // Set max-batch-size to 1 if we have determined that batching is
  // supported and max-batch-size is not specified. We need to update
  // the configuration itself as well as the cached value we have already
  // initialized in the model state.
  if (can_support_batching) {
    if (MaxBatchSize() == 0) {
      int default_max_batch_size = 0;
      {
        TRITONBACKEND_Backend* backend;
        THROW_IF_BACKEND_INSTANCE_ERROR(
            TRITONBACKEND_ModelBackend(TritonModel(), &backend));
        void* state;
        THROW_IF_BACKEND_INSTANCE_ERROR(
            TRITONBACKEND_BackendState(backend, &state));
        default_max_batch_size = reinterpret_cast<BackendConfiguration*>(state)
                                     ->default_max_batch_size_;
      }
      int max_batch_size = std::max(default_max_batch_size, 0);

      triton::common::TritonJson::Value mbs_value;
      ModelConfig().Find("max_batch_size", &mbs_value);
      mbs_value.SetInt(max_batch_size);
      SetMaxBatchSize(max_batch_size);

      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string(
               "autofilled max_batch_size to " +
               std::to_string(max_batch_size) + " for model '") +
           Name() +
           "' since batching is supporrted but no max_batch_size is "
           "specified "
           "in model configuration. Must specify max_batch_size to utilize "
           "autofill with a larger max batch size")
              .c_str());
    }

    // Check to see if we need to turn on dynamic batching
    // since model supports batching
    if (MaxBatchSize() > 1) {
      triton::common::TritonJson::Value value;
      bool found_sequence_batching =
          ModelConfig().Find("sequence_batching", &value);
      bool found_dynamic_batching =
          ModelConfig().Find("dynamic_batching", &value);
      if (!found_sequence_batching && !found_dynamic_batching) {
        triton::common::TritonJson::Value dynamic_batching(
            ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
        RETURN_IF_ERROR(
            ModelConfig().Add("dynamic_batching", std::move(dynamic_batching)));
      }
    }

  } else if (MaxBatchSize() != 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("autofill failed for model '") + Name() +
         "': model does not support batching while non-zero max_batch_size"
         " is specified")
            .c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteIO(const char* key, const OnnxTensorInfoMap& io_infos)
{
  triton::common::TritonJson::Value existing_ios;
  bool found_ios = ModelConfig().Find(key, &existing_ios);

  triton::common::TritonJson::Value ios(
      ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
  for (const auto& io_info : io_infos) {
    triton::common::TritonJson::Value io(
        ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
    RETURN_IF_ERROR(io.AddString("name", io_info.first));
    RETURN_IF_ERROR(io.AddString(
        "data_type", OnnxDataTypeToModelConfigDataType(io_info.second.type_)));

    // The model signature supports batching then the first dimension
    // is -1 and should not appear in the model configuration 'dims'
    // that we are creating.
    const auto& io_info_dims = io_info.second.dims_;
    triton::common::TritonJson::Value dims(
        ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
    for (size_t i = (MaxBatchSize() > 0) ? 1 : 0; i < io_info_dims.size();
         ++i) {
      RETURN_IF_ERROR(dims.AppendInt(io_info_dims[i]));
    }

    // If dims are empty then must use a reshape...
    if (dims.ArraySize() == 0) {
      RETURN_IF_ERROR(dims.AppendInt(1));
      triton::common::TritonJson::Value reshape(
          ModelConfig(), triton::common::TritonJson::ValueType::OBJECT);
      triton::common::TritonJson::Value reshape_dims(
          ModelConfig(), triton::common::TritonJson::ValueType::ARRAY);
      RETURN_IF_ERROR(reshape.Add("shape", std::move(reshape_dims)));
      // Empty reshape with `max_batch_size` indicates a scalar tensor in the
      // model configuration which is not a valid model configuration.
      if (MaxBatchSize() > 0) {
        RETURN_IF_ERROR(io.Add("reshape", std::move(reshape)));
      }
    }
    RETURN_IF_ERROR(io.Add("dims", std::move(dims)));
    RETURN_IF_ERROR(ios.Append(std::move(io)));
  }

  if (found_ios) {
    existing_ios.Swap(ios);
  } else {
    ModelConfig().Add(key, std::move(ios));
  }

  return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState();

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Execute...
  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);
  void ReleaseOrtRunResources();
  TRITONSERVER_Error* ValidateBooleanSequenceControl(
      triton::common::TritonJson::Value& sequence_batching,
      const std::string& control_kind, bool required, bool* have_control);
  TRITONSERVER_Error* ValidateTypedSequenceControl(
      triton::common::TritonJson::Value& sequence_batching,
      const std::string& control_kind, bool required, bool* have_control);
  TRITONSERVER_Error* ValidateInputs(const size_t expected_input_cnt);
  TRITONSERVER_Error* ValidateOutputs();
  TRITONSERVER_Error* OrtRun(
      std::vector<TRITONBACKEND_Response*>* responses,
      const uint32_t response_count);
  TRITONSERVER_Error* SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      BackendInputCollector* collector, std::vector<const char*>* input_names,
      bool* cuda_copy);
  TRITONSERVER_Error* SetStringInputTensor(
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses, const char* input_name,
      std::vector<const char*>* string_ptrs, bool* cuda_copy);
  void SetStringInputBuffer(
      const std::string& name, const std::vector<size_t>& expected_byte_sizes,
      const std::vector<size_t>& expected_element_cnts,
      std::vector<TRITONBACKEND_Response*>* responses, char* input_buffer,
      std::vector<const char*>* string_ptrs);
  void FillStringData(std::vector<const char*>* string_ptrs, size_t cnt);
  TRITONSERVER_Error* ReadOutputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  TRITONSERVER_Error* ReadOutputTensor(
      std::vector<int64_t>& batchn_shape, TRITONSERVER_DataType& dtype,
      OrtValue* output_tensor, void** output_buffer,
      std::vector<std::vector<char>>& string_buffers,
      std::vector<size_t>& offsets);
  bool SetStringOutputBuffer(
      const std::string& name, const char* content, const size_t* offsets,
      std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);
  bool SetStringStateBuffer(
      const std::string& name, const char* content, const size_t* offsets,
      std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);
  bool SetStringBuffer(
      const std::string& name, const char* content, const size_t* offsets,
      std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses, bool state);

  ModelState* model_state_;

  // The full path to the ONNX model file.
  std::string model_path_;

  // Onnx Runtime variables that are used across runs on this
  // instance.
  OrtSession* session_;
  OrtAllocator* default_allocator_;
  OrtMemoryInfo* cuda_allocator_info_;
  const OrtMemoryInfo* cpu_allocator_info_;
  OrtIoBinding* io_binding_;
  OrtRunOptions* runOptions_;
  // map of output name -> bound mem type and id
  std::unordered_map<std::string, std::pair<TRITONSERVER_MemoryType, int64_t>>
      output_device_info_;
  // map of output name -> tensor info
  OnnxTensorInfoMap output_tensor_infos_;

  // map of input name -> tensor info
  OnnxTensorInfoMap input_tensor_infos_;

  // A map from scalar output tensors to the dimension specified in model config
  std::unordered_map<std::string, std::vector<int64_t>> scalar_outputs_;

  // Onnx Runtime variables that will be reset and used for every run
  // on this instance.
  std::vector<OrtValue*> input_tensors_;
  std::vector<OrtValue*> output_tensors_;
  OrtValue** output_buffer_;
  std::vector<BackendMemory*> input_tensor_memories_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state), session_(nullptr), default_allocator_(nullptr),
      cuda_allocator_info_(nullptr), cpu_allocator_info_(nullptr),
      io_binding_(nullptr), output_buffer_(nullptr)
{
  THROW_IF_BACKEND_INSTANCE_ERROR(model_state->LoadModel(
      ArtifactFilename(), Kind(), DeviceId(), &model_path_, &session_,
      &default_allocator_, CudaStream()));

  if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    THROW_IF_BACKEND_INSTANCE_ORT_ERROR(ort_api->CreateMemoryInfo(
        "Cuda", OrtAllocatorType::OrtArenaAllocator, DeviceId(),
        OrtMemTypeDefault, &cuda_allocator_info_));
  }

  THROW_IF_BACKEND_INSTANCE_ORT_ERROR(
      ort_api->AllocatorGetInfo(default_allocator_, &cpu_allocator_info_));

  THROW_IF_BACKEND_INSTANCE_ORT_ERROR(
      ort_api->CreateIoBinding(session_, &io_binding_));

  THROW_IF_BACKEND_INSTANCE_ORT_ERROR(ort_api->CreateRunOptions(&runOptions_));

  // Read configs that needs to be set in RunOptions
  triton::common::TritonJson::Value params;
  if (model_state->ModelConfig().Find("parameters", &params)) {
    triton::common::TritonJson::Value json_value;
    const char* enable_memory_arena_shrinkage_key =
        "memory.enable_memory_arena_shrinkage";
    if (params.Find(enable_memory_arena_shrinkage_key, &json_value)) {
      std::string string_value;
      THROW_IF_BACKEND_MODEL_ERROR(
          json_value.MemberAsString("string_value", &string_value));

      LOG_MESSAGE(
          TRITONSERVER_LOG_VERBOSE,
          (std::string("Configuring ") + enable_memory_arena_shrinkage_key +
           " to " + string_value)
              .c_str());
      THROW_IF_BACKEND_MODEL_ORT_ERROR(ort_api->AddRunConfigEntry(
          runOptions_, enable_memory_arena_shrinkage_key,
          string_value.c_str()));
    }
  }

  size_t expected_input_cnt = 0;
  {
    triton::common::TritonJson::Value inputs;
    if (model_state->ModelConfig().Find("input", &inputs)) {
      expected_input_cnt = inputs.ArraySize();
      // Skip the optional inputs which are initializers
      for (size_t i = 0; i < inputs.ArraySize(); i++) {
        triton::common::TritonJson::Value input;
        THROW_IF_BACKEND_INSTANCE_ERROR(inputs.IndexAsObject(i, &input));
        bool is_optional;
        THROW_IF_BACKEND_INSTANCE_ERROR(
            input.MemberAsBool("optional", &is_optional));
        if (is_optional) {
          expected_input_cnt--;
        }
      }
    }

    triton::common::TritonJson::Value config_batch_inputs;
    if (model_state->ModelConfig().Find("batch_input", &config_batch_inputs)) {
      expected_input_cnt += config_batch_inputs.ArraySize();
    }
  }

  // If this is a sequence model then make sure that the required
  // inputs are present in the model and have the correct shape and
  // datatype.
  triton::common::TritonJson::Value sequence_batching;
  if (model_state->ModelConfig().Find(
          "sequence_batching", &sequence_batching)) {
    bool have_start, have_end, have_ready, have_corrid;
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateBooleanSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_START", false /* required */,
        &have_start));
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateBooleanSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_END", false /* required */,
        &have_end));
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateBooleanSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_READY", false /* required */,
        &have_ready));
    THROW_IF_BACKEND_INSTANCE_ERROR(ValidateTypedSequenceControl(
        sequence_batching, "CONTROL_SEQUENCE_CORRID", false /* required */,
        &have_corrid));
    if (have_start) {
      expected_input_cnt += 1;
    }
    if (have_end) {
      expected_input_cnt += 1;
    }
    if (have_ready) {
      expected_input_cnt += 1;
    }
    if (have_corrid) {
      expected_input_cnt += 1;
    }

    // Add the state inputs to the expected count
    triton::common::TritonJson::Value states;
    if (sequence_batching.Find("state", &states)) {
      expected_input_cnt += states.ArraySize();
    }
  }

  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateInputs(expected_input_cnt));
  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateOutputs());
}

ModelInstanceState::~ModelInstanceState()
{
  ReleaseOrtRunResources();
  ort_api->ReleaseRunOptions(runOptions_);
  ort_api->ReleaseIoBinding(io_binding_);
  ort_api->ReleaseMemoryInfo(cuda_allocator_info_);
  if (session_ != nullptr) {
    OnnxLoader::UnloadSession(session_);
  }
  // 'default_allocator_' is default allocator which is managed by ONNX
  // Runtime
}

void
ModelInstanceState::ReleaseOrtRunResources()
{
  ort_api->ClearBoundInputs(io_binding_);
  for (auto& tensor : input_tensors_) {
    if (tensor != nullptr) {
      ort_api->ReleaseValue(tensor);
    }
  }
  input_tensors_.clear();

  // first release the Ortvalues
  ort_api->ClearBoundOutputs(io_binding_);
  for (auto& tensor : output_tensors_) {
    if (tensor != nullptr) {
      ort_api->ReleaseValue(tensor);
    }
  }
  output_tensors_.clear();

  // next release the allocated buffer using the specified allocator
  if (output_buffer_) {
    auto free_status =
        ort_api->AllocatorFree(default_allocator_, output_buffer_);
    output_buffer_ = nullptr;
    if (free_status != nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("onnx runtime allocator free error:") +
           std::to_string(ort_api->GetErrorCode(free_status)) +
           ort_api->GetErrorMessage(free_status))
              .c_str());
      ort_api->ReleaseStatus(free_status);
    }
  }

  for (BackendMemory* mem : input_tensor_memories_) {
    delete mem;
  }
  input_tensor_memories_.clear();
}

TRITONSERVER_Error*
ModelInstanceState::ValidateBooleanSequenceControl(
    triton::common::TritonJson::Value& sequence_batching,
    const std::string& control_kind, bool required, bool* have_control)
{
  std::string tensor_name;
  std::string tensor_datatype;
  RETURN_IF_ERROR(GetBooleanSequenceControlProperties(
      sequence_batching, model_state_->Name(), control_kind, required,
      &tensor_name, &tensor_datatype, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr));
  *have_control = !tensor_name.empty();
  if (*have_control) {
    OnnxTensorInfoMap input_tensor_infos;
    RETURN_IF_ERROR(
        InputInfos(session_, default_allocator_, input_tensor_infos));
    const auto& iit = input_tensor_infos.find(tensor_name);
    if (iit == input_tensor_infos.end()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("configuration specified sequence control '") +
           tensor_name + "', but model does not provide that input")
              .c_str());
    }

    // Control tensors must have shape [1].
    const int nonbatch_start_idx = (model_state_->MaxBatchSize() > 0) ? 1 : 0;
    std::vector<int64_t> debatched_dims;
    for (size_t i = nonbatch_start_idx; i < iit->second.dims_.size(); i++) {
      debatched_dims.push_back(iit->second.dims_[i]);
    }

    if ((debatched_dims.size() != 1) || (debatched_dims[0] != 1)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', sequence control '" + tensor_name + "' in model has dims " +
           ShapeToString(debatched_dims) + " but dims [1] is expected")
              .c_str());
    }

    if (ModelConfigDataTypeToOnnxDataType(tensor_datatype) !=
        iit->second.type_) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', sequence control '" + tensor_name +
           "', the model expects data-type " +
           OnnxDataTypeName(iit->second.type_) +
           " but the model configuration specifies data-type " +
           tensor_datatype)
              .c_str());
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateTypedSequenceControl(
    triton::common::TritonJson::Value& sequence_batching,
    const std::string& control_kind, bool required, bool* have_control)
{
  std::string tensor_name;
  std::string tensor_datatype;
  RETURN_IF_ERROR(GetTypedSequenceControlProperties(
      sequence_batching, model_state_->Name(), control_kind, required,
      &tensor_name, &tensor_datatype));
  *have_control = !tensor_name.empty();
  if (*have_control) {
    OnnxTensorInfoMap input_tensor_infos;
    RETURN_IF_ERROR(
        InputInfos(session_, default_allocator_, input_tensor_infos));
    const auto& iit = input_tensor_infos.find(tensor_name);
    if (iit == input_tensor_infos.end()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("configuration specified sequence control '") +
           tensor_name + "', but model does not provide that input")
              .c_str());
    }

    // Control tensors must have shape [1].
    const int nonbatch_start_idx = (model_state_->MaxBatchSize() > 0) ? 1 : 0;
    std::vector<int64_t> debatched_dims;
    for (size_t i = nonbatch_start_idx; i < iit->second.dims_.size(); i++) {
      debatched_dims.push_back(iit->second.dims_[i]);
    }

    if ((debatched_dims.size() != 1) || (debatched_dims[0] != 1)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', sequence control '" + tensor_name + "' in model has dims " +
           ShapeToString(debatched_dims) + " but dims [1] is expected")
              .c_str());
    }

    if (ModelConfigDataTypeToOnnxDataType(tensor_datatype) !=
        iit->second.type_) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', sequence control '" + tensor_name +
           "', the model expects data-type " +
           OnnxDataTypeName(iit->second.type_) +
           " but the model configuration specifies data-type " +
           tensor_datatype)
              .c_str());
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateInputs(const size_t expected_input_cnt)
{
  std::set<std::string> input_tensor_names;
  RETURN_IF_ERROR(InputNames(session_, input_tensor_names));
  RETURN_IF_ERROR(
      InputInfos(session_, default_allocator_, input_tensor_infos_));

  std::set<std::string> overridable_initializer_tensor_names;
  RETURN_IF_ERROR(OverridableInitializerNames(
      session_, overridable_initializer_tensor_names));

  OnnxTensorInfoMap overridable_initializer_tensor_infos;
  RETURN_IF_ERROR(OverridableInitializerInfos(
      session_, default_allocator_, overridable_initializer_tensor_infos));

  if (input_tensor_infos_.size() != expected_input_cnt) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("unable to load model '") + model_state_->Name() +
         "', configuration expects " + std::to_string(expected_input_cnt) +
         " inputs, model provides " +
         std::to_string(input_tensor_infos_.size()))
            .c_str());
  }

  // Merge 'overridable_xxx' into 'input_xxx' as they can be request inputs,
  // and all request inputs are checked against 'input_xxx'
  for (const auto& name : overridable_initializer_tensor_names) {
    input_tensor_names.emplace(name);
  }

  for (const auto& info : overridable_initializer_tensor_infos) {
    input_tensor_infos_[info.first] = info.second;
  }

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("input", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
    bool io_optional;
    RETURN_IF_ERROR(io.MemberAsBool("optional", &io_optional));

    if (io_optional && model_state_->MaxBatchSize() != 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', optional input '" + io_name +
           "' is not supported for models that support batching")
              .c_str());
    }

    const auto& tensor_names = input_tensor_names;
    const auto& tensor_infos = input_tensor_infos_;
    auto iit = tensor_infos.find(io_name);
    if (iit == tensor_infos.end()) {
      RETURN_IF_ERROR(CheckAllowedModelInput(io, tensor_names));
    }

    auto onnx_data_type = ModelConfigDataTypeToOnnxDataType(io_dtype);
    if (onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unsupported datatype ") + io_dtype + " for input '" +
           io_name + "' for model '" + model_state_->Name() + "'")
              .c_str());
    } else if (onnx_data_type != iit->second.type_) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', configuration expects datatype " + io_dtype + " for input '" +
           io_name + "', model provides TYPE_" +
           TRITONSERVER_DataTypeString(
               ConvertFromOnnxDataType(iit->second.type_)))
              .c_str());
    }

    // If a reshape is provided for the input then use that when
    // validating that the model matches what is expected.
    std::vector<int64_t> dims;
    triton::common::TritonJson::Value reshape;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(ParseShape(reshape, "shape", &dims));
    } else {
      RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
    }

    triton::common::TritonJson::Value allow_ragged_batch_json;
    bool allow_ragged_batch = false;
    if (io.Find("allow_ragged_batch", &allow_ragged_batch_json)) {
      RETURN_IF_ERROR(allow_ragged_batch_json.AsBool(&allow_ragged_batch));
    }
    if (io_optional && allow_ragged_batch) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', configuration expects model provides input with shape [-1] "
           "for ragged input '" +
           io_name + "', which is not supported for optional input")
              .c_str());
    }
    if (allow_ragged_batch) {
      const std::vector<int64_t>& model_shape = iit->second.dims_;
      // Make sure the input has shape [-1]
      if ((model_shape.size() != 1) || (model_shape[0] != WILDCARD_DIM)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("unable to load model '") + model_state_->Name() +
             "', configuration expects model provides input with shape [-1]  "
             "for ragged input '" +
             io_name + "', model provides " + ShapeToString(model_shape))
                .c_str());
      }
    } else {
      // Only compare the dimensions if the tensor is not scalar
      if (iit->second.dims_.size() != 0) {
        RETURN_IF_ERROR(CompareDimsSupported(
            model_state_->Name(), io_name, iit->second.dims_, dims,
            model_state_->MaxBatchSize(), false /* compare_exact */));
      } else {
        // if max_batch_size == 0 and is a scalar tensor all the
        // dimensions specified must be equal to 1
        for (auto& dim : dims) {
          if (dim != 1) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("unable to load model '") + model_state_->Name() +
                 "', scalar tensor '" + io_name +
                 "', should only provide 1 in the model configuration when the "
                 "model doesn't support batching. Model configuration "
                 "provided: " +
                 ShapeToString(dims) + ".")
                    .c_str());
          }
        }
      }
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateOutputs()
{
  std::set<std::string> output_tensor_names;
  RETURN_IF_ERROR(OutputNames(session_, output_tensor_names));

  RETURN_IF_ERROR(
      OutputInfos(session_, default_allocator_, output_tensor_infos_));

  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("output", &ios));
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    std::string io_dtype;
    RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));

    auto iit = output_tensor_infos_.find(io_name);
    if (iit == output_tensor_infos_.end()) {
      RETURN_IF_ERROR(CheckAllowedModelOutput(io, output_tensor_names));
    }

    auto onnx_data_type = ModelConfigDataTypeToOnnxDataType(io_dtype);
    if (onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unsupported datatype ") + io_dtype + " for output '" +
           io_name + "' for model '" + model_state_->Name() + "'")
              .c_str());
    } else if (onnx_data_type != iit->second.type_) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to load model '") + model_state_->Name() +
           "', configuration expects datatype " + io_dtype + " for output '" +
           io_name + "', model provides TYPE_" +
           TRITONSERVER_DataTypeString(
               ConvertFromOnnxDataType(iit->second.type_)))
              .c_str());
    }

    // If a reshape is provided for the input then use that when
    // validating that the model matches what is expected.
    std::vector<int64_t> dims;
    triton::common::TritonJson::Value reshape;
    if (io.Find("reshape", &reshape)) {
      RETURN_IF_ERROR(ParseShape(reshape, "shape", &dims));
    } else {
      RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
    }

    // The batch output shape doesn't necessarily match the model
    if (model_state_->FindBatchOutput(io_name) == nullptr) {
      // Only compare the dimensions if the tensor is not scalar
      if (iit->second.dims_.size() != 0) {
        RETURN_IF_ERROR(CompareDimsSupported(
            model_state_->Name(), io_name, iit->second.dims_, dims,
            model_state_->MaxBatchSize(), true /* compare_exact */));
      } else {
        for (auto& dim : dims) {
          if (dim != 1) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("unable to load model '") + model_state_->Name() +
                 "', scalar tensor '" + io_name +
                 "', should only provide 1 in the model configuration when the "
                 "model doesn't support batching. Model configuration "
                 "provided: " +
                 ShapeToString(dims) + ".")
                    .c_str());
          }
        }

        // store the dimension for reference.
        scalar_outputs_[io_name] = dims;
      }
    }
  }

  triton::common::TritonJson::Value sequence_batching;
  if (model_state_->ModelConfig().Find(
          "sequence_batching", &sequence_batching)) {
    triton::common::TritonJson::Value states;
    if (sequence_batching.MemberAsArray("state", &states)) {
      for (size_t i = 0; i < states.ArraySize(); i++) {
        triton::common::TritonJson::Value state;
        RETURN_IF_ERROR(states.IndexAsObject(i, &state));
        std::string state_name;
        RETURN_IF_ERROR(state.MemberAsString("name", &state_name));
        std::string state_dtype;
        RETURN_IF_ERROR(state.MemberAsString("data_type", &state_dtype));
        std::vector<int64_t> dims;
        RETURN_IF_ERROR(ParseShape(state, "dims", &dims));

        auto iit = output_tensor_infos_.find(state_name);
        if (iit == output_tensor_infos_.end()) {
          RETURN_IF_ERROR(CheckAllowedModelOutput(state, output_tensor_names));
        }

        auto onnx_data_type = ModelConfigDataTypeToOnnxDataType(state_dtype);
        if (onnx_data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              (std::string("unsupported datatype ") + state_dtype +
               " for output state '" + state_name + "' for model '" +
               model_state_->Name() + "'")
                  .c_str());
        } else if (onnx_data_type != iit->second.type_) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("unable to load model '") + model_state_->Name() +
               "', configuration expects datatype " + state_dtype +
               " for output state '" + state_name + "', model provides TYPE_" +
               TRITONSERVER_DataTypeString(
                   ConvertFromOnnxDataType(iit->second.type_)))
                  .c_str());
        }
      }
    }
  }

  return nullptr;  // success
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() + " with " +
       std::to_string(request_count) + " requests")
          .c_str());

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  const int max_batch_size = model_state_->MaxBatchSize();

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  size_t total_batch_size = 0;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to ONNX Runtime backend for '" + Name() +
                  "'")
                  .c_str()));
      return;
    }

    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RequestsRespondWithError(requests, request_count, err);
        return;
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size)) {
    RequestsRespondWithError(
        requests, request_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "batch size " + std::to_string(total_batch_size) + " for '" +
                Name() + "', max allowed is " + std::to_string(max_batch_size))
                .c_str()));
    return;
  }

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response unique_ptr will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error, we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  bool all_response_failed = false;

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  // Use scoped class to clean up ORT tensors and other resources that
  // need to persist until ORT run completes.
  struct ScopedCleanup {
    ScopedCleanup(ModelInstanceState* ctx) : ctx_(ctx) {}
    ~ScopedCleanup()
    {
      if (ctx_ != nullptr) {
        ctx_->ReleaseOrtRunResources();
      }
    }
    ModelInstanceState* ctx_;
  } io_tensor_wrapper(this);

  std::vector<const char*> input_names;
  bool cuda_copy = false;
  BackendInputCollector collector(
      requests, request_count, &responses, model_state_->TritonMemoryManager(),
      model_state_->EnablePinnedInput(), CudaStream(), nullptr, nullptr, 0,
      HostPolicyName().c_str());
  RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
      responses, request_count, all_response_failed,
      SetInputTensors(
          total_batch_size, requests, request_count, &responses, &collector,
          &input_names, &cuda_copy));

  if (!all_response_failed) {
    // Set preferred memory type and id. This will be used while querying
    // memory type to be used for output buffer.
    TRITONSERVER_MemoryType preferred_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t preferred_memory_type_id = 0;
    if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
      preferred_memory_type = TRITONSERVER_MEMORY_GPU;
      preferred_memory_type_id = DeviceId();
    }

    // Request to retrieve all model outputs. 'output_names' and
    // 'output_tensors_' are parallel vectors and so must be kept in
    // sync. [TODO] should collect only the outputs needed by some
    // request.
    for (auto& output_name : StateForModel()->ModelOutputs()) {
      output_tensors_.emplace_back(nullptr);

      TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t memory_type_id = 0;

      // Get data type for this output. If this is a string then
      // use CPU for binding output otherwise, query the preferred location
      // for this output and bind accordingly. In case of any errors we
      // fallback to binding the output to CPU.
      auto iit = output_tensor_infos_.find(output_name.first);
      if (iit == output_tensor_infos_.end()) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_VERBOSE,
            (std::string(
                 "Error while retrieving output data type. Using cpu "
                 "as preferred location for output: " +
                 output_name.first)
                 .c_str()));
      } else if (iit->second.type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        // Query the memory type of destination output buffer. Bind the
        // output to this destination memory type. The destination memory type
        // for an output for all requests should be same. So use any request
        // for this query.
        memory_type = preferred_memory_type;
        memory_type_id = preferred_memory_type_id;
        auto err = TRITONBACKEND_RequestOutputBufferProperties(
            requests[0], output_name.first.c_str(), /*byte_size*/ nullptr,
            &memory_type, &memory_type_id);

        if (err != nullptr) {
          LOG_MESSAGE(
              TRITONSERVER_LOG_VERBOSE,
              (std::string(
                   "Output Properties Unavailable. Using cpu as "
                   "preferred location for output: " +
                   output_name.first +
                   " Error: " + TRITONSERVER_ErrorMessage(err))
                   .c_str()));
          memory_type = TRITONSERVER_MEMORY_CPU;
          memory_type_id = 0;
        }
      }

      // If the cuda allocator is not set, bind the output to CPU.
      if (cuda_allocator_info_ == nullptr) {
        memory_type = TRITONSERVER_MEMORY_CPU;
        memory_type_id = 0;
      }

      // finally save the derived mem type and device id as we need it for
      // reading the outputs.
      output_device_info_[output_name.first] = {memory_type, memory_type_id};

      RESPOND_ALL_AND_SET_TRUE_IF_ORT_ERROR(
          responses, request_count, all_response_failed,
          ort_api->BindOutputToDevice(
              io_binding_, output_name.first.c_str(),
              memory_type == TRITONSERVER_MEMORY_GPU ? cuda_allocator_info_
                                                     : cpu_allocator_info_));
    }
  }

  // Wait for any in-flight input tensor copies to complete.
#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(CudaStream());
  }
#endif

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  if (!all_response_failed) {
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        OrtRun(&responses, request_count));
  }

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  if (!all_response_failed) {
    RESPOND_ALL_AND_SET_TRUE_IF_ERROR(
        responses, request_count, all_response_failed,
        ReadOutputTensors(
            total_batch_size, requests, request_count, &responses));
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  // Send all the responses that haven't already been sent because of
  // an earlier error. Note that the responses are not set to nullptr
  // here as we need that indication below to determine if the request
  // we successful or not.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send onnxruntime backend response");
    }
  }

  // Report statistics for each request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  if (!all_response_failed) {
    // Report the entire batch statistics.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportBatchStatistics(
            TritonModelInstance(), total_batch_size, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting batch request statistics");
  }
}

TRITONSERVER_Error*
ModelInstanceState::OrtRun(
    std::vector<TRITONBACKEND_Response*>* responses,
    const uint32_t response_count)
{
  RETURN_IF_ORT_ERROR(
      ort_api->RunWithBinding(session_, runOptions_, io_binding_));
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, std::vector<const char*>* input_names,
    bool* cuda_copy)
{
  const int max_batch_size = model_state_->MaxBatchSize();

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInputCount(requests[0], &input_count));

  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input* input;
    RETURN_IF_ERROR(
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        input, &input_name, &input_datatype, &input_shape, &input_dims_count,
        nullptr, nullptr));

    input_names->emplace_back(input_name);
    input_tensors_.emplace_back(nullptr);

    std::vector<int64_t> batchn_shape;
    // For a ragged input tensor, the tensor shape should be
    // the flatten shape of the whole batch
    if (StateForModel()->IsInputRagged(input_name)) {
      batchn_shape = std::vector<int64_t>{0};
      for (size_t idx = 0; idx < request_count; idx++) {
        TRITONBACKEND_Input* input;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            TRITONBACKEND_RequestInput(requests[idx], input_name, &input));
        const int64_t* input_shape;
        uint32_t input_dims_count;
        int64_t element_cnt = 0;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]), TRITONBACKEND_InputProperties(
                                      input, nullptr, nullptr, &input_shape,
                                      &input_dims_count, nullptr, nullptr));
        RESPOND_AND_SET_NULL_IF_ERROR(
            &((*responses)[idx]),
            GetElementCount(input_shape, input_dims_count, &element_cnt));

        batchn_shape[0] += element_cnt;
      }
    }
    // The shape for the entire input batch, [total_batch_size, ...]
    else {
      batchn_shape =
          std::vector<int64_t>(input_shape, input_shape + input_dims_count);
      if (max_batch_size != 0) {
        batchn_shape[0] = total_batch_size;
      }
    }

    if (input_datatype != TRITONSERVER_TYPE_BYTES) {
      // The input must be in contiguous CPU memory. Use appropriate
      // allocator info to bind inputs to the right device. .i.e bind inputs
      // to GPU if they are being provided on GPU.
      const char* input_buffer;
      size_t batchn_byte_size;
      TRITONSERVER_MemoryType memory_type;
      int64_t memory_type_id;
      std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>>
          allowed_input_types;
      if (Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
        allowed_input_types = {
            {TRITONSERVER_MEMORY_GPU, DeviceId()},
            {TRITONSERVER_MEMORY_CPU_PINNED, 0},
            {TRITONSERVER_MEMORY_CPU, 0}};
      } else {
        allowed_input_types = {
            {TRITONSERVER_MEMORY_CPU_PINNED, 0}, {TRITONSERVER_MEMORY_CPU, 0}};
      }

      RETURN_IF_ERROR(collector->ProcessTensor(
          input_name, nullptr, 0, allowed_input_types, &input_buffer,
          &batchn_byte_size, &memory_type, &memory_type_id));

      auto iti = input_tensor_infos_.find(input_name);
      if (iti == input_tensor_infos_.end()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                std::string(
                    "Failed to retrieve the ONNX input tensor info from '") +
                input_name + "'.")
                .c_str());
      }

      // Create ORT Tensor
      if (iti->second.dims_.size() == 0) {
        // scalar tensor
        RETURN_IF_ORT_ERROR(ort_api->CreateTensorWithDataAsOrtValue(
            memory_type == TRITONSERVER_MEMORY_GPU ? cuda_allocator_info_
                                                   : cpu_allocator_info_,
            (void*)input_buffer, batchn_byte_size, nullptr /* scalar */,
            0 /* number of dims */, ConvertToOnnxDataType(input_datatype),
            &input_tensors_.back()));
      } else {
        RETURN_IF_ORT_ERROR(ort_api->CreateTensorWithDataAsOrtValue(
            memory_type == TRITONSERVER_MEMORY_GPU ? cuda_allocator_info_
                                                   : cpu_allocator_info_,
            (void*)input_buffer, batchn_byte_size, batchn_shape.data(),
            batchn_shape.size(), ConvertToOnnxDataType(input_datatype),
            &input_tensors_.back()));
      }
      RETURN_IF_ORT_ERROR(
          ort_api->BindInput(io_binding_, input_name, input_tensors_.back()));
    } else {
      // For BYTES input, we need to convert the serialized string
      // representation into what is required for ORT. ORT expects a
      // vector of char*, one for each element. For each tensor we get
      // a copy of the data in a contiguous CPU buffer and then
      // in-place modify that from the Triton
      // <int32_len><bytes><int32_len><bytes>... serialization into a
      // <bytes><null-terminator><bytes><null-terminator>... serialization
      // and then initialize 'string_ptrs' to point to each <bytes>.
      std::vector<const char*> string_ptrs;

      SetStringInputTensor(
          requests, request_count, responses, input_name, &string_ptrs,
          cuda_copy);

      RETURN_IF_ORT_ERROR(ort_api->CreateTensorAsOrtValue(
          default_allocator_, batchn_shape.data(), batchn_shape.size(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, &input_tensors_.back()));
      RETURN_IF_ORT_ERROR(ort_api->FillStringTensor(
          input_tensors_.back(), string_ptrs.data(), string_ptrs.size()));
      RETURN_IF_ORT_ERROR(
          ort_api->BindInput(io_binding_, input_name, input_tensors_.back()));
    }
  }

  // Process batch input if any
  for (const auto& batch_input : StateForModel()->BatchInputs()) {
    std::vector<int64_t> shape;
    collector->BatchInputShape(batch_input, &shape);

    for (const auto& input_name : batch_input.TargetNames()) {
      input_names->emplace_back(input_name.c_str());
      input_tensors_.emplace_back(nullptr);

      const char* dst_buffer;
      size_t dst_buffer_byte_size;
      TRITONSERVER_MemoryType dst_memory_type;
      int64_t dst_memory_type_id;

      // Batch inputs are always created on CPU
      RESPOND_ALL_AND_SET_NULL_IF_ERROR(
          (*responses), responses->size(),
          collector->ProcessBatchInput(
              batch_input, nullptr, 0, {{TRITONSERVER_MEMORY_CPU, 0}},
              &dst_buffer, &dst_buffer_byte_size, &dst_memory_type,
              &dst_memory_type_id));

      // Create ORT Tensor
      RETURN_IF_ORT_ERROR(ort_api->CreateTensorWithDataAsOrtValue(
          cpu_allocator_info_, (void*)dst_buffer, dst_buffer_byte_size,
          shape.data(), shape.size(),
          ConvertToOnnxDataType(batch_input.DataType()),
          &input_tensors_.back()));

      RETURN_IF_ORT_ERROR(ort_api->BindInput(
          io_binding_, input_name.c_str(), input_tensors_.back()));
    }
  }

  // Finalize...
  *cuda_copy |= collector->Finalize();
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::SetStringInputTensor(
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses, const char* input_name,
    std::vector<const char*>* string_ptrs, bool* cuda_copy)
{
  size_t total_byte_size = 0;
  std::vector<size_t> expected_byte_sizes;
  std::vector<size_t> expected_element_cnts;
  expected_byte_sizes.reserve(request_count);
  expected_element_cnts.reserve(request_count);
  for (size_t ridx = 0; ridx < request_count; ++ridx) {
    TRITONBACKEND_Input* in;
    RESPOND_AND_SET_NULL_IF_ERROR(
        &((*responses)[ridx]),
        TRITONBACKEND_RequestInput(requests[ridx], input_name, &in));

    const int64_t* input_shape;
    uint32_t input_dims_count;
    uint64_t input_byte_size;
    RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
        in, nullptr, nullptr, &input_shape, &input_dims_count, &input_byte_size,
        nullptr));

    // Skip input in this request if error response has already been sent.
    if ((*responses)[ridx] == nullptr) {
      expected_byte_sizes.push_back(0);
      expected_element_cnts.push_back(0);
    } else {
      int64_t element_cnt = 0;
      RETURN_IF_ERROR(
          GetElementCount(input_shape, input_dims_count, &element_cnt));
      expected_element_cnts.push_back(element_cnt);
      expected_byte_sizes.push_back(input_byte_size);
    }

    total_byte_size += expected_byte_sizes.back();
  }

  // For string input, the copy to contiguous buffer is needed because ORT
  // expects elements to be C strings thus we need to modify input buffer.
  // Reserve one more byte at the end of input_buffer to ensure last
  // element of String data can become valid C string.
  BackendMemory* input_memory;
  RETURN_IF_ERROR(BackendMemory::Create(
      model_state_->TritonMemoryManager(),
      {BackendMemory::AllocationType::CPU_PINNED_POOL,
       BackendMemory::AllocationType::CPU},
      0 /* memory_type_id */, total_byte_size + 1, &input_memory));
  input_tensor_memories_.push_back(input_memory);

  const TRITONSERVER_MemoryType mem_type = input_memory->MemoryType();
  char* input_buffer = input_memory->MemoryPtr();

  size_t buffer_offset = 0;
  for (size_t ridx = 0; ridx < request_count; ++ridx) {
    TRITONBACKEND_Input* in;
    TRITONSERVER_Error* err =
        TRITONBACKEND_RequestInput(requests[ridx], input_name, &in);
    if ((err == nullptr) && ((*responses)[ridx] != nullptr)) {
      uint32_t input_buffer_count;
      RETURN_IF_ERROR(TRITONBACKEND_InputPropertiesForHostPolicy(
          in, HostPolicyName().c_str(), nullptr, nullptr, nullptr, nullptr,
          nullptr, &input_buffer_count));

      size_t input_offset = 0;
      for (size_t idx = 0; idx < input_buffer_count; ++idx) {
        const void* src_buffer;
        size_t src_byte_size;
        TRITONSERVER_MemoryType src_memory_type;
        int64_t src_memory_type_id;
        err = TRITONBACKEND_InputBufferForHostPolicy(
            in, HostPolicyName().c_str(), idx, &src_buffer, &src_byte_size,
            &src_memory_type, &src_memory_type_id);
        if (err == nullptr) {
          if ((input_offset + src_byte_size) > expected_byte_sizes[ridx]) {
            err = TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INVALID_ARG,
                (std::string("buffer size for input '") + input_name +
                 "' exceeds batch byte size " +
                 std::to_string(expected_byte_sizes[ridx]))
                    .c_str());
          } else {
            bool cuda_used = false;
            err = CopyBuffer(
                input_name, src_memory_type, src_memory_type_id, mem_type, 0,
                src_byte_size, src_buffer,
                input_buffer + buffer_offset + input_offset, CudaStream(),
                &cuda_used);
            *cuda_copy |= cuda_used;
          }
        }

        if (err == nullptr) {
          input_offset += src_byte_size;
        } else {
          break;
        }
      }
    }

    if (err != nullptr) {
      if ((*responses)[ridx] != nullptr) {
        RESPOND_AND_SET_NULL_IF_ERROR(&((*responses)[ridx]), err);
      }

      TRITONSERVER_ErrorDelete(err);
    }

    buffer_offset += expected_byte_sizes[ridx];
  }

#ifdef TRITON_ENABLE_GPU
  // Synchronize to ensure the buffer is ready to be modified
  if (*cuda_copy) {
    cudaStreamSynchronize(CudaStream());
    *cuda_copy = false;
  }
#endif  // TRITON_ENABLE_GPU

  // Modify input buffer and set string expected by ORT
  SetStringInputBuffer(
      input_name, expected_byte_sizes, expected_element_cnts, responses,
      input_buffer, string_ptrs);
  input_buffer[total_byte_size] = 0;
  return nullptr;
}

void
ModelInstanceState::SetStringInputBuffer(
    const std::string& input_name,
    const std::vector<size_t>& expected_byte_sizes,
    const std::vector<size_t>& expected_element_cnts,
    std::vector<TRITONBACKEND_Response*>* responses, char* input_buffer,
    std::vector<const char*>* string_ptrs)
{
  std::vector<std::pair<const char*, const uint32_t>> str_list;
  // offset for each response
  size_t buffer_copy_offset = 0;
  for (size_t idx = 0; idx < expected_byte_sizes.size(); idx++) {
    const size_t expected_byte_size = expected_byte_sizes[idx];
    const size_t expected_element_cnt = expected_element_cnts[idx];

    if ((*responses)[idx] != nullptr) {
      char* data_content = input_buffer + buffer_copy_offset;
      TRITONSERVER_Error* err = ValidateStringBuffer(
          data_content, expected_byte_size, expected_element_cnt,
          input_name.c_str(), &str_list);
      // Set string values.
      for (const auto& [addr, len] : str_list) {
        // Make first byte of size info 0, so that if there is string data
        // in front of it, the data becomes valid C string.
        *const_cast<char*>(addr - sizeof(uint32_t)) = 0;
        string_ptrs->push_back(addr);
      }

      size_t element_cnt = str_list.size();
      if (err != nullptr) {
        RESPOND_AND_SET_NULL_IF_ERROR(&((*responses)[idx]), err);
        FillStringData(string_ptrs, expected_element_cnt - element_cnt);
      }
      str_list.clear();
    }
    buffer_copy_offset += expected_byte_size;
  }
}

void
ModelInstanceState::FillStringData(
    std::vector<const char*>* string_ptrs, size_t cnt)
{
  static const char* empty = "";
  for (size_t c = 0; c < cnt; c++) {
    string_ptrs->push_back(empty);
  }
}

TRITONSERVER_Error*
ModelInstanceState::ReadOutputTensor(
    std::vector<int64_t>& batchn_shape, TRITONSERVER_DataType& dtype,
    OrtValue* output_tensor, void** output_buffer,
    std::vector<std::vector<char>>& string_buffers,
    std::vector<size_t>& offsets)
{
  // Get output type and shape
  OrtTypeInfo* typeinfo;
  RETURN_IF_ORT_ERROR(ort_api->GetTypeInfo(output_tensor, &typeinfo));
  std::unique_ptr<OrtTypeInfo, TypeInfoDeleter> typeinfo_wrapper(typeinfo);

  const OrtTensorTypeAndShapeInfo* type_and_shape;
  RETURN_IF_ORT_ERROR(
      ort_api->CastTypeInfoToTensorInfo(typeinfo, &type_and_shape));

  size_t num_dims;
  RETURN_IF_ORT_ERROR(ort_api->GetDimensionsCount(type_and_shape, &num_dims));
  batchn_shape.resize(num_dims);
  RETURN_IF_ORT_ERROR(ort_api->GetDimensions(
      type_and_shape, batchn_shape.data(), batchn_shape.size()));

  ONNXTensorElementDataType type;
  RETURN_IF_ORT_ERROR(ort_api->GetTensorElementType(type_and_shape, &type));
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    int64_t element_count = 0;
    size_t total_length = 0;
    RETURN_IF_ERROR(GetElementCount(batchn_shape, &element_count));
    RETURN_IF_ORT_ERROR(
        ort_api->GetStringTensorDataLength(output_tensor, &total_length));

    string_buffers.emplace_back(std::vector<char>(total_length));
    auto content = string_buffers.back().data();
    offsets.reserve(element_count + 1);
    RETURN_IF_ORT_ERROR(ort_api->GetStringTensorContent(
        output_tensor, content, total_length, offsets.data(), element_count));
    // Mark "passed end byte offset"
    offsets[element_count] = total_length;

  } else {
    // Fixed size data type...
    RETURN_IF_ORT_ERROR(
        ort_api->GetTensorMutableData(output_tensor, output_buffer));
  }

  dtype = ConvertFromOnnxDataType(type);

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->TritonMemoryManager(),
      model_state_->MaxBatchSize() > 0, model_state_->EnablePinnedInput(),
      CudaStream());

  // Use to hold string output contents
  bool cuda_copy = false;
  auto& model_outputs = StateForModel()->ModelOutputs();

  size_t output_count = 0;
  RETURN_IF_ORT_ERROR(ort_api->GetBoundOutputValues(
      io_binding_, default_allocator_, &output_buffer_, &output_count));
  if (output_count != model_outputs.size()) {
    RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        ("Retrieved output count is not equal to expected count.")));
  }

  std::vector<std::vector<char>> string_buffers;
  auto model_outputs_it = model_outputs.begin();
  for (size_t idx = 0; idx < model_outputs.size(); idx++, model_outputs_it++) {
    OrtValue* output_tensor = output_tensors_[idx] = output_buffer_[idx];
    const std::string& name = model_outputs_it->first;
    auto& output_tensor_pair = model_outputs_it->second;

    auto output_device_info_iter = output_device_info_.find(name);
    if (output_device_info_iter == output_device_info_.end()) {
      RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("device info for output tensor '") + name +
           "' not found")
              .c_str()));
    }

    const auto& alloc_perference = output_device_info_iter->second;

    const BatchOutput* batch_output = StateForModel()->FindBatchOutput(name);
    if (batch_output == nullptr) {
      if (output_tensor == nullptr) {
        RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("output tensor '") + name + "' is not found")
                .c_str()));
      }

      std::vector<int64_t> batchn_shape;
      TRITONSERVER_DataType dtype;
      void* output_buffer;
      std::vector<std::vector<char>> string_buffers;
      std::vector<size_t> offsets;

      RETURN_IF_ERROR(ReadOutputTensor(
          batchn_shape, dtype, output_tensor, &output_buffer, string_buffers,
          offsets));

      // If the number of dimensions is equal to zero, it means that it is a
      // scalar and it would use the dimensions specified in the model
      // configuration.
      if (batchn_shape.size() == 0) {
        auto scalar_output_dims_it = scalar_outputs_.find(name);
        if (scalar_output_dims_it == scalar_outputs_.end()) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "Failed to find the scalar output dimension for " + name +
                  " in the model configuration.")
                  .c_str());
        }
        batchn_shape = scalar_output_dims_it->second;
      }

      if (output_tensor_pair.first != -1) {
        if (dtype == TRITONSERVER_TYPE_BYTES) {
          auto content = string_buffers.back().data();
          cuda_copy |= SetStringOutputBuffer(
              name, content, offsets.data(), &batchn_shape, requests,
              request_count, responses);
        } else {
          responder.ProcessTensor(
              name, dtype, batchn_shape, reinterpret_cast<char*>(output_buffer),
              alloc_perference.first, alloc_perference.second);
        }
      }

      if (output_tensor_pair.second != -1) {
        std::vector<TRITONBACKEND_State*> states;
        if (dtype == TRITONSERVER_TYPE_BYTES) {
          auto content = string_buffers.back().data();
          cuda_copy |= SetStringStateBuffer(
              name, content, offsets.data(), &batchn_shape, requests,
              request_count, responses);
        } else {
          states = responder.ProcessStateTensor(
              name, dtype, batchn_shape, reinterpret_cast<char*>(output_buffer),
              alloc_perference.first, alloc_perference.second);
        }

        // Update the states
        for (auto& state : states) {
          RETURN_IF_ERROR(TRITONBACKEND_StateUpdate(state));
        }
      }


    } else {
      char* output_buffer = nullptr;
      RETURN_IF_ORT_ERROR(
          ort_api->GetTensorMutableData(output_tensor, (void**)&output_buffer));
      responder.ProcessBatchOutput(
          name, *batch_output, output_buffer, alloc_perference.first,
          alloc_perference.second);
    }
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif  // TRITON_ENABLE_GPU
  return nullptr;
}

bool
ModelInstanceState::SetStringStateBuffer(
    const std::string& name, const char* content, const size_t* offsets,
    std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  return SetStringBuffer(
      name, content, offsets, batchn_shape, requests, request_count, responses,
      true /* state */);
}

bool
ModelInstanceState::SetStringOutputBuffer(
    const std::string& name, const char* content, const size_t* offsets,
    std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  return SetStringBuffer(
      name, content, offsets, batchn_shape, requests, request_count, responses,
      false /* state */);
}
bool
ModelInstanceState::SetStringBuffer(
    const std::string& name, const char* content, const size_t* offsets,
    std::vector<int64_t>* batchn_shape, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses, bool state)
{
  size_t element_idx = 0;
  bool cuda_copy = false;
  for (size_t ridx = 0; ridx < request_count; ++ridx) {
    const auto& request = requests[ridx];
    auto& response = (*responses)[ridx];

    // batchn_shape holds the shape of the entire tensor batch. When
    // batching is enabled override the first batch dimension with each
    // requests batch size (reusing for efficiency).
    if (model_state_->MaxBatchSize() > 0) {
      TRITONBACKEND_Input* input;
      TRITONBACKEND_RequestInputByIndex(request, 0 /* index */, &input);
      const int64_t* shape;
      TRITONBACKEND_InputProperties(
          input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
      (*batchn_shape)[0] = shape[0];
    }

    int64_t expected_element_cnt = 0;
    RESPOND_AND_SET_NULL_IF_ERROR(
        &response, GetElementCount(*batchn_shape, &expected_element_cnt));

    // If 'request' requested this output then copy it from
    // 'content'. If it did not request this output then just skip it
    // in the 'content'.
    bool need_output = false;
    if (!state) {
      if (response != nullptr) {
        uint32_t output_count;
        RESPOND_AND_SET_NULL_IF_ERROR(
            &response,
            TRITONBACKEND_RequestOutputCount(request, &output_count));
        if (response != nullptr) {
          for (uint32_t output_idx = 0; output_idx < output_count;
               output_idx++) {
            const char* req_output_name;
            RESPOND_AND_SET_NULL_IF_ERROR(
                &response, TRITONBACKEND_RequestOutputName(
                               request, output_idx, &req_output_name));
            if ((response != nullptr) && (req_output_name == name)) {
              need_output = true;
              break;
            }
          }
        }
      }
    } else {
      // need_output must be always set to true for state tensors.
      need_output = true;
    }

    if (need_output) {
      TRITONSERVER_Error* err;
      TRITONBACKEND_Output* response_output;
      TRITONBACKEND_State* response_state;
      if (!state) {
        err = TRITONBACKEND_ResponseOutput(
            response, &response_output, name.c_str(), TRITONSERVER_TYPE_BYTES,
            batchn_shape->data(), batchn_shape->size());
      } else {
        err = TRITONBACKEND_StateNew(
            &response_state, request, name.c_str(), TRITONSERVER_TYPE_BYTES,
            batchn_shape->data(), batchn_shape->size());
      }
      if (err == nullptr) {
        // Calculate expected byte size in advance using string offsets
        const size_t data_byte_size =
            offsets[element_idx + expected_element_cnt] - offsets[element_idx];
        const size_t expected_byte_size =
            data_byte_size + sizeof(uint32_t) * expected_element_cnt;

        TRITONSERVER_MemoryType actual_memory_type =
            TRITONSERVER_MEMORY_CPU_PINNED;
        int64_t actual_memory_type_id = 0;
        void* buffer;
        if (!state) {
          err = TRITONBACKEND_OutputBuffer(
              response_output, &buffer, expected_byte_size, &actual_memory_type,
              &actual_memory_type_id);
        } else {
          err = TRITONBACKEND_StateBuffer(
              response_state, &buffer, expected_byte_size, &actual_memory_type,
              &actual_memory_type_id);
        }
        if (err == nullptr) {
          bool cuda_used = false;
          size_t copied_byte_size = 0;
          for (size_t e = 0; e < expected_element_cnt; ++e) {
            const uint32_t len =
                offsets[element_idx + e + 1] - offsets[element_idx + e];
            // Prepend size of the string
            err = CopyBuffer(
                name, TRITONSERVER_MEMORY_CPU /* src_memory_type */,
                0 /* src_memory_type_id */, actual_memory_type,
                actual_memory_type_id, sizeof(uint32_t),
                static_cast<const void*>(&len),
                static_cast<char*>(buffer) + copied_byte_size, stream_,
                &cuda_used);
            if (err != nullptr) {
              break;
            }

            cuda_copy |= cuda_used;
            copied_byte_size += sizeof(uint32_t);

            // Copy raw string content
            err = CopyBuffer(
                name, TRITONSERVER_MEMORY_CPU /* src_memory_type */,
                0 /* src_memory_type_id */, actual_memory_type,
                actual_memory_type_id, len, content + offsets[element_idx + e],
                static_cast<char*>(buffer) + copied_byte_size, stream_,
                &cuda_used);
            if (err != nullptr) {
              break;
            }

            cuda_copy |= cuda_used;
            copied_byte_size += len;
          }
        }
      }

      RESPOND_AND_SET_NULL_IF_ERROR(&response, err);
      if (state) {
        RESPOND_AND_SET_NULL_IF_ERROR(
            &response, TRITONBACKEND_StateUpdate(response_state));
      }
    }

    element_idx += expected_element_cnt;
  }

  return cuda_copy;
}

/////////////

extern "C" {

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("Triton TRITONBACKEND API version: ") +
         std::to_string(api_version_major) + "." +
         std::to_string(api_version_minor) + " does not support '" + name +
         "' TRITONBACKEND API version: " +
         std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
         std::to_string(TRITONBACKEND_API_VERSION_MINOR))
            .c_str());
  }

  // The backend configuration may contain information needed by the
  // ort backend, such as command-line arguments.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  triton::common::TritonJson::Value backend_config;
  TRITONSERVER_Error* err = nullptr;
  if (byte_size != 0) {
    err = backend_config.Parse(buffer, byte_size);
  }

  RETURN_IF_ERROR(err);

  // Onetime initialization for the onnxruntime loader.
  RETURN_IF_ERROR(OnnxLoader::Init(backend_config));

  std::unique_ptr<BackendConfiguration> lconfig(new BackendConfiguration());
  triton::common::TritonJson::Value cmdline;
  if (backend_config.Find("cmdline", &cmdline)) {
    triton::common::TritonJson::Value value;
    std::string value_str;
    if (cmdline.Find("default-max-batch-size", &value)) {
      RETURN_IF_ERROR(value.AsString(&value_str));
      int lvalue;
      RETURN_IF_ERROR(ParseIntValue(value_str, &lvalue));
      lconfig->default_max_batch_size_ = lvalue;
    }
  }
  // Check if device memory tracker is explicitly enabled
  if (DeviceMemoryTracker::EnableFromBackendConfig(backend_config)) {
    lconfig->enable_memory_tracker_ = DeviceMemoryTracker::Init();
  }
  RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(
      backend, reinterpret_cast<void*>(lconfig.get())));

  lconfig.release();
  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  if (BackendConfiguration::RetrieveFrom(backend).enable_memory_tracker_) {
    DeviceMemoryTracker::Fini();
  }
  LOG_IF_ERROR(OnnxLoader::Stop(), "failed to stop OnnxLoader");
  void* state = nullptr;
  LOG_IF_ERROR(
      TRITONBACKEND_BackendState(backend, &state),
      "failed to get backend state");
  if (state != nullptr) {
    delete reinterpret_cast<BackendConfiguration*>(state);
  }
  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());


  // Utilizing DeviceMemoryTracker behavior that function calls with
  // 'nullptr' for usage will be no-ops.
  std::unique_ptr<DeviceMemoryTracker::MemoryUsage> lusage;
  if (BackendConfiguration::RetrieveFrom(model).enable_memory_tracker_) {
    lusage.reset(new DeviceMemoryTracker::MemoryUsage());
    DeviceMemoryTracker::TrackThreadMemoryUsage(lusage.get());
  }


  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  if (lusage) {
    DeviceMemoryTracker::UntrackThreadMemoryUsage(lusage.get());
    TRITONSERVER_BufferAttributes** ba_array;
    uint32_t ba_len = 0;
    RETURN_IF_ERROR(lusage->SerializeToBufferAttributes(&ba_array, &ba_len));
    RETURN_IF_ERROR(
        TRITONBACKEND_ModelReportMemoryUsage(model, ba_array, ba_len));
  }
  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  // NOTE: If the corresponding TRITONBACKEND_BackendAttribute is enabled by the
  // backend for parallel model instance loading, the
  // TRITONBACKEND_ModelInstanceInitialize may be called concurrently.
  // Therefore, this function should be thread-safe.
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Utilizing DeviceMemoryTracker behavior that function calls with
  // 'nullptr' for usage will be no-ops.
  std::unique_ptr<DeviceMemoryTracker::MemoryUsage> lusage;
  if (BackendConfiguration::RetrieveFrom(instance).enable_memory_tracker_) {
    lusage.reset(new DeviceMemoryTracker::MemoryUsage());
    DeviceMemoryTracker::TrackThreadMemoryUsage(lusage.get());
  }


  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  if (lusage) {
    DeviceMemoryTracker::UntrackThreadMemoryUsage(lusage.get());
    TRITONSERVER_BufferAttributes** ba_array;
    uint32_t ba_len = 0;
    RETURN_IF_ERROR(lusage->SerializeToBufferAttributes(&ba_array, &ba_len));
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceReportMemoryUsage(
        instance, ba_array, ba_len));
  }

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.
  instance_state->ProcessRequests(requests, request_count);

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_GetBackendAttribute(
    TRITONBACKEND_Backend* backend,
    TRITONBACKEND_BackendAttribute* backend_attributes)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      "TRITONBACKEND_GetBackendAttribute: setting attributes");
  // This backend can safely handle parallel calls to
  // TRITONBACKEND_ModelInstanceInitialize (thread-safe).
  RETURN_IF_ERROR(TRITONBACKEND_BackendAttributeSetParallelModelInstanceLoading(
      backend_attributes, true));

  return nullptr;
}

}  // extern "C"
}}}  // namespace triton::backend::onnxruntime
