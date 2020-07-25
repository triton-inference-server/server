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

#include "src/backends/backend/examples/backend_utils.h"

#include <dirent.h>
#include <sys/stat.h>

namespace nvidia { namespace inferenceserver { namespace backend {

TRITONSERVER_Error*
ParseShape(
    TritonJson::Value& io, const std::string& name, std::vector<int64_t>* shape)
{
  TritonJson::Value shape_array;
  RETURN_IF_ERROR(io.MemberAsArray(name.c_str(), &shape_array));
  for (size_t i = 0; i < shape_array.ArraySize(); ++i) {
    int64_t d;
    RETURN_IF_ERROR(shape_array.IndexAsInt(i, &d));
    shape->push_back(d);
  }

  return nullptr;  // success
}

std::string
ShapeToString(const int64_t* dims, const size_t dims_count)
{
  bool first = true;

  std::string str("[");
  for (size_t i = 0; i < dims_count; ++i) {
    const int64_t dim = dims[i];
    if (!first) {
      str += ",";
    }
    str += std::to_string(dim);
    first = false;
  }

  str += "]";
  return str;
}

std::string
ShapeToString(const std::vector<int64_t>& shape)
{
  return ShapeToString(shape.data(), shape.size());
}

int64_t
GetElementCount(const int64_t* dims, const size_t dims_count)
{
  bool first = true;
  int64_t cnt = 0;
  for (size_t i = 0; i < dims_count; i++) {
    if (dims[i] == WILDCARD_DIM) {
      return -1;
    }

    if (first) {
      cnt = dims[i];
      first = false;
    } else {
      cnt *= dims[i];
    }
  }

  return cnt;
}

int64_t
GetElementCount(const std::vector<int64_t>& shape)
{
  return GetElementCount(shape.data(), shape.size());
}

int64_t
GetByteSize(
    const TRITONSERVER_DataType& dtype, const std::vector<int64_t>& dims)
{
  size_t dt_size = TRITONSERVER_DataTypeByteSize(dtype);
  if (dt_size == 0) {
    return -1;
  }

  int64_t cnt = GetElementCount(dims);
  if (cnt == -1) {
    return -1;
  }

  return cnt * dt_size;
}

TRITONSERVER_Error*
ReadInputTensor(
    TRITONBACKEND_Request* request, const std::string& input_name, char* buffer,
    size_t* buffer_byte_size)
{
  TRITONBACKEND_Input* input;
  RETURN_IF_ERROR(
      TRITONBACKEND_RequestInput(request, input_name.c_str(), &input));

  uint64_t input_byte_size;
  uint32_t input_buffer_count;
  RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
      input, nullptr, nullptr, nullptr, nullptr, &input_byte_size,
      &input_buffer_count));
  RETURN_ERROR_IF_FALSE(
      input_byte_size <= *buffer_byte_size, TRITONSERVER_ERROR_INVALID_ARG,
      std::string(
          "buffer to small for input tensor '" + input_name + "', " +
          std::to_string(*buffer_byte_size) + " < " +
          std::to_string(input_byte_size)));

  size_t output_buffer_offset = 0;
  for (uint32_t b = 0; b < input_buffer_count; ++b) {
    const void* input_buffer = nullptr;
    uint64_t input_buffer_byte_size = 0;
    TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t input_memory_type_id = 0;
    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        input, b, &input_buffer, &input_buffer_byte_size, &input_memory_type,
        &input_memory_type_id));
    RETURN_ERROR_IF_FALSE(
        input_memory_type != TRITONSERVER_MEMORY_GPU,
        TRITONSERVER_ERROR_INTERNAL,
        std::string("expected input tensor in CPU memory"));

    memcpy(buffer + output_buffer_offset, input_buffer, input_buffer_byte_size);
    output_buffer_offset += input_buffer_byte_size;
  }

  *buffer_byte_size = input_byte_size;

  return nullptr;  // success
}

TRITONSERVER_Error*
CheckAllowedModelInput(
    TritonJson::Value& io, const std::set<std::string>& allowed)
{
  std::string io_name;
  RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
  if (allowed.find(io_name) == allowed.end()) {
    std::string astr;
    for (const auto& a : allowed) {
      if (!astr.empty()) {
        astr.append(", ");
      }
      astr.append(a);
    }

    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected inference input '" + io_name +
            "', allowed inputs are: " + astr)
            .c_str());
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
CheckAllowedModelOutput(
    TritonJson::Value& io, const std::set<std::string>& allowed)
{
  std::string io_name;
  RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
  if (allowed.find(io_name) == allowed.end()) {
    std::string astr;
    for (const auto& a : allowed) {
      if (!astr.empty()) {
        astr.append(", ");
      }
      astr.append(a);
    }

    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected inference output '" + io_name +
            "', allowed outputs are: " + astr)
            .c_str());
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
GetBooleanSequenceControlProperties(
    TritonJson::Value& batcher, const std::string& model_name,
    const std::string& control_kind, const bool required,
    std::string* tensor_name, std::string* tensor_datatype,
    float* fp32_false_value, float* fp32_true_value, int32_t* int32_false_value,
    int32_t* int32_true_value)
{
  // Make sure same tensor is not configured for multiple controls
  std::set<std::string> seen_tensors;

  // Make sure the control kind is not mentioned multiple times.
  bool seen_control = false;

  TritonJson::Value control_inputs;
  if (batcher.Find("control_input", &control_inputs)) {
    for (size_t ci_idx = 0; ci_idx < control_inputs.ArraySize(); ci_idx++) {
      TritonJson::Value control_input;
      RETURN_IF_ERROR(control_inputs.IndexAsObject(ci_idx, &control_input));
      std::string input_name;
      RETURN_IF_ERROR(control_input.MemberAsString("name", &input_name));
      if (input_name.empty()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string(
                 "sequence batching control tensor must have a name for ") +
             model_name)
                .c_str());
      }

      if (seen_tensors.find(input_name) != seen_tensors.end()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("sequence batching control tensor '") + input_name +
             "' is specified for multiple control kinds for " + model_name)
                .c_str());
      }

      seen_tensors.insert(input_name);
      TritonJson::Value controls;
      if (control_input.Find("control", &controls)) {
        for (size_t c_idx = 0; c_idx < controls.ArraySize(); c_idx++) {
          TritonJson::Value c;
          RETURN_IF_ERROR(controls.IndexAsObject(c_idx, &c));
          std::string kind_str;
          RETURN_IF_ERROR(c.MemberAsString("kind", &kind_str));
          if (kind_str == control_kind) {
            if (seen_control) {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string(
                       "sequence batching specifies multiple " + control_kind +
                       " tensors for " + model_name)
                       .c_str()));
            }

            *tensor_name = input_name;
            seen_control = true;

            TritonJson::Value int32_false_true, fp32_false_true;
            bool found_int32 =
                (c.Find("int32_false_true", &int32_false_true) &&
                 (int32_false_true.ArraySize() > 0));
            bool found_fp32 =
                (c.Find("fp32_false_true", &fp32_false_true) &&
                 (fp32_false_true.ArraySize() > 0));
            if (found_fp32 && found_int32) {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string(
                       "sequence batching specifies both 'int32_false_true' "
                       "and "
                       "'fp32_false_true' for " +
                       control_kind + " for " + model_name))
                      .c_str());
            }
            if (!(found_int32 || found_fp32)) {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string(
                       "sequence batching must specify either "
                       "'int32_false_true' or "
                       "'fp32_false_true' for " +
                       control_kind + " for " + model_name))
                      .c_str());
            }
            if (found_int32) {
              if (int32_false_true.ArraySize() != 2) {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    (std::string(
                         "sequence batching control 'int32_false_true' must "
                         "have "
                         "exactly 2 entries for " +
                         control_kind + " for " + model_name))
                        .c_str());
              }
              if (tensor_datatype != nullptr) {
                *tensor_datatype = "TYPE_INT32";
              }
              if (int32_false_value != nullptr) {
                int64_t value;
                RETURN_IF_ERROR(int32_false_true.IndexAsInt(0, &value));
                *int32_false_value = value;
              }
              if (int32_true_value != nullptr) {
                int64_t value;
                RETURN_IF_ERROR(int32_false_true.IndexAsInt(1, &value));
                *int32_true_value = value;
              }
            } else {
              if (fp32_false_true.ArraySize() != 2) {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INVALID_ARG,
                    (std::string(
                         "sequence batching control 'fp32_false_true' must "
                         "have exactly "
                         "2 entries for " +
                         control_kind + " for " + model_name))
                        .c_str());
              }

              if (tensor_datatype != nullptr) {
                *tensor_datatype = "TYPE_FP32";
              }
              if (fp32_false_value != nullptr) {
                double value = 0.0;
                RETURN_IF_ERROR(fp32_false_true.IndexAsDouble(0, &value));
                *fp32_false_value = value;
              }
              if (fp32_true_value != nullptr) {
                double value = 0.0;
                RETURN_IF_ERROR(fp32_false_true.IndexAsDouble(1, &value));
                *fp32_true_value = value;
              }
            }
          }
        }
      }
    }
  }

  if (!seen_control) {
    if (required) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string(
               "sequence batching control tensor must specify a " +
               control_kind + " value for " + model_name))
              .c_str());
    }

    tensor_name->clear();
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
GetTypedSequenceControlProperties(
    TritonJson::Value& batcher, const std::string& model_name,
    const std::string& control_kind, const bool required,
    std::string* tensor_name, std::string* tensor_datatype)
{
  // Make sure same tensor is not configured for multiple controls
  std::set<std::string> seen_tensors;

  // Make sure the control kind is not mentioned multiple times.
  bool seen_control = false;

  TritonJson::Value control_inputs;
  if (batcher.Find("control_input", &control_inputs)) {
    for (size_t ci_idx = 0; ci_idx < control_inputs.ArraySize(); ci_idx++) {
      TritonJson::Value control_input;
      RETURN_IF_ERROR(control_inputs.IndexAsObject(ci_idx, &control_input));
      std::string input_name;
      RETURN_IF_ERROR(control_input.MemberAsString("name", &input_name));
      if (input_name.empty()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string(
                 "sequence batching control tensor must have a name for ") +
             model_name)
                .c_str());
      }
      if (seen_tensors.find(input_name) != seen_tensors.end()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("sequence batching control tensor '") + input_name +
             "' is specified for multiple control kinds for " + model_name)
                .c_str());
      }

      seen_tensors.insert(input_name);
      TritonJson::Value controls;
      if (control_input.Find("control", &controls)) {
        for (size_t c_idx = 0; c_idx < controls.ArraySize(); c_idx++) {
          TritonJson::Value c;
          RETURN_IF_ERROR(controls.IndexAsObject(c_idx, &c));
          std::string kind_str;
          RETURN_IF_ERROR(c.MemberAsString("kind", &kind_str));
          if (kind_str == control_kind) {
            if (seen_control) {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string(
                       "sequence batching specifies multiple " + control_kind +
                       " tensors for " + model_name)
                       .c_str()));
            }

            *tensor_name = input_name;
            if (tensor_datatype != nullptr) {
              c.MemberAsString("data_type", tensor_datatype);
            }

            seen_control = true;

            TritonJson::Value int32_false_true, fp32_false_true;
            bool found_int32 =
                (c.Find("int32_false_true", &int32_false_true) &&
                 (int32_false_true.ArraySize() > 0));
            bool found_fp32 =
                (c.Find("fp32_false_true", &fp32_false_true) &&
                 (fp32_false_true.ArraySize() > 0));
            if (found_int32 || found_fp32) {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  (std::string(
                       "sequence batching must not specify either "
                       "'int32_false_true' "
                       "nor 'fp32_false_true' for " +
                       control_kind + " for " + model_name))
                      .c_str());
            }
          }
        }
      }
    }
  }

  if (!seen_control) {
    if (required) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string(
               "sequence batching control tensor must specify a " +
               control_kind + " value for " + model_name))
              .c_str());
    }

    tensor_name->clear();
  }

  return nullptr;  // success
}

namespace {
#ifdef TRITON_ENABLE_GPU
#define RETURN_IF_CUDA_ERR(X, MSG)                                        \
  do {                                                                    \
    cudaError_t err__ = (X);                                              \
    if (err__ != cudaSuccess) {                                           \
      return TRITONSERVER_ErrorNew(                                       \
          TRITONSERVER_ERROR_INTERNAL,                                    \
          std::string((MSG) + ": " + cudaGetErrorString(err__)).c_str()); \
    }                                                                     \
  } while (false)
#endif  // TRITON_ENABLE_GPU
}  // namespace

TRITONSERVER_Error*
CopyBuffer(
    const std::string& msg, const TRITONSERVER_MemoryType src_memory_type,
    const int64_t src_memory_type_id,
    const TRITONSERVER_MemoryType dst_memory_type,
    const int64_t dst_memory_type_id, const size_t byte_size, const void* src,
    void* dst, cudaStream_t cuda_stream, bool* cuda_used)
{
  *cuda_used = false;

  // For CUDA memcpy, all host to host copy will be blocked in respect to the
  // host, so use memcpy() directly. In this case, need to be careful on whether
  // the src buffer is valid.
  if ((src_memory_type != TRITONSERVER_MEMORY_GPU) &&
      (dst_memory_type != TRITONSERVER_MEMORY_GPU)) {
    memcpy(dst, src, byte_size);
  } else {
#ifdef TRITON_ENABLE_GPU
    // [TODO] use cudaMemcpyDefault if UVM is supported for the device
    auto copy_kind = cudaMemcpyDeviceToDevice;
    if (src_memory_type != TRITONSERVER_MEMORY_GPU) {
      copy_kind = cudaMemcpyHostToDevice;
    } else if (dst_memory_type != TRITONSERVER_MEMORY_GPU) {
      copy_kind = cudaMemcpyDeviceToHost;
    }

    if ((src_memory_type_id != dst_memory_type_id) &&
        (copy_kind == cudaMemcpyDeviceToDevice)) {
      RETURN_IF_CUDA_ERR(
          cudaMemcpyPeerAsync(
              dst, dst_memory_type_id, src, src_memory_type_id, byte_size,
              cuda_stream),
          msg + ": failed to perform CUDA copy");
    } else {
      RETURN_IF_CUDA_ERR(
          cudaMemcpyAsync(dst, src, byte_size, copy_kind, cuda_stream),
          msg + ": failed to perform CUDA copy");
    }

    *cuda_used = true;
#else
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        std::string(msg + ": try to use CUDA copy while GPU is not supported")
            .c_str());
#endif  // TRITON_ENABLE_GPU
  }

  return nullptr;  // success
}

namespace {
TRITONSERVER_Error*
GetDirectoryContents(const std::string& path, std::set<std::string>* contents)
{
  DIR* dir = opendir(path.c_str());
  if (dir == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("failed to open directory ") + path).c_str());
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string entryname = entry->d_name;
    if ((entryname != ".") && (entryname != "..")) {
      contents->insert(entryname);
    }
  }

  closedir(dir);

  return nullptr;  // success
}

TRITONSERVER_Error*
IsDirectory(const std::string& path, bool* is_dir)
{
  *is_dir = false;

  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("failed to stat file ") + path).c_str());
  }

  *is_dir = S_ISDIR(st.st_mode);
  return nullptr;  // success
}

std::string
JoinPath(std::initializer_list<std::string> segments)
{
  std::string joined;

  for (const auto& seg : segments) {
    if (joined.empty()) {
      joined = seg;
    } else if (!seg.empty() && (seg[0] == '/')) {  // IsAbsolutePath(seg)
      if (joined[joined.size() - 1] == '/') {
        joined.append(seg.substr(1));
      } else {
        joined.append(seg);
      }
    } else {  // !IsAbsolutePath(seg)
      if (joined[joined.size() - 1] != '/') {
        joined.append("/");
      }
      joined.append(seg);
    }
  }

  return joined;
}

}  // namespace

TRITONSERVER_Error*
ModelPaths(
    const char* model_repository_path, uint64_t version,
    const bool ignore_directories, const bool ignore_files,
    std::unordered_map<std::string, std::string>* model_paths)
{
  std::set<std::string> model_files;
  // Read all the files in 'path' and filter by type for different requirements
  auto path = JoinPath({model_repository_path, std::to_string(version)});
  RETURN_IF_ERROR(GetDirectoryContents(path, &model_files));
  if (ignore_directories) {
    // Erase directory entries...
    for (auto iter = model_files.begin(); iter != model_files.end();) {
      bool is_dir;
      RETURN_IF_ERROR(IsDirectory(JoinPath({path, *iter}), &is_dir));
      if (is_dir) {
        iter = model_files.erase(iter);
      } else {
        ++iter;
      }
    }
  }
  if (ignore_files) {
    // Erase non-directory entries...
    for (auto iter = model_files.begin(); iter != model_files.end();) {
      bool is_dir;
      RETURN_IF_ERROR(IsDirectory(JoinPath({path, *iter}), &is_dir));
      if (!is_dir) {
        iter = model_files.erase(iter);
      } else {
        ++iter;
      }
    }
  }

  for (const auto& filename : model_files) {
    const auto model_path = JoinPath({path, filename});
    model_paths->emplace(
        std::piecewise_construct, std::make_tuple(filename),
        std::make_tuple(model_path));
  }
  return nullptr;  // success
}


}}}  // namespace nvidia::inferenceserver::backend
