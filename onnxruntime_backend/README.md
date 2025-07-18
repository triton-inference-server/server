<!--
# Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# ONNX Runtime Backend

The Triton backend for the [ONNX
Runtime](https://github.com/microsoft/onnxruntime). You can learn more
about Triton backends in the [backend
repo](https://github.com/triton-inference-server/backend). Ask
questions or report problems on the [issues
page](https://github.com/triton-inference-server/onnxruntime_backend/issues).

Use a recent cmake to build and install in a local directory.
Typically you will want to build an appropriate ONNX Runtime
implementation as part of the build. You do this by specifying a ONNX
Runtime version and a Triton container version that you want to use
with the backend. You can find the combination of versions used in a
particular Triton release in the TRITON_VERSION_MAP at the top of
build.py in the branch matching the Triton release you are interested
in. For example, to build the ONNX Runtime backend for Triton 23.04,
use the versions from TRITON_VERSION_MAP in the [r23.04 branch of
build.py](https://github.com/triton-inference-server/server/blob/r23.04/build.py#L73).

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_BUILD_ONNXRUNTIME_VERSION=1.14.1 -DTRITON_BUILD_CONTAINER_VERSION=23.04 ..
$ make install
```

The resulting install/backends/onnxruntime directory can be added to a
Triton installation as /opt/tritonserver/backends/onnxruntime.

The following required Triton repositories will be pulled and used in
the build. By default the "main" branch/tag will be used for each repo
but the listed CMake argument can be used to override.

* triton-inference-server/backend: -DTRITON_BACKEND_REPO_TAG=[tag]
* triton-inference-server/core: -DTRITON_CORE_REPO_TAG=[tag]
* triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=[tag]

You can add TensorRT support to the ONNX Runtime backend by using
-DTRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON. You can add OpenVino support
by using -DTRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON
-DTRITON_BUILD_ONNXRUNTIME_OPENVINO_VERSION=\<version\>, where
\<version\> is the OpenVino version to use and should match the
TRITON_VERSION_MAP entry as described above. So, to build with both
TensorRT and OpenVino support:

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_BUILD_ONNXRUNTIME_VERSION=1.14.1 -DTRITON_BUILD_CONTAINER_VERSION=23.04 -DTRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON -DTRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON -DTRITON_BUILD_ONNXRUNTIME_OPENVINO_VERSION=2021.2.200 ..
$ make install
```


## ONNX Runtime with TensorRT optimization
TensorRT can be used in conjunction with an ONNX model to further optimize the
performance. To enable TensorRT optimization you must set the model configuration
appropriately. There are several optimizations available for TensorRT, like
selection of the compute precision and workspace size. The optimization
parameters and their description are as follows.


* `precision_mode`: The precision used for optimization. Allowed values are "FP32", "FP16" and "INT8". Default value is "FP32".
* `max_workspace_size_bytes`: The maximum GPU memory the model can use temporarily during execution. Default value is 1GB.
* `int8_calibration_table_name`: Specify INT8 calibration table name. Applicable when precision_mode=="INT8" and the models do not contain Q/DQ nodes. If calibration table is provided for model with Q/DQ nodes then ORT session creation will fail.
* `int8_use_native_calibration_table`: Calibration table to use. Allowed values are 1 (use native TensorRT generated calibration table) and 0 (use ORT generated calibration table). Default is 0. **Note: Latest calibration table file needs to be copied to trt_engine_cache_path before inference. Calibration table is specific to models and calibration data sets. Whenever new calibration table is generated, old file in the path should be cleaned up or be replaced.
* `trt_engine_cache_enable`: Enable engine caching.
* `trt_engine_cache_path`: Specify engine cache path.

To explore the usage of more parameters, follow the mapping table below and
check [ONNX Runtime doc](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#execution-provider-options) for detail.

> Please link to the latest ONNX Runtime binaries in CMake or build from
[main branch of ONNX Runtime](https://github.com/microsoft/onnxruntime/tree/main) to enable latest options.

### Parameter mapping between ONNX Runtime and Triton ONNXRuntime Backend

| Key in Triton model configuration | Value in Triton model config                        | Corresponding TensorRT EP option in ONNX Runtime | Type   |
| --------------------------------- | --------------------------------------------------- | :----------------------------------------------- | :----- |
| max_workspace_size_bytes          | e.g: "4294967296"                                   | trt_max_workspace_size                           | int    |
| trt_max_partition_iterations      | e.g: "1000"                                         | trt_max_partition_iterations                     | int    |
| trt_min_subgraph_size             | e.g: "1"                                            | trt_min_subgraph_size                            | int    |
| precision_mode                    | "FP16"                                              | trt_fp16_enable                                  | bool   |
| precision_mode                    | "INT8"                                              | trt_int8_enable                                  | bool   |
| int8_calibration_table_name       |                                                     | trt_int8_calibration_table_name                  | string |
| int8_use_native_calibration_table | e.g: "1" or "true", "0" or "false"                  | trt_int8_use_native_calibration_table            | bool   |
| trt_dla_enable                    |                                                     | trt_dla_enable                                   | bool   |
| trt_dla_core                      | e.g: "0"                                            | trt_dla_core                                     | int    |
| trt_engine_cache_enable           | e.g: "1" or "true", "0" or "false"                  | trt_engine_cache_enable                          | bool   |
| trt_engine_cache_path             |                                                     | trt_engine_cache_path                            | string |
| trt_engine_cache_prefix           |                                                     | trt_engine_cache_prefix                          | string |
| trt_dump_subgraphs                | e.g: "1" or "true", "0" or "false"                  | trt_dump_subgraphs                               | bool   |
| trt_force_sequential_engine_build | e.g: "1" or "true", "0" or "false"                  | trt_force_sequential_engine_build                | bool   |
| trt_context_memory_sharing_enable | e.g: "1" or "true", "0" or "false"                  | trt_context_memory_sharing_enable                | bool   |
| trt_layer_norm_fp32_fallback      | e.g: "1" or "true", "0" or "false"                  | trt_layer_norm_fp32_fallback                     | bool   |
| trt_timing_cache_enable           | e.g: "1" or "true", "0" or "false"                  | trt_timing_cache_enable                          | bool   |
| trt_timing_cache_path             |                                                     | trt_timing_cache_path                            | string |
| trt_force_timing_cache            | e.g: "1" or "true", "0" or "false"                  | trt_force_timing_cache                           | bool   |
| trt_detailed_build_log            | e.g: "1" or "true", "0" or "false"                  | trt_detailed_build_log                           | bool   |
| trt_build_heuristics_enable       | e.g: "1" or "true", "0" or "false"                  | trt_build_heuristics_enable                      | bool   |
| trt_sparsity_enable               | e.g: "1" or "true", "0" or "false"                  | trt_sparsity_enable                              | bool   |
| trt_builder_optimization_level    | e.g: "3"                                            | trt_builder_optimization_level                   | int    |
| trt_auxiliary_streams             | e.g: "-1"                                           | trt_auxiliary_streams                            | int    |
| trt_tactic_sources                | e.g: "-CUDNN,+CUBLAS";                              | trt_tactic_sources                               | string |
| trt_extra_plugin_lib_paths        |                                                     | trt_extra_plugin_lib_paths                       | string |
| trt_profile_min_shapes            | e.g: "input1:dim1xdimd2...,input2:dim1xdim2...,..." | trt_profile_min_shapes                           | string |
| trt_profile_max_shapes            | e.g: "input1:dim1xdimd2...,input2:dim1xdim2...,..." | trt_profile_max_shapes                           | string |
| trt_profile_opt_shapes            | e.g: "input1:dim1xdimd2...,input2:dim1xdim2...,..." | trt_profile_opt_shapes                           | string |
| trt_cuda_graph_enable             | e.g: "1" or "true", "0" or "false"                  | trt_cuda_graph_enable                            | bool   |
| trt_dump_ep_context_model         | e.g: "1" or "true", "0" or "false"                  | trt_dump_ep_context_model                        | bool   |
| trt_ep_context_file_path          |                                                     | trt_ep_context_file_path                         | string |
| trt_ep_context_embed_mode         | e.g: "1"                                            | trt_ep_context_embed_mode                        | int    |

The section of model config file specifying these parameters will look like:

```
.
.
.
optimization { execution_accelerators {
  gpu_execution_accelerator : [ {
    name : "tensorrt"
    parameters { key: "precision_mode" value: "FP16" }
    parameters { key: "max_workspace_size_bytes" value: "1073741824" }}
    parameters { key: "trt_engine_cache_enable" value: "1" }}
  ]
}}
.
.
.
```

## ONNX Runtime with CUDA Execution Provider optimization
When GPU is enabled for ORT, CUDA execution provider is enabled. If TensorRT is
also enabled then CUDA EP is treated as a fallback option (only comes into
picture for nodes which TensorRT cannot execute). If TensorRT is not enabled
then CUDA EP is the primary EP which executes the models. ORT enabled
configuring options for CUDA EP to further optimize based on the specific model
and user scenarios. There are several optimizations available, please refer to
the [ONNX Runtime doc](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#cuda-execution-provider)
for more details. To enable CUDA EP optimization you must set the model
configuration appropriately:

```
optimization { execution_accelerators {
  gpu_execution_accelerator : [ {
    name : "cuda"
    parameters { key: "cudnn_conv_use_max_workspace" value: "0" }
    parameters { key: "use_ep_level_unified_stream" value: "1" }}
  ]
}}
```

### Deprecated Parameters
The way to specify these specific parameters as shown below is deprecated. For
backward compatibility, these parameters are still supported. Please use the
above method to specify the parameters.

* `cudnn_conv_algo_search`: CUDA Convolution algorithm search configuration.
Available options are 0 - EXHAUSTIVE (expensive exhaustive benchmarking using
cudnnFindConvolutionForwardAlgorithmEx). This is also the default option,
1 - HEURISTIC (lightweight heuristic based search using
cudnnGetConvolutionForwardAlgorithm_v7), 2 - DEFAULT (default algorithm using
CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)

* `gpu_mem_limit`: CUDA memory limit. To use all possible memory pass in maximum
size_t. Defaults to SIZE_MAX.

* `arena_extend_strategy`: Strategy used to grow the memory arena. Available
options are: 0 = kNextPowerOfTwo, 1 = kSameAsRequested. Defaults to 0.

* `do_copy_in_default_stream`: Flag indicating if copying needs to take place on
the same stream as the compute stream in the CUDA EP. Available options are:
0 = Use separate streams for copying and compute, 1 = Use the same stream for
copying and compute. Defaults to 1.

In the model config file, specifying these parameters will look like:

```
.
.
.
parameters { key: "cudnn_conv_algo_search" value: { string_value: "0" } }
parameters { key: "gpu_mem_limit" value: { string_value: "4294967200" } }
.
.
.

```


## ONNX Runtime with OpenVINO optimization

[OpenVINO](https://docs.openvinotoolkit.org/latest/index.html) can be
used in conjunction with an ONNX model to further optimize
performance. To enable OpenVINO optimization you must set the model
configuration as shown below.

```
.
.
.
optimization { execution_accelerators {
  cpu_execution_accelerator : [ {
    name : "openvino"
  } ]
}}
.
.
.
```

## Other Optimization Options with ONNX Runtime

Details regarding when to use these options and what to expect from them can be
found [here](https://onnxruntime.ai/docs/performance/tune-performance.html)

### Model Config Options
* `intra_op_thread_count`: Sets the number of threads used to parallelize the
execution within nodes. A value of 0 means ORT will pick a default which is
number of cores.
* `inter_op_thread_count`: Sets the number of threads used to parallelize the
execution of the graph (across nodes). If sequential execution is enabled this
value is ignored.
A value of 0 means ORT will pick a default which is number of cores.
* `execution_mode`: Controls whether operators in the graph are executed
sequentially or in parallel. Usually when the model has many branches, setting
this option to 1 .i.e. "parallel" will give you better performance. Default is
0 which is "sequential execution."
* `level`: Refers to the graph optimization level. By default all optimizations
are enabled. Allowed values are -1, 1 and 2. -1 refers to BASIC optimizations,
1 refers to basic plus extended optimizations like fusions and 2 refers to all
optimizations being disabled. Please find the details
[here](https://onnxruntime.ai/docs/performance/graph-optimizations.html).

```
optimization {
  graph : {
    level : 1
}}

parameters { key: "intra_op_thread_count" value: { string_value: "0" } }
parameters { key: "execution_mode" value: { string_value: "0" } }
parameters { key: "inter_op_thread_count" value: { string_value: "0" } }

```
* `enable_mem_arena`: Use 1 to enable the arena and 0 to disable. See
[this](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#a0bbd62df2b3c119636fba89192240593)
for more information.
* `enable_mem_pattern`: Use 1 to enable memory pattern and 0 to disable.
See [this](https://onnxruntime.ai/docs/api/c/struct_ort_api.html#ad13b711736956bf0565fea0f8d7a5d75)
for more information.
* `memory.enable_memory_arena_shrinkage`:
See [this](https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h)
for more information.
* `session.use_device_allocator_for_initializers`: Use "1" to enable using device allocator for allocating initialized tensor memory and "0" to disable. The default is "0". See [this](https://onnxruntime.ai/docs/get-started/with-c.html) for more information.

### Command line options

#### Thread Pools

When intra and inter op threads is set to 0 or a value higher than 1, by default
ORT creates threadpool per session. This may not be ideal in every scenario,
therefore ORT also supports global threadpools. When global threadpools are
enabled ORT creates 1 global threadpool which is shared by every session.
Use the backend config to enable global threadpool. When global threadpool is
enabled, intra and inter op num threads config should also be provided via
backend config. Config values provided in model config will be ignored.

```
--backend-config=onnxruntime,enable-global-threadpool=<0,1>, --backend-config=onnxruntime,intra_op_thread_count=<int> , --backend-config=onnxruntime,inter_op_thread_count=<int>
```

#### Default Max Batch Size

The default-max-batch-size value is used for max_batch_size during
[Autocomplete](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#auto-generated-model-configuration)
when no other value is found. Assuming server was not launched with
`--disable-auto-complete-config` command-line option, the onnxruntime backend
will set the max_batch_size of the model to this default value under the
following conditions:

1. Autocomplete has determined the model is capable of batching requests.
2. max_batch_size is 0 in the model configuration or max_batch_size
   is omitted from the model configuration.

If max_batch_size > 1 and no
[scheduler](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#scheduling-and-batching)
is provided, the dynamic batch scheduler will be used.

```
--backend-config=onnxruntime,default-max-batch-size=<int>
```

The default value of `default-max-batch-size` is 4.
