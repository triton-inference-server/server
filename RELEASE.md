<!--
# Copyright 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Release Notes for 2.25.0

## New Freatures and Improvements

* New 
  [support for multiple cloud credentials](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md#cloud-storage-with-credential-file-beta) 
  has been enabled. This feature is in beta and is subject to change.

* Models using custom backends which implement 
  [auto-complete configuration](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#auto-generated-model-configuration), 
  can be loaded without explicit config.pbtxt file if they are named in form 
  `<model_name>.<backend_name>`.

* Users can specify a maximum memory limit when loading models onto the GPU  
  with the new 
  [--model-load-gpu-limit](https://github.com/triton-inference-server/server/blob/b3d7a3375e7adb1341724c0ac34661b4cde23cd2/src/main.cc#L629-L635)
  tritonserver option and the 
  [TRITONSERVER_ServerOptionsSetModelLoadDeviceLimit](https://github.com/triton-inference-server/core/blob/c9cd6630ecb04bb26e2110cd65a37f23aec8153b/include/triton/core/tritonserver.h#L1861-L1872) C API function

* Added new documentation, 
  [Performance Tuning](https://github.com/triton-inference-server/server/blob/main/docs/performance_tuning.md), with a step by step guide to optimize models for 
  production

* From this release onwards Triton will default to 
  [TensorFlow version 2.X.](https://github.com/triton-inference-server/tensorflow_backend/tree/main#--backend-configtensorflowversionint) 
  TensorFlow version 1.X can still be manually specified via backend config.

* PyTorch backend has improved performance by using a separate CUDA Stream for 
  each model instance when the instance kind is GPU.

* Refer to the 22.08 column of the 
  [Frameworks Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) 
  for container image versions on which the 22.08 inference server container is 
  based.

* Model Analyzer's profile subcommand now analyzes the results after Profile is 
  completed. Usage of the Analyze subcommand is deprecated. See 
  [Model Analyzer's documentation](https://github.com/triton-inference-server/model_analyzer/blob/main/docs/cli.md#subcommand-profile) 
  for further details.

## Known Issues

* There is no Jetpack release for 22.08, the latest release is 22.07.

* Auto-complete may cause an increase in server start time. To avoid a start 
  time increase, users can provide the full model configuration and launch the 
  server with `--disable-auto-complete-config`.

* When auto-completing some model configs, backends may generate a model config 
  even though there is not enough metadata (ex. Graphdef models for TensorFlow 
  Backend). The user will see the model successfully load but fail to inference. 
  In this case the user should provide the full model configuration for these 
  models or use the `--disable-auto-complete-config` CLI option to show which 
  models fail to load.

* Auto-complete does not support PyTorch models due to lack of metadata in the 
  model. It can only verify that the number of inputs and the input names 
  matches what is specified in the model configuration. There is no model 
  metadata about the number of outputs and datatypes. Related PyTorch bug: 
  https://github.com/pytorch/pytorch/issues/38273

* Auto-complete is not supported in the OpenVINO backend

* Perf Analyzer stability criteria has been changed which may result in 
  reporting instability for scenarios that were previously considered stable. 
  This change has been made to improve the accuracy of Perf Analyzer results. 
  If you observe this message, it can be resolved by increasing the 
  `--measurement-interval` in the time windows mode or 
  `--measurement-request-count` in the count windows mode.

* Triton Client PIP wheels for ARM SBSA are not available from PyPI and pip will 
  install an incorrect Jetson version of Triton Client library for Arm SBSA.
  
  The correct client wheel file can be pulled directly from the Arm SBSA SDK 
  image and manually installed.

* Traced models in PyTorch seem to create overflows when int8 tensor values are 
  transformed to int32 on the GPU. 

  Refer to https://github.com/pytorch/pytorch/issues/66930 for more information.

* Triton cannot retrieve GPU metrics with MIG-enabled GPU devices (A100 and A30).

* Triton metrics might not work if the host machine is running a separate DCGM 
  agent on bare-metal or in a container.

* Model Analyzer reported values for GPU utilization and GPU power are known to 
  be inaccurate and generally lower than reality.
