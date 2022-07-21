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

# Release Notes for 2.24.0

## New Freatures and Improvements

* [Auto-Complete](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#auto-generated-model-configuration) 
  is enabled by default. The `--strict-model-config` option has been soft 
  deprecated, use the new  `--disable-auto-complete-config` CLI option instead.

* New example backend demonstrating 
  [Business Logic Scripting in C++](https://github.com/triton-inference-server/backend/blob/r22.07/examples/backends/bls/README.md).

* Users can provide values for 
  ["init_ops"](https://github.com/triton-inference-server/tensorflow_backend/tree/r22.07#parameters) 
  in Tensorflow TF1.x GraphDef models through json file.

* New 
  [asyncio compatible API](https://github.com/triton-inference-server/client#python-asyncio-support-beta) 
  to the Python GRPC/HTTP APIs.

* Added  thread pool to reduce service downtime for concurrently loading models. 
  The thread pool size is configurable with the new `--model-load-thread-count`
  tritonserver option. You can find more information 
  [here](https://github.com/triton-inference-server/server/blob/main/docs/model_management.md#concurrently-loading-models). 

* Model Analyzer now doesn't require `config.pbtxt` file for models that can be 
  auto-completed in Triton.

* Refer to the 22.07 column of the 
  [Frameworks Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) 
  for container image versions on which the 22.07 inference server container is 
  based.

## Known Issues

* JetPack release will be published later in the month in order to align with 
  JetPack SDK public availability.

* Auto-complete could cause an increase in server start time. To avoid a start 
  time increase, users should provide the full model configuration.

* When auto-completing some model configs, backends may generate a model config 
  even though there is not enough metadata (ex. Graphdef models for Tensorflow 
  Backend). The user will see the model successfully load but fail to inference.
  In this case the user should provide the full model configuration for these 
  models or use the `--disable-auto-complete-config` CLI option to show which 
  models fail to load.

* Can't do autocomplete for PyTorch models, not enough metadata. Can only verify
  that the number of inputs is correct and the input names match what is 
  specified in the model configuration. No info about number of outputs and 
  datatypes. Related pytorch bug: 
  https://github.com/pytorch/pytorch/issues/38273.

* Running inference on multiple TensorRT model instances in Triton may fail 
  with signal(6). The issue is expected to be fixed in a future release. Details 
  can be found at https://github.com/triton-inference-server/server/issues/4566.

* Perf Analyzer stability criteria has been changed which may result in 
  reporting instability for scenarios that were previously considered stable. 
  This change has been made to improve the accuracy of Perf Analyzer results. 
  If you observe this message, it can be resolved by increasing the 
  `--measurement-interval` in the time windows mode or 
  `--measurement-request-count` in the count windows mode.

* Unlike previously noted, 22.07 is the last release that defaults to 
  [TensorFlow version 1](https://github.com/triton-inference-server/tensorflow_backend/tree/r22.06#--backend-configtensorflowversionint).
  From 22.08 onwards Triton will change the default TensorFlow 
  version to 2.X.

* Triton PIP wheels for ARM SBSA are not available from PyPI and pip will 
  install an incorrect Jetson version of Triton for Arm SBSA. 
  
  The correct wheel file can be pulled directly from the Arm SBSA SDK image and 
  manually installed.

* Traced models in PyTorch seem to create overflows when int8 tensor values are 
  transformed to int32 on the GPU. 
  
  Refer to issue https://github.com/pytorch/pytorch/issues/66930 for more 
  information.

* Triton cannot retrieve GPU metrics with MIG-enabled GPU devices (A100 and A30).

* Triton metrics might not work if the host machine is running a separate DCGM 
  agent on bare-metal or in a container.

* Starting from 22.02, the Triton container, which uses the 22.02 or above 
  PyTorch container, will report an error during model loading in the PyTorch 
  backend when using scripted models that were exported in the legacy format 
  (using our 19.09 or previous PyTorch NGC containers corresponding to 
  PyTorch 1.2.0 or previous releases). 
  
  To load the model successfully in Triton, you need to export the model again 
  by using a recent version of PyTorch.
