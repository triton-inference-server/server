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

# Release Notes for 2.22.0

## New Freatures and Improvements

* Triton In-Process API is available in 
  [Java](https://github.com/triton-inference-server/server/blob/r22.05/docs/inference_protocols.md#java-bindings-for-in-process-triton-server-api).

* Python backend supports the 
  [decoupled API](https://github.com/triton-inference-server/python_backend/tree/r22.05#decoupled-mode-beta) 
  as BETA release.

* You may load models with 
  [file content](https://github.com/triton-inference-server/server/blob/r22.05/docs/protocol/extension_model_repository.md#load) 
  provided during the Triton Server API invocation.

* Triton supports 
  [BF16 data type](https://github.com/triton-inference-server/server/blob/r22.05/docs/model_configuration.md#datatypes).

* PyTorch backend supports 
  [1-dimensional String I/O](https://github.com/triton-inference-server/pytorch_backend/tree/r22.05#important-note).

* Explicit model control mode supports 
  [loading all models at startup](https://github.com/triton-inference-server/server/blob/r22.05/docs/model_management.md#model-control-mode-explicit).

* You may specify 
  [customized GRPC channel settings](https://github.com/triton-inference-server/client/blob/r22.05/src/python/library/tritonclient/grpc/__init__.py#L193-L200) 
  in the GRPC client library.

* Triton In-Process API supports 
  [dynamic model repository registration](https://github.com/triton-inference-server/core/blob/r22.05/include/triton/core/tritonserver.h#L1903-L1923).

* [Improve build pipeline](https://github.com/triton-inference-server/server/blob/r22.05/docs/build.md) 
  in `build.py` and generate build scripts used for pipeline examination.

* ONNX Runtime backend updated to ONNX Runtime version 1.11.1 in both Ubuntu and 
  Windows versions of Triton.

* Refer to the 22.05 column of the 
  [Frameworks Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) 
  for container image versions on which the 22.05 inference server container is 
  based.

## Known Issues

* Triton PIP wheels for ARM SBSA are not available from PyPI and pip will 
  install an incorrect Jetson version of Triton for Arm SBSA. 
  
  The correct wheel file can be pulled directly from the Arm SBSA SDK image and 
  manually installed.

* Traced models in PyTorch seem to create overflows when int8 tensor values are 
  transformed to int32 on the GPU. 
  
  Refer to [pytorch/pytorch#66930](http://pytorch/pytorch#66930) for more 
  information.

* Triton cannot retrieve GPU metrics with MIG-enabled GPU devices (A100 and A30).

* Triton metrics might not work if the host machine is running a separate DCGM 
  agent on bare-metal or in a container.

* Running a PyTorch TorchScript model using the PyTorch backend, where multiple 
  instances of a model are configured can lead to a slowdown in model execution 
  due to the following PyTorch issue: 
  [pytorch/pytorch#27902](http://pytorch/pytorch#27902).

* Starting in 22.02, the Triton container, which uses the 22.05 PyTorch 
  container, will report an error during model loading in the PyTorch backend 
  when using scripted models that were exported in the legacy format (using our 
  19.09 or previous PyTorch NGC containers corresponding to PyTorch 1.2.0 or 
  previous releases). 
  
  To load the model successfully in Triton, you need to export the model again 
  by using a recent version of PyTorch.
