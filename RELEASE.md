<!--
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Release Notes for 2.38.0

## New Freatures and Improvements

* Triton now has Python bindings for the C API. Please refer to 
  [this PR](https://github.com/triton-inference-server/core/pull/265) for 
  usage.

* Triton now forwards request parameters to each of the composing models of an 
  ensemble model.

* The Filesystem API now supports named temporary cache directories when 
  downloading models using the repository agent.

* Added the number of requests currently in the queue to the metrics API. 
  Documentation can be found 
  [here](https://github.com/triton-inference-server/server/blob/r23.09/docs/user_guide/metrics.md#pending-request-count-queue-size-per-model).

* Python backend models can now respond with error codes in addition to error 
  messages.

* TensorRT backend now supports 
  [TensortRT version compatibility](https://github.com/triton-inference-server/tensorrt_backend/tree/r23.09#command-line-options) 
  across models generated with the same major version of TensorRT. Use the 
  `--backend-config=tensorrt,--version-compatible=true` flag to enable this 
  feature. 

* Triton’s backend API now supports accessing the inference response outputs by 
  name or by index. See the new API 
  [here](https://github.com/triton-inference-server/core/blob/r23.09/include/triton/core/tritonbackend.h#L1572-L1608).

* The Python backend now supports loading 
  [Pytorch models directly](https://github.com/triton-inference-server/python_backend/tree/r23.08#pytorch-platform-experimental). 
  This feature is experimental and should be treated as Beta.

* Fixed an issue where if the user didn't call `SetResponseReleaseCallback`, 
  canceling a new request could cancel the old response factory as well. Now 
  when canceling a request which is being re-used, a new response factory is 
  created for each inference.

* Refer to the 23.09 column of the 
  [Frameworks Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) 
  for container image versions on which the 23.09 inference server container is 
  based.

## Known Issues

* When using decoupled models, there is a possibility that response order as 
  sent from the backend may not match with the order in which these responses 
  are received by the streaming gRPC client. Note that this only applies to 
  responses from different requests. Any responses corresponding to the same 
  request will still be received in their expected order, relative to each 
  other.

* The FasterTransformer backend is only officially supported for 22.12, though 
  it can be built for Triton container versions up to 23.07. 

* The Java CAPI is known to have intermittent segfaults we’re looking for a 
  root cause.

* Some systems which implement `malloc()` may not release memory back to the 
  operating system right away causing a false memory leak. This can be mitigated 
  by using a different malloc implementation. Tcmalloc and jemalloc are 
  installed in the Triton container and can be 
  [used by specifying the library in LD_PRELOAD](https://github.com/triton-inference-server/server/blob/r22.12/docs/user_guide/model_management.md). 
  We recommend experimenting with both `tcmalloc` and `jemalloc` to determine 
  which one works better for your use case.

* Auto-complete may cause an increase in server start time. To avoid a start 
  time increase, users can provide the full model configuration and launch the 
  server with `--disable-auto-complete-config`.

* Auto-complete does not support PyTorch models due to lack of metadata in the 
  model. It can only verify that the number of inputs and the input names 
  matches what is specified in the model configuration. There is no model 
  metadata about the number of outputs and datatypes. Related PyTorch bug: 
  https://github.com/pytorch/pytorch/issues/38273

* Triton Client PIP wheels for ARM SBSA are not available from PyPI and pip 
  will install an incorrect Jetson version of Triton Client library for Arm 
  SBSA. The correct client wheel file can be pulled directly from the Arm SBSA 
  SDK image and manually installed.

* Traced models in PyTorch seem to create overflows when int8 tensor values are 
  transformed to int32 on the GPU. Refer to 
  https://github.com/pytorch/pytorch/issues/66930 for more information.

* Triton cannot retrieve GPU metrics with MIG-enabled GPU devices (A100 and A30).

* Triton metrics might not work if the host machine is running a separate DCGM 
  agent on bare-metal or in a container.

* When cloud storage (AWS, GCS, AZURE) is used as a model repository and a model 
  has multiple versions, Triton creates an extra local copy of the cloud model’s 
  folder in the temporary directory, which is deleted upon server’s shutdown. 
