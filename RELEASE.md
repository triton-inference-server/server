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

# Release Notes for 2.35.0

## New Freatures and Improvements

* Support for 
  [KIND\_MODEL instance type](https://github.com/triton-inference-server/pytorch_backend/tree/r23.06#model-instance-group-kind) 
  has been extended to the PyTorch backend.

* The gRPC clients can now indicate whether they want to receive the flags 
  associated with each response. This can help the clients to 
  [programmatically determine](https://github.com/triton-inference-server/server/blob/r23.06/docs/user_guide/decoupled_models.md#knowing-when-a-decoupled-inference-request-is-complete) 
  when all the responses for a given request have been received on the client 
  side for decoupled models.

* Added beta support for using
  [Redis](https://github.com/triton-inference-server/redis_cache/tree/r23.06) as 
  a cache for inference requests.

* The 
  [statistics extension](https://github.com/triton-inference-server/server/blob/r23.06/docs/protocol/extension_statistics.md) 
  now includes the memory usage of the loaded models This statistics is 
  currently implemented only for TensorRT and ONNXRuntime backends.

* Added support for batch inputs in ragged batching for PyTorch backend.

* Added 
  [serial sequences](https://github.com/triton-inference-server/client/blob/main/src/c%2B%2B/perf_analyzer/docs/cli.md#--serial-sequences) 
  mode for Perf Analyzer.

* Refer to the 23.06 column of the 
  [Frameworks Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) 
  for container image versions on which the 23.06 inference server container is 
  based.

## Known Issues

* The Fastertransfer backend build only works with Triton 23.04 and older
  releases.

* OpenVINO 2022.1 is used in the OpenVINO backend and the OpenVINO execution
  provider for the Onnxruntime Backend. OpenVINO 2022.1 is not officially 
  supported on Ubuntu 22.04 and should be treated as beta.

* Some systems which implement `malloc()` may not release memory back to the
  operating system right away causing a false memory leak. This can be mitigate 
  by using a different malloc implementation. `tcmalloc` and `jemalloc` are 
  installed in the Triton container and can be used by specifying the library in 
  LD_PRELOAD.
  
  We recommend experimenting with both `tcmalloc` and `jemalloc` to determine which 
  one works better for your use case.

* Auto-complete may cause an increase in server start time. To avoid a start
  time increase, users can provide the full model configuration and launch the 
  server with `--disable-auto-complete-config`.

* Auto-complete does not support PyTorch models due to lack of metadata in the
  model. It can only verify that the number of inputs and the input names 
  matches what is specified in the model configuration. There is no model 
  metadata about the number of outputs and datatypes. Related PyTorch bug: 
  https://github.com/pytorch/pytorch/issues/38273

* Triton Client PIP wheels for ARM SBSA are not available from PyPI and pip will
  install an incorrect Jetson version of Triton Client library for Arm SBSA. The 
  correct client wheel file can be pulled directly from the Arm SBSA SDK image 
  and manually installed.

* Traced models in PyTorch seem to create overflows when int8 tensor values are 
  transformed to int32 on the GPU. Refer to 
  https://github.com/pytorch/pytorch/issues/66930 for more information.

* Triton cannot retrieve GPU metrics with MIG-enabled GPU devices (A100 and
  A30).

* Triton metrics might not work if the host machine is running a separate DCGM
  agent on bare-metal or in a container.


