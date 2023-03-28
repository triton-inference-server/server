<!--
# Copyright 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Release Notes for 2.32.0

## New Freatures and Improvements

* Added the 
  [Parameters Extension](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_parameters.md) 
  which allows an inference request to provide custom parameters that cannot be 
  provided as inputs. These parameters can be used in the python backend as 
  described 
  [here](https://github.com/triton-inference-server/python_backend#inference-request-parameters).

* Added support for models that use decoupled API for Business Scripting Logic 
  (BLS) in Python backend. Examples can be found 
  [here](https://github.com/triton-inference-server/python_backend/blob/main/examples/decoupled/README.md). 

* The same model name can be used across different repositories if the 
  `--model-namespacing` flag is set. 

* Triton’s Response Cache feature has been converted internally to a shared 
  library implementation of the new 
  [TRITONCACHE APIs](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritoncache.h), 
  similar to how backends and repo agents are used today. The default cache 
  implementation is 
  [local_cache](https://github.com/triton-inference-server/local_cache), which 
  is equivalent to the fixed-size in-memory buffer implementation used before. 
  The `--response-cache-byte-size` flag will continue to function in the same 
  way, but the `--cache-config` flag will be the preferred method of cache 
  configuration moving forward. For more information, see the cache 
  documentation 
  [here](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/response_cache.md).

* Triton’s 
  [trace tool](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/trace.md) 
  now supports tracing for `request_id`.

* Refer to the 23.03 column of the 
  [Frameworks Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) 
  for container image versions on which the 23.03 inference server container is 
  based.

## Known Issues

* Support for TensorFlow1 will be removed starting from 23.04.

* Some systems which implement `malloc()` may not release memory back to the 
  operating system right away causing a false memory leak. This can be mitigated 
  by using a different malloc implementation. Tcmalloc is installed in the 
  Triton container and can be 
  [used by specifying the library in LD_PRELOAD](https://github.com/triton-inference-server/server/blob/r23.03/docs/user_guide/model_management.md#model-control-mode-explicit).

* Auto-complete may cause an increase in server start time. To avoid a start 
  time increase, users can provide the full model configuration and launch the 
  server with `--disable-auto-complete-config`.

* Auto-complete does not support PyTorch models due to lack of metadata in the 
  model. It can only verify that the number of inputs and the input names 
  matches what is specified in the model configuration. There is no model 
  metadata about the number of outputs and datatypes. Related PyTorch bug: 
  https://github.com/pytorch/pytorch/issues/38273

* Triton Client PIP wheels for ARM SBSA are not available from PyPI and pip will 
  install an incorrect Jetson version of Triton Client library for Arm SBSA. 

  The correct client wheel file can be pulled directly from the Arm SBSA SDK 
  image and manually installed.

* Traced models in PyTorch seem to create overflows when int8 tensor values are
  transformed to `int32` on the GPU. 

  Refer to https://github.com/pytorch/pytorch/issues/66930 for more information.

* Triton cannot retrieve GPU metrics with MIG-enabled GPU devices (A100 and A30).

* Triton metrics might not work if the host machine is running a separate DCGM 
  agent on bare-metal or in a container.