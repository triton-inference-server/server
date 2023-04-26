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

# Release Notes for 2.33.0

## New Freatures and Improvements

* Triton can now load models concurrently reducing the server start-up times.

* Sequence batcher with direct scheduling strategy now includes experimental 
  support for 
  [schedule policy](https://github.com/triton-inference-server/server/blob/r23.04/docs/protocol/extension_schedule_policy.md#sequence-batcher-with-direct-scheduling-strategy).

* Triton’s 
  [ragged batching](https://github.com/triton-inference-server/server/blob/r23.04/docs/user_guide/ragged_batching.md) 
  support has been extended to PyTorch backend.

* Triton can now 
  [forward HTTP/GRPC headers as inference request parameters to](https://github.com/triton-inference-server/server/blob/r23.04/docs/protocol/extension_parameters.md#forwarding-httpgrpc-headers-as-parameters) 
  the backend.

* Triton python backend’s 
  [business logic scripting](https://github.com/triton-inference-server/python_backend/tree/r23.04#business-logic-scripting) 
  now allows developers to select a specific device to receive output tensors 
  from a BLS call.

* Triton latency metrics can now be obtained as configurable quantiles over a 
  sliding time window using 
  [experimental metrics summary support](https://github.com/triton-inference-server/server/blob/r23.04/docs/user_guide/metrics.md#summaries).

* Users can now 
  [restrict the access of the protocols](https://github.com/triton-inference-server/server/blob/r23.04/docs/customization_guide/inference_protocols.md#limit-endpoint-access-beta) 
  on a given Triton endpoint.

* Triton now provides a limited support for 
  [tracing inference requests using OpenTelemetry Trace APIs](https://github.com/triton-inference-server/server/blob/r23.04/docs/user_guide/trace.md#opentelemetry-trace-support).

* Model Analyzer now supports 
  [BLS Models](https://github.com/triton-inference-server/model_analyzer/blob/r23.04/docs/config_search.md#bls-model-search).

* Refer to the 23.04 column of the 
  [Frameworks Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) 
  for container image versions on which the 23.04 inference server container is 
  based.

## Known Issues

* Tensorflow backend no longer supports TensorFlow version 1.

* [Triton Inferentia guide](https://github.com/triton-inference-server/python_backend/tree/main/inferentia#inferentia-setup) 
  is out of date. Some users have reported issues with running Triton on AWS 
  Inferentia instances.

* Some systems which implement `malloc()` may not release memory back to the 
  operating system right away causing a false memory leak. This can be mitigated 
  by using a different malloc implementation. Tcmalloc is installed in the 
  Triton container and can be 
  [used by specifying the library in `LD_PRELOAD`](https://github.com/triton-inference-server/server/blob/r22.12/docs/user_guide/model_management.md).

* Auto-complete may cause an increase in server start time. To avoid a start 
  time increase, users can provide the full model configuration and launch the 
  server with `--disable-auto-complete-config`.

* Auto-complete does not support PyTorch models due to lack of metadata in the 
  model. It can only verify that the number of inputs and the input names 
  matches what is specified in the model configuration. There is no model 
  metadata about the number of outputs and datatypes. Related PyTorch bug: 
  https://github.com/pytorch/pytorch/issues/38273.

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
