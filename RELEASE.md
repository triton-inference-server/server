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

# Release Notes for 2.37.0

## New Freatures and Improvements

* Triton can load model instances in parallel for supporting backends. See [TRITONBACKEND_BackendAttributeSetParallelModelInstanceLoading](https://github.com/triton-inference-server/backend/tree/r23.08#tritonbackend_backendattribute) for more details. As of 23.08, only [python](https://github.com/triton-inference-server/python_backend/tree/r23.08) and [onnxruntime](https://github.com/triton-inference-server/onnxruntime_backend/tree/r23.08) backends support loading model instances in parallel.

* Python backend models can capture [trace for composing child](https://github.com/triton-inference-server/server/blob/r23.08/docs/user_guide/trace.md#tracing-for-bls-models) models when executing BLS requests.

* Triton OpenTelemetry Tracing exposes [resource settings](https://github.com/triton-inference-server/server/blob/r23.08/docs/user_guide/trace.md#opentelemetry-trace-apis-settings) which can be used to configure the service name and version.

* Python backend supports directly [loading and serving PyTorch models](https://github.com/triton-inference-server/python_backend/tree/r23.08#pytorch-platform-experimental) with torch.compile().

* Exposed [preserve_ordering](https://github.com/triton-inference-server/common/blob/r23.08/protobuf/model_config.proto#L1461-L1481) field to oldest strategy sequence batcher. The default behavior of the oldest strategy sequence batcher to preserve response order across the independent requests belonging to different sequences is changed from True to False. Note: This setting does not impact order of responses within a sequence.

* Refer to the 23.08 column of the
  [Frameworks Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)
  for container image versions on which the 23.08 inference server container is
  based.

## Known Issues

* Triton uses OpenTelemetry CPP library version, which can cause Triton to [crash](https://github.com/triton-inference-server/server/issues/6202), when OpenTelemetry’s exporter timeouts.

* When using decoupled models, there is a possibility that response order as sent from the backend may not match with the order in which these responses are received by the streaming gRPC client.

* The
  ["fastertransformer_backend"](https://github.com/triton-inference-server/fastertransformer_backend) is only officially supported for 22.12, though it can be built for Triton container versions up to 23.07.

* The Java CAPI is known to have intermittent segfaults we’re looking for a root cause.

* Some systems which implement `malloc()` may not release memory back to the
  operating system right away causing a false memory leak. This can be mitigate
  by using a different malloc implementation. `tcmalloc` and `jemalloc` are
  installed in the Triton container and can be
  [used by specifying the library in LD_PRELOAD](https://github.com/triton-inference-server/server/blob/r22.12/docs/user_guide/model_management.md).

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