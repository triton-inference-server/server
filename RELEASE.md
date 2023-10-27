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

# Release Notes for 2.39.0

## New Freatures and Improvements

* Added support for handling client-side request cancellation in Triton server and backends. ([server docs](https://github.com/triton-inference-server/server/blob/r23.10/docs/user_guide/request_cancellation.md), [client docs](https://github.com/triton-inference-server/client/tree/r23.10#request-cancellation)).

* Triton can deploy supported models on the vLLM engine using the new [vLLM backend](https://github.com/triton-inference-server/vllm_backend/tree/r23.10). A new container with vLLM backend is available on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags) for 23.10.

* Triton now supports the [TensorRT-LLM backend](https://github.com/triton-inference-server/tensorrtllm_backend/tree/release/0.5.0). This backend uses the [Nvidia TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/release/0.5.0), which replaces the [Fastertransformer backend](https://github.com/triton-inference-server/fastertransformer_backend). A new container with TensorRT-LLM backend is available on [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags) for 23.10.

* Added [Generate](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_generate.md) extension (beta) which provides better REST APIs for inference on Large Language Models.

* New tutorials with respect to how to run vLLM with the new REST API, how to run Llama2 with TensorRT-LLM backend, and how to run with HuggingFace models in the [tutorial repo](https://github.com/triton-inference-server/tutorials).

* Support Scalar I/O in ONNXRuntime backend.

* Added support for writing custom backends in python, a.k.a. [Python-based backends](https://github.com/triton-inference-server/backend/blob/main/docs/python_based_backends.md#python-based-backends).

* Refer to the 23.10 column of the [Frameworks Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) for container image versions on which the 23.10 inference server container is based.

## Known Issues

* For its initial release, the TensorRT-LLM [backend](https://github.com/triton-inference-server/tensorrtllm_backend) provides limited support of Triton extensions and features.

* The TensorRT-LLM backend may core dump on server shutdown. This impacts server teardown only and will not impact inferencing.

* When a model uses a backend which is not found, Triton would reference the missing  backend as `backend_name /model.py” in the error message. This is already fixed for future releases.

* When using decoupled models, there is a possibility that response order as sent from the backend may not match with the order in which these responses are received by the streaming gRPC client. Note that this only applies to responses from different requests. Any responses corresponding to the same request will still be received in their expected order, relative to each other.

* The FasterTransformer backend is only officially supported for 22.12, though it can be built for Triton container versions up to 23.07.

* The Java CAPI is known to have intermittent segfaults we’re looking for a root cause.

* Some systems which implement malloc() may not release memory back to the operating system right away causing a false memory leak. This can be mitigated by using a different malloc implementation. Tcmalloc and jemalloc are installed in the Triton container and can be [used by specifying the library in LD_PRELOAD](https://github.com/triton-inference-server/server/blob/r22.12/docs/user_guide/model_management.md). We recommend experimenting with both tcmalloc and jemalloc to determine which one works better for your use case.

* Auto-complete may cause an increase in server start time. To avoid a start time increase, users can provide the full model configuration and launch the server with `--disable-auto-complete-config`.

* Auto-complete does not support PyTorch models due to lack of metadata in the model. It can only verify that the number of inputs and the input names matches what is specified in the model configuration. There is no model metadata about the number of outputs and datatypes. Related PyTorch bug: https://github.com/pytorch/pytorch/issues/38273

* Triton Client PIP wheels for ARM SBSA are not available from PyPI and pip will install an incorrect Jetson version of Triton Client library for Arm SBSA. The correct client wheel file can be pulled directly from the Arm SBSA SDK image and manually installed.

* Traced models in PyTorch seem to create overflows when int8 tensor values are transformed to int32 on the GPU. Refer to https://github.com/pytorch/pytorch/issues/66930 for more information.

* Triton cannot retrieve GPU metrics with MIG-enabled GPU devices (A100 and A30).

* Triton metrics might not work if the host machine is running a separate DCGM agent on bare-metal or in a container.

* When cloud storage (AWS, GCS, AZURE) is used as a model repository and a model has multiple versions, Triton creates an extra local copy of the cloud model’s folder in the temporary directory, which is deleted upon server’s shutdown.
