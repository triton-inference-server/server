<!--
# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# NVIDIA Triton Inference Server

Triton Inference Server is an open source inference serving software that streamlines
AI inferencing. Triton Inference Server enables teams to deploy any AI model from multiple deep
learning and machine learning frameworks, including TensorRT, TensorFlow,
PyTorch, ONNX, OpenVINO, Python, RAPIDS FIL, and more. Triton supports inference
across cloud, data center, edge and embedded devices on NVIDIA GPUs, x86 and ARM
CPU, or AWS Inferentia. Triton Inference Server delivers optimized performance
for many query types, including real time, batched, ensembles and audio/video
streaming. Triton inference Server is part of
[NVIDIA AI Enterprise](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/),
a software platform that accelerates the data science pipeline and streamlines
the development and deployment of production AI.

  <!-- :::
  :align: center
  [![Getting Started Video](https://img.youtube.com/vi/NQDtfSi5QF4/1.jpg)](https://www.youtube.com/watch?v=NQDtfSi5QF4)
  ::: -->

<div>
<iframe width="560" height="315" src="https://www.youtube.com/embed/NQDtfSi5QF4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>



## Triton Architecture

The following figure shows the Triton Inference Server high-level
architecture. The [model repository](../user_guide/model_repository.md) is a
file-system based repository of the models that Triton will make
available for inferencing. Inference requests arrive at the server via
either [HTTP/REST or GRPC](../customization_guide/inference_protocols.md) or by the [C
API](../customization_guide/inprocess_c_api.md) and are then routed to the appropriate per-model
scheduler. Triton implements [multiple scheduling and batching
algorithms](#models-and-schedulers) that can be configured on a
model-by-model basis. Each model's scheduler optionally performs
batching of inference requests and then passes the requests to the
[backend](https://github.com/triton-inference-server/backend/blob/main/README.md)
corresponding to the model type. The backend performs inferencing
using the inputs provided in the batched requests to produce the
requested outputs. The outputs are then returned.

Triton supports a [backend C
API](https://github.com/triton-inference-server/backend/blob/main/README.md#triton-backend-api)
that allows Triton to be extended with new functionality such as
custom pre- and post-processing operations or even a new deep-learning
framework.

The models being served by Triton can be queried and controlled by a
dedicated [model management API](../user_guide/model_management.md) that is
available by HTTP/REST or GRPC protocol, or by the C API.

Readiness and liveness health endpoints and utilization, throughput
and latency metrics ease the integration of Triton into deployment
framework such as Kubernetes.

![Triton Architecture Diagram](../user_guide/images/arch.jpg)

## Triton major features

Major features include:

- [Supports multiple deep learning
  frameworks](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton)
- [Supports multiple machine learning
  frameworks](https://github.com/triton-inference-server/fil_backend)
- [Concurrent model
  execution](../user_guide/model_execution.md#concurrent-model-execution)
- [Dynamic batching](../user_guide/batcher.md#dynamic-batcher)
- [Sequence batching](../user_guide/batcher.md#sequence-batcher) and
  [implicit state management](../user_guide/implicit_state_management.md#implicit-state-management)
  for stateful models
- Provides [Backend API](https://github.com/triton-inference-server/backend) that
  allows adding custom backends and pre/post processing operations
- Model pipelines using
  [Ensembling](../user_guide/ensemble_models.md#ensemble-models) or [Business
  Logic Scripting
  (BLS)](../user_guide/bls.md#business-logic-scripting)
- [HTTP/REST and GRPC inference
  protocols](../customization_guide/inference_protocols.md) based on the community
  developed [KServe
  protocol](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2)
- A [C API](../customization_guide/inprocess_c_api.md) and
  [Java API](../customization_guide/inprocess_java_api.md)
  allow Triton to link directly into your application for edge and other in-process use cases
- [Metrics](../user_guide/metrics.md) indicating GPU utilization, server
  throughput, server latency, and more

Join the [Triton and TensorRT community](https://www.nvidia.com/en-us/deep-learning-ai/triton-tensorrt-newsletter/) and stay current on the latest product updates, bug fixes, content, best
practices, and more. Need enterprise support? NVIDIA global support is available
for Triton Inference Server with the [NVIDIA AI Enterprise software suite](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/).

See the [Latest Release Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/) for updates on the newest features and bug fixes.