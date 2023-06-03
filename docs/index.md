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

::::{grid}
:reverse:
:gutter: 2 1 1 1
:margin: 4 4 1 1

:::{grid-item}
:columns: 4

```{image} ./_static/nvidia-logo-vert-rgb-blk-for-screen.png
:width: 300px
```
:::
:::{grid-item}
:columns: 8
:class: sd-fs-3

NVIDIA Triton Inference Server

:::
::::

Triton Inference Server is an open source inference serving software that streamlines AI inferencing.

  <!-- :::
  :align: center
  [![Getting Started Video](https://img.youtube.com/vi/NQDtfSi5QF4/1.jpg)](https://www.youtube.com/watch?v=NQDtfSi5QF4)
  ::: -->

<div>
<iframe width="560" height="315" src="https://www.youtube.com/embed/NQDtfSi5QF4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

# Triton

Triton enables teams to deploy any AI model from multiple deep learning and machine learning frameworks, including TensorRT, TensorFlow, PyTorch, ONNX, OpenVINO, Python, RAPIDS FIL, and more. Triton supports inference across cloud, data center,edge and embedded devices on NVIDIA GPUs, x86 and ARM CPU, or AWS Inferentia. Triton delivers optimized performance for many query types, including real time, batched, ensembles and audio/video streaming.

Major features include:

- [Supports multiple deep learning
  frameworks](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton)
- [Supports multiple machine learning
  frameworks](https://github.com/triton-inference-server/fil_backend)
- [Concurrent model
  execution](user_guide/architecture.md#concurrent-model-execution)
- [Dynamic batching](user_guide/model_configuration.md#dynamic-batcher)
- [Sequence batching](user_guide/model_configuration.md#sequence-batcher) and 
  [implicit state management](user_guide/architecture.md#implicit-state-management)
  for stateful models
- Provides [Backend API](https://github.com/triton-inference-server/backend) that
  allows adding custom backends and pre/post processing operations
- Model pipelines using
  [Ensembling](user_guide/architecture.md#ensemble-models) or [Business
  Logic Scripting
  (BLS)](https://github.com/triton-inference-server/python_backend#business-logic-scripting)
- [HTTP/REST and GRPC inference
  protocols](customization_guide/inference_protocols.md) based on the community
  developed [KServe
  protocol](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2)
- A [C API](customization_guide/inference_protocols.md#in-process-triton-server-api) and
  [Java API](customization_guide/inference_protocols.md#java-bindings-for-in-process-triton-server-api)
  allow Triton to link directly into your application for edge and other in-process use cases
- [Metrics](user_guide/metrics.md) indicating GPU utilization, server
  throughput, server latency, and more

Join the [Triton and TensorRT community](https://www.nvidia.com/en-us/deep-learning-ai/triton-tensorrt-newsletter/) and stay current on the latest product updates, bug fixes, content, best 
practices, and more. Need enterprise support? NVIDIA global support is available 
for Triton Inference Server with the [NVIDIA AI Enterprise software suite](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/).

See the [Lastest Release Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-23-05.html#rel-23-05) for updates on the newest features and bug fixes.
