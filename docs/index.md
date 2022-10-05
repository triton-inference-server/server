---
title: Triton Inference Server
---


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

Join the Triton and TensorRT community and stay current on the latest product updates, bug fixes, content, best practices, and more. Need enterprise support? NVIDIA global support is available for Triton Inference Server with the NVIDIA AI Enterprise software suite.
