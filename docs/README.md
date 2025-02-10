<!--
# Copyright 2018-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# **Triton Inference Server Documentation**

| [Installation](README.md#installation) | [Getting Started](README.md#getting-started) | [User Guide](README.md#user-guide) | [API Guide](protocol/README.md) | [Additional Resources](README.md#resources) | [Customization Guide](README.md#customization-guide) |
| ------------ | --------------- | --------------- | ------------ | --------------- | --------------- |

**New to Triton Inference Server?** Make use of
[these tutorials](https://github.com/triton-inference-server/tutorials)
 to begin your Triton journey!

## **Installation**
Before you can use the Triton Docker image you must install
[Docker](https://docs.docker.com/engine/install). If you plan on using
a GPU for inference you must also install the [NVIDIA Container
Toolkit](https://github.com/NVIDIA/nvidia-docker). DGX users should
follow [Preparing to use NVIDIA
Containers](http://docs.nvidia.com/deeplearning/dgx/preparing-containers/index.html).

Pull the image using the following command.

```
$ docker pull nvcr.io/nvidia/tritonserver:<yy.mm>-py3
```

Where \<yy.mm\> is the version of Triton that you want to pull. For a complete list of all the variants and versions of the Triton Inference Server Container,  visit the [NGC Page](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver). More information about customizing the Triton Container can be found in [this section](customization_guide/compose.md) of the User Guide.

## **Getting Started**

This guide covers the simplest possible workflow for deploying a model using a Triton Inference Server.
- [Create a Model Repository](getting_started/quickstart.md#create-a-model-repository)
- [Launch Triton](getting_started/quickstart.md#launch-triton)
- [Send an Inference Request](getting_started/quickstart.md#send-an-inference-request)

Triton Inference Server has a considerable list versatile and powerful features. All new users are recommended to explore the [User Guide](README.md#user-guide) and the [additional resources](README.md#resources) sections for features most relevant to their use case.

## **User Guide**
The User Guide describes how to configure Triton, organize and configure your models, use the C++ and Python clients, etc. This guide includes the following:
* Creating a Model Repository [[Overview](README.md#model-repository) || [Details](user_guide/model_repository.md)]
* Writing a Model Configuration [[Overview](README.md#model-configuration) || [Details](user_guide/model_configuration.md)]
* Buillding a Model Pipeline [[Overview](README.md#model-pipeline)]
* Managing Model Availability [[Overview](README.md#model-management) || [Details](user_guide/model_management.md)]
* Collecting Server Metrics [[Overview](README.md#metrics) || [Details](user_guide/metrics.md)]
* Supporting Custom Ops/layers [[Overview](README.md#framework-custom-operations) || [Details](user_guide/custom_operations.md)]
* Using the Client API [[Overview](README.md#client-libraries-and-examples) || [Details](https://github.com/triton-inference-server/client)]
* Cancelling Inference Requests [[Overview](README.md#cancelling-inference-requests) || [Details](user_guide/request_cancellation.md)]
* Analyzing Performance [[Overview](README.md#performance-analysis)]
* Deploying on edge (Jetson) [[Overview](README.md#jetson-and-jetpack)]
* Debugging Guide [Details](./user_guide/debugging_guide.md)

### Model Repository
[Model Repositories](user_guide/model_repository.md) are the organizational hub for using Triton. All models, configuration files, and additional resources needed to serve the models are housed inside a model repository.
- [Cloud Storage](user_guide/model_repository.md#model-repository-locations)
- [File Organization](user_guide/model_repository.md#model-files)
- [Model Versioning](user_guide/model_repository.md#model-versions)
### Model Configuration

A [Model Configuration](user_guide/model_configuration.md) file is where you set the model-level options, such as output tensor reshaping and dynamic batch sizing.

#### Required Model Configuration

Triton Inference Server requires some [Minimum Required parameters](user_guide/model_configuration.md#minimal-model-configuration) to be filled in the Model Configuration. These required parameters essentially pertain to the structure of the model. For TensorFlow, ONNX and TensorRT models, users can rely on Triton to [Auto Generate](user_guide/model_configuration.md#auto-generated-model-configuration) the Minimum Required model configuration.
- [Maximum Batch Size - Batching and Non-Batching Models](user_guide/model_configuration.md#maximum-batch-size)
- [Input and Output Tensors](user_guide/model_configuration.md#inputs-and-outputs)
    - [Tensor Datatypes](user_guide/model_configuration.md#datatypes)
    - [Tensor Reshape](user_guide/model_configuration.md#reshape)
    - [Shape Tensor](user_guide/model_configuration.md#shape-tensors)

#### Versioning Models
Users need the ability to save and serve different versions of models based on business requirements. Triton allows users to set policies to make available different versions of the model as needed. [Learn More](user_guide/model_configuration.md#version-policy).

#### Instance Groups
Triton allows users to use of multiple instances of the same model. Users can specify how many instances (copies) of a model to load and whether to use GPU or CPU. If the model is being loaded on GPU, users can also select which GPUs to use. [Learn more](user_guide/model_configuration.md#instance-groups).
- [Specifying Multiple Model Instances](user_guide/model_configuration.md#multiple-model-instances)
- [CPU and GPU Instances](user_guide/model_configuration.md#cpu-model-instance)
- [Configuring Rate Limiter](user_guide/model_configuration.md#rate-limiter-configuration)

#### Optimization Settings

The Model Configuration ModelOptimizationPolicy property is used to specify optimization and prioritization settings for a model. These settings control if/how a model is optimized by the backend and how it is scheduled and executed by Triton. See the [ModelConfig Protobuf](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto) and [Optimization Documentation](user_guide/optimization.md#optimization) for the currently available settings.
- [Framework-Specific Optimization](user_guide/optimization.md#framework-specific-optimization)
  - [ONNX-TensorRT](user_guide/optimization.md#onnx-with-tensorrt-optimization-ort-trt)
  - [ONNX-OpenVINO](user_guide/optimization.md#onnx-with-openvino-optimization)
  - [TensorFlow-TensorRT](user_guide/optimization.md#tensorflow-with-tensorrt-optimization-tf-trt)
  - [TensorFlow-Mixed-Precision](user_guide/optimization.md#tensorflow-automatic-fp16-optimization)
- [NUMA Optimization](user_guide/optimization.md#numa-optimization)

#### Scheduling and Batching

Triton supports batching individual inference requests to improve compute resource utilization. This is extremely important as individual requests typically will not saturate GPU resources thus not leveraging the parallelism provided by GPUs to its extent. Learn more about Triton's [Batcher and Scheduler](user_guide/model_configuration.md#scheduling-and-batching).
- [Default Scheduler - Non-Batching](user_guide/model_configuration.md#default-scheduler)
- [Dynamic Batcher](user_guide/model_configuration.md#dynamic-batcher)
  - [How to Configure Dynamic Batcher](user_guide/model_configuration.md#recommended-configuration-process)
    - [Delayed Batching](user_guide/model_configuration.md#delayed-batching)
    - [Preferred Batch Size](user_guide/model_configuration.md#preferred-batch-sizes)
  - [Preserving Request Ordering](user_guide/model_configuration.md#preserve-ordering)
  - [Priority Levels](user_guide/model_configuration.md#priority-levels)
  - [Queuing Policies](user_guide/model_configuration.md#queue-policy)
  - [Ragged Batching](user_guide/ragged_batching.md)
- [Sequence Batcher](user_guide/model_configuration.md#sequence-batcher)
  - [Stateful Models](user_guide/model_execution.md#stateful-models)
  - [Control Inputs](user_guide/model_execution.md#control-inputs)
  - [Implicit State - Stateful Inference Using a Stateless Model](user_guide/implicit_state_management.md#implicit-state-management)
  - [Sequence Scheduling Strategies](user_guide/architecture.md#scheduling-strategies)
    - [Direct](user_guide/architecture.md#direct)
    - [Oldest](user_guide/architecture.md#oldest)

#### Rate Limiter
Rate limiter manages the rate at which requests are scheduled on model instances by Triton. The rate limiter operates across all models loaded in Triton to allow cross-model prioritization. [Learn more](user_guide/rate_limiter.md).

#### Model Warmup
For a few of the Backends (check [Additional Resources](README.md#resources)) some or all of initialization is deferred until the first inference request is received, the benefit is resource conservation but comes with the downside of the initial requests getting processed slower than expected. Users can pre-"warm up" the model by instructing Triton to initialize the model. [Learn more](user_guide/model_configuration.md#model-warmup).

#### Inference Request/Response Cache
Triton has a feature which allows inference responses to get cached. [Learn More](user_guide/response_cache.md).

### Model Pipeline
Building ensembles is as easy as adding an addition configuration file which outlines the specific flow of tensors from one model to another. Any additional changes required by the model ensemble can be made in existing (individual) model configurations.
- [Model Ensemble](user_guide/architecture.md#ensemble-models)
- [Business Logic Scripting (BLS)](https://github.com/triton-inference-server/python_backend#business-logic-scripting)
### Model Management
Users can specify policies in the model configuration for loading and unloading of models. This [section](user_guide/model_management.md) covers user selectable policy details.
- [Explicit Model Loading and Unloading](user_guide/model_management.md#model-control-mode-explicit)
- [Modifying the Model Repository](user_guide/model_management.md#modifying-the-model-repository)
### Metrics
Triton provides Prometheus metrics like GPU Utilization, Memory Usage, Latency and more. Learn about [available metrics](user_guide/metrics.md).
### Framework Custom Operations
Some frameworks provide the option of building custom layers/operations. These can be added to specific Triton Backends for the those frameworks. [Learn more](user_guide/custom_operations.md)
- [TensorRT](user_guide/custom_operations.md#tensorrt)
- [TensorFlow](user_guide/custom_operations.md#tensorflow)
- [PyTorch](user_guide/custom_operations.md#pytorch)
- [ONNX](user_guide/custom_operations.md#onnx)
### Client Libraries and Examples
Use the [Triton Client](https://github.com/triton-inference-server/client) API to integrate client applications over the network HTTP/gRPC API or integrate applications directly with Triton using CUDA shared memory to remove network overhead.
- [C++ HTTP/GRPC Libraries](https://github.com/triton-inference-server/client#client-library-apis)
- [Python HTTP/GRPC Libraries](https://github.com/triton-inference-server/client#client-library-apis)
- [Java HTTP Library](https://github.com/triton-inference-server/client/tree/main/src/java)
- GRPC Generated Libraries
  - [go](https://github.com/triton-inference-server/client/tree/main/src/grpc_generated/go)
  - [Java/Scala](https://github.com/triton-inference-server/client/tree/main/src/grpc_generated/java)
  - [Javascript](https://github.com/triton-inference-server/client/tree/main/src/grpc_generated/javascript)
- [Shared Memory Extension](protocol/extension_shared_memory.md)
### Cancelling Inference Requests
Triton can detect and handle requests that have been cancelled from the client-side. This [document](user_guide/request_cancellation.md) discusses scope and limitations of the feature.
### Performance Analysis
Understanding Inference performance is key to better resource utilization. Use Triton's Tools to costomize your deployment.
- [Performance Tuning Guide](user_guide/performance_tuning.md)
- [Optimization](user_guide/optimization.md)
- [Model Analyzer](user_guide/model_analyzer.md)
- [Performance Analyzer](https://github.com/triton-inference-server/perf_analyzer/blob/main/README.md)
- [Inference Request Tracing](user_guide/trace.md)
### Jetson and JetPack
Triton can be deployed on edge devices. Explore [resources](user_guide/jetson.md) and [examples](examples/jetson/README.md).

## **Resources**

The following resources are recommended to explore the full suite of Triton Inference Server's functionalities.
- **Clients**: Triton Inference Server comes with C++, Python and Java APIs with which users can send HTTP/REST or gRPC(possible extensions for other languages) requests. Explore the [client repository](https://github.com/triton-inference-server/server/tree/main/docs/protocol) for examples and documentation.

- **Configuring Deployment**: Triton comes with three tools which can be used to configure deployment setting, measure performance and recommend optimizations.
  - [Model Analyzer](https://github.com/triton-inference-server/model_analyzer) Model Analyzer is CLI tool built to recommend deployment configurations for Triton Inference Server based on user's Quality of Service Requirements. It also generates detailed reports about model performance to summarize the benefits and trade offs of different configurations.
  - [Perf Analyzer](https://github.com/triton-inference-server/perf_analyzer/blob/main/README.md):
  Perf Analyzer is a CLI application built to generate inference requests and
  measures the latency of those requests and throughput of the model being
  served.
  - [Model Navigator](https://github.com/triton-inference-server/model_navigator):
  The Triton Model Navigator is a tool that provides the ability to automate the process of moving model from source to optimal format and configuration for deployment on Triton Inference Server. The tool supports export model from source to all possible formats and applies the Triton Inference Server backend optimizations.

- **Backends**: Triton supports a wide variety of frameworks used to run models. Users can extend this functionality by creating custom backends.
  - [PyTorch](https://github.com/triton-inference-server/pytorch_backend): Widely used Open Source DL Framework
  - [TensorFlow](https://github.com/triton-inference-server/tensorflow_backend): Widely used Open Source DL Framework
  - [TensorRT](https://github.com/triton-inference-server/tensorrt_backend): NVIDIA [TensorRT](https://developer.nvidia.com/tensorrt) is an inference acceleration SDK that provide a with range of graph optimizations, kernel optimization, use of lower precision, and more.
  - [ONNX](https://github.com/triton-inference-server/onnxruntime_backend): ONNX Runtime is a cross-platform inference and training machine-learning accelerator.
  - [OpenVINO](https://github.com/triton-inference-server/openvino_backend): OpenVINO™ is an open-source toolkit for optimizing and deploying AI inference.
  - [Paddle Paddle](https://github.com/triton-inference-server/paddlepaddle_backend): Widely used Open Source DL Framework
  - [Python](https://github.com/triton-inference-server/python_backend): Users can add custom business logic, or any python code/model for serving requests.
  - [Forest Inference Library](https://github.com/triton-inference-server/fil_backend): Backend built for forest models trained by several popular machine learning frameworks (including XGBoost, LightGBM, Scikit-Learn, and cuML)
  - [DALI](https://github.com/triton-inference-server/dali_backend): NVIDIA [DALI](https://developer.nvidia.com/dali) is a Data Loading Library purpose built to accelerated pre-processing and data loading steps in a Deep Learning Pipeline.
  - [HugeCTR](https://github.com/triton-inference-server/hugectr_backend): HugeCTR is a GPU-accelerated recommender framework designed to distribute training across multiple GPUs and nodes and estimate Click-Through Rates
  - [Managed Stateful Models](https://github.com/triton-inference-server/stateful_backend): This backend automatically manages the input and output states of a model. The states are associated with a sequence id and need to be tracked for inference requests associated with the sequence id.
  - [Faster Transformer](https://github.com/triton-inference-server/fastertransformer_backend): NVIDIA [FasterTransformer](https://github.com/NVIDIA/FasterTransformer/) (FT) is a library implementing an accelerated engine for the inference of transformer-based neural networks, with a special emphasis on large models, spanning many GPUs and nodes in a distributed manner.
  - [Building Custom Backends](https://github.com/triton-inference-server/backend/tree/main/examples#tutorial)
  - [Sample Custom Backend: Repeat_backend](https://github.com/triton-inference-server/repeat_backend): Backend built to demonstrate sending of zero, one, or multiple responses per request.

## **Customization Guide**
This guide describes how to build and test Triton and also how Triton can be extended with new functionality.

- [Build](customization_guide/build.md)
- [Protocols and APIs](customization_guide/inference_protocols.md).
- [Backends](https://github.com/triton-inference-server/backend)
- [Repository Agents](customization_guide/repository_agents.md)
- [Test](customization_guide/test.md)
