<!--
# Copyright 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Triton Inference Server Documentation

## User Guide
The User Guide describes how to use Triton as an inference solution, including information on how to configure Triton, how to organize and configure your models, how to use the C++ and Python clients, etc. 

- [QuickStart](quickstart.md)
  - [Install Triton](quickstart.md#install-triton-docker-image)
  - [Create Model Repository](quickstart.md#create-a-model-repository)
  - [Run Triton](quickstart.md#run-triton)
- [Model Repository](model_repository.md)
  - [Cloud Storage](model_repository.md#model-repository-locations)
  - [File Organization](model_repository.md#model-files)
  - [Model Versioning](model_repository.md#model-versions)
- [Model Configuration](model_configuration.md)
  - [Required Model Configuration](model_configuration.md#minimal-model-configuration)
    - [Maximum Batch Size - Batching and Non-Batching Models](model_configuration.md#maximum-batch-size)
    - [Input and Output Tensors](model_configuration.md#inputs-and-outputs)
      - [Tensor Datatypes](model_configuration.md#datatypes)
      - [Tensor Reshape](model_configuration.md#reshape)
      - [Shape Tensor](model_configuration.md#shape-tensors)
  - [Auto-Generate Required Model Configuration](model_configuration.md#auto-generated-model-configuration)
  - [Version Policy](model_configuration.md#version-policy)
  - [Instance Groups](model_configuration.md#instance-groups)
    - [Specifying Multiple Model Instances](model_configuration.md#multiple-model-instances)
    - [CPU and GPU Instances](model_configuration.md#cpu-model-instance)
    - [Configuring Rate Limiter](model_configuration.md#rate-limiter-configuration)
  - [Optimization Settings](model_configuration.md#optimization-policy)
    - [Framework-Specific Optimization](optimization.md#framework-specific-optimization)
      - [ONNX-TensorRT](optimization.md#onnx-with-tensorrt-optimization-ort-trt)
      - [ONNX-OpenVINO](optimization.md#onnx-with-openvino-optimization)
      - [TensorFlow-TensorRT](optimization.md#tensorflow-with-tensorrt-optimization-tf-trt)
      - [TensorFlow-Mixed-Precision](optimization.md#tensorflow-automatic-fp16-optimization)
    - [NUMA Optimization](optimization.md#numa-optimization)
  - [Scheduling and Batching](model_configuration.md#scheduling-and-batching)
    - [Default Scheduler - Non-Batching](model_configuration.md#default-scheduler)
    - [Dynamic Batcher](model_configuration.md#dynamic-batcher)
      - [How to Configure Dynamic Batcher](model_configuration.md#recommended-configuration-process)
        - [Delayed Batching](model_configuration.md#delayed-batching)
        - [Preferred Batch Size](model_configuration.md#preferred-batch-sizes)
      - [Preserving Request Ordering](model_configuration.md#preserve-ordering)
      - [Priority Levels](model_configuration.md#priority-levels)
      - [Queuing Policies](model_configuration.md#queue-policy)
      - [Ragged Batching](ragged_batching.md)
    - [Sequence Batcher](model_configuration.md#sequence-batcher)
      - [Stateful Models](architecture.md#stateful-models)
      - [Control Inputs](architecture.md#control-inputs)
      - [Implicit State - Stateful Inference Using a Stateless Model](architecture.md#implicit-state-management)
      - [Sequence Scheduling Strategies](architecture.md#scheduling-strategies)
        - [Direct](architecture.md#direct)
        - [Oldest](architecture.md#oldest)
    - [Rate Limiter](rate_limiter.md)
  - [Model Warmup](model_configuration.md#model-warmup)
  - [Inference Request/Response Cache](model_configuration.md#response-cache)
- Model Pipeline
  - [Model Ensemble](architecture.md#ensemble-models)
  - [Business Logic Scripting (BLS)](https://github.com/triton-inference-server/python_backend#business-logic-scripting)
- [Model Management](model_management.md)
  - [Explicit Model Loading and Unloading](model_management.md#model-control-mode-explicit)
  - [Modifying the Model Repository](model_management.md#modifying-the-model-repository)
- [Metrics](metrics.md)
- [Framework Custom Operations](custom_operations.md)
  - [TensorRT](custom_operations.md#tensorrt)
  - [TensorFlow](custom_operations.md#tensorflow)
  - [PyTorch](custom_operations.md#pytorch)
  - [ONNX](custom_operations.md#onnx)
- [Client Libraries and Examples](https://github.com/triton-inference-server/client)
  - [C++ HTTP/GRPC Libraries](https://github.com/triton-inference-server/client#client-library-apis)
  - [Python HTTP/GRPC Libraries](https://github.com/triton-inference-server/client#client-library-apis)
  - [Java HTTP Library](https://github.com/triton-inference-server/client/tree/main/src/java)
  - GRPC Generated Libraries
    - [go](https://github.com/triton-inference-server/client/tree/main/src/grpc_generated/go)
    - [Java/Scala](https://github.com/triton-inference-server/client/tree/main/src/grpc_generated/java)
    - [Javascript](https://github.com/triton-inference-server/client/tree/main/src/grpc_generated/javascript)
- Performance Analysis
  - [Performance Tuning Guide](performance_tuning.md)
  - [Optimization](optimization.md)
  - [Model Analyzer](model_analyzer.md)
  - [Performance Analyzer](perf_analyzer.md)
  - [Inference Request Tracing](trace.md)
- [Jetson and JetPack](jetson.md)

## Developer Guide
The Developer Guide describes how to build and test Triton and also how Triton can be extended with new functionality.

- [Build](build.md)
- [Protocols and APIs](inference_protocols.md).
- [Backends](https://github.com/triton-inference-server/backend)
- [Repository Agents](repository_agents.md)
- [Test](test.md)
