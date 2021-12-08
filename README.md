<!--
# Copyright 2018-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Triton Inference Server

**LATEST RELEASE: You are currently on the main branch which tracks
under-development progress towards the next release.**

Triton Inference Server provides a cloud and edge inferencing solution
optimized for both CPUs and GPUs. Triton supports an HTTP/REST and
GRPC protocol that allows remote clients to request inferencing for
any model being managed by the server. For edge deployments, Triton is
available as a shared library with a C API that allows the full
functionality of Triton to be included directly in an
application.

The current release of the Triton Inference Server is 2.16.0 and
corresponds to the 21.11 release of the tritonserver container on
[NVIDIA GPU Cloud (NGC)](https://ngc.nvidia.com). The branch for this
release is
[r21.11](https://github.com/triton-inference-server/server/tree/r21.11).

## Features

* [Deep learning
  frameworks](https://github.com/triton-inference-server/backend).
  Triton supports TensorRT, TensorFlow GraphDef, TensorFlow
  SavedModel, ONNX, PyTorch TorchScript and OpenVINO model
  formats. Both TensorFlow 1.x and TensorFlow 2.x are
  supported. Triton also supports TensorFlow-TensorRT, ONNX-TensorRT
  and PyTorch-TensorRT integrated models.

* [Machine learning
  frameworks](https://github.com/triton-inference-server/fil_backend).
  Triton supports popular machine learning frameworks such as XGBoost,
  LightGBM, Scikit-Learn and cuML using the [RAPIDS Forest Inference
  Library](https://medium.com/rapids-ai/rapids-forest-inference-library-prediction-at-100-million-rows-per-second-19558890bc35).

* [Concurrent model
  execution](docs/architecture.md#concurrent-model-execution). Triton
  can simultaneously run multiple models (or multiple instances of the
  same model) using the same or different deep-learning and
  machine-learning frameworks.

* [Dynamic batching](docs/architecture.md#models-and-schedulers). For
  models that support batching, Triton implements multiple scheduling
  and batching algorithms that combine individual inference requests
  together to improve inference throughput. These scheduling and
  batching decisions are transparent to the client requesting
  inference.

* [Extensible
  backends](https://github.com/triton-inference-server/backend). In
  addition to deep-learning frameworks, Triton provides a *backend
  API* that allows Triton to be extended with any model execution
  logic implemented in
  [Python](https://github.com/triton-inference-server/python_backend)
  or
  [C++](https://github.com/triton-inference-server/backend/blob/main/README.md#triton-backend-api),
  while still benefiting from full CPU and GPU support, concurrent
  execution, dynamic batching and other features provided by Triton.

* Model pipelines using
  [Ensembling](docs/architecture.md#ensemble-models) or [Business
  Logic Scripting
  (BLS)](https://github.com/triton-inference-server/python_backend#business-logic-scripting).
  A Triton *ensemble* represents a pipeline of one or more models and
  the connection of input and output tensors between those
  models. *BLS* allows a pipeline along with extra business logic to
  be represented in Python. In both cases a single inference request
  will trigger the execution of the entire pipeline.

* [HTTP/REST and GRPC inference
  protocols](docs/inference_protocols.md) based on the community
  developed [KServe
  protocol](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2).

* A [C API](docs/inference_protocols.md#c-api) allows Triton to be
  linked directly into your application for edge and other in-process
  use cases.

* [Metrics](docs/metrics.md) indicating GPU utilization, server
  throughput, and server latency. The metrics are provided in
  Prometheus data format.

## Documentation

**The master branch documentation tracks the upcoming,
under-development release and so may not be accurate for the current
release of Triton. See the [r21.11
documentation](https://github.com/triton-inference-server/server/tree/r21.11#documentation)
for the current release.**

[Triton Architecture](docs/architecture.md) gives a high-level
overview of the structure and capabilities of the inference
server. There is also an [FAQ](docs/faq.md). Additional documentation
is divided into [*user*](#user-documentation) and
[*developer*](#developer-documentation) sections. The *user*
documentation describes how to use Triton as an inference solution,
including information on how to configure Triton, how to organize and
configure your models, how to use the C++ and Python clients, etc. The
*developer* documentation describes how to build and test Triton and
also how Triton can be extended with new functionality.

The Triton [Release
Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html)
and [Support
Matrix](https://docs.nvidia.com/deeplearning/dgx/support-matrix/index.html)
indicate the required versions of the NVIDIA Driver and CUDA, and also
describe supported GPUs.

### User Documentation

* [QuickStart](docs/quickstart.md)
  * [Install Triton](docs/quickstart.md#install-triton-docker-image)
  * [Create Model Repository](docs/quickstart.md#create-a-model-repository)
  * [Run Triton](docs/quickstart.md#run-triton)
* [Model Repository](docs/model_repository.md)
  * [Cloud Storage](docs/model_repository.md#model-repository-locations)
  * [File Organization](docs/model_repository.md#model-files)
  * [Model Versioning](docs/model_repository.md#model-versions)
* [Model Configuration](docs/model_configuration.md)
  * [Required Model Configuration](docs/model_configuration.md#minimal-model-configuration)
    * [Maximum Batch Size - Batching and Non-Batching Models](docs/model_configuration.md#maximum-batch-size)
    * [Input and Output Tensors](docs/model_configuration.md#inputs-and-outputs)
      * [Tensor Datatypes](docs/model_configuration.md#datatypes)
      * [Tensor Reshape](docs/model_configuration.md#reshape)
      * [Shape Tensor](docs/model_configuration.md#shape-tensors)
  * [Auto-Generate Required Model Configuration](docs/model_configuration.md#auto-generated-model-configuration)
  * [Version Policy](docs/model_configuration.md#version-policy)
  * [Instance Groups](docs/model_configuration.md#instance-groups)
    * [Specifying Multiple Model Instances](docs/model_configuration.md#multiple-model-instances)
    * [CPU and GPU Instances](docs/model_configuration.md#cpu-model-instance)
    * [Configuring Rate Limiter](docs/model_configuration.md#rate-limiter-configuration)
  * [Optimization Settings](docs/model_configuration.md#optimization_policy)
    * [Framework-Specific Optimization](docs/optimization.md#framework-specific-optimization)
      * [ONNX-TensorRT](docs/optimization.md#onnx-with-tensorrt-optimization)
      * [ONNX-OpenVINO](docs/optimization.md#onnx-with-openvino-optimization)
      * [TensorFlow-TensorRT](docs/optimization.md#tensorflow-with-tensorrt-optimization)
      * [TensorFlow-Mixed-Precision](docs/optimization.md#tensorflow-automatic-fp16-optimization)
    * [NUMA Optimization](docs/optimization.md#numa-optimization)
  * [Scheduling and Batching](docs/model_configuration.md#scheduling-and-batching)
    * [Default Scheduler - Non-Batching](docs/model_configuration.md#default-scheduler)
    * [Dynamic Batcher](docs/model_configuration.md#dynamic-batcher)
      * [How to Configure Dynamic Batcher](docs/model_configuration.md#recommended-configuration-process)
        * [Delayed Batching](docs/model_configuration.md#delayed-batching)
        * [Preferred Batch Size](docs/model_configuration.md#preferred-batch-sizes)
      * [Preserving Request Ordering](docs/model_configuration.md#preserve-ordering)
      * [Priority Levels](docs/model_configuration.md#priority-levels)
      * [Queuing Policies](docs/model_configuration.md#queue-policy)
      * [Ragged Batching](docs/ragged_batching.md)
    * [Sequence Batcher](docs/model_configuration.md#sequence-batcher)
      * [Stateful Models](docs/architecture.md#stateful-models)
      * [Control Inputs](docs/architecture.md#control-inputs)
      * [Implicit State - Stateful Inference Using a Stateless Model](docs/architecture.md#implicit-state-management)
      * [Sequence Scheduling Strategies](docs/architecture.md#scheduling-strateties)
        * [Direct](docs/architecture.md#direct)
        * [Oldest](docs/architecture.md#oldest)
    * [Rate Limiter](docs/rate_limiter.md)
  * [Model Warmup](docs/model_configuration.md#model-warmup)
  * [Inference Request/Response Cache](docs/model_configuration.md#response-cache)
* Model Pipeline
  * [Model Ensemble](docs/architecture.md#ensemble-models)
  * [Business Logic Scripting (BLS)](https://github.com/triton-inference-server/python_backend#business-logic-scripting)
* [Model Management](docs/model_management.md)
  * [Explicit Model Loading and Unloading](docs/model_management.md#model-control-mode-explicit)
  * [Modifying the Model Repository](docs/model_management.md#modifying-the-model-repository)
* [Metrics](docs/metrics.md)
* [Framework Custom Operations](docs/custom_operations.md)
  * [TensorRT](docs/custom_operations.md#tensorrt)
  * [TensorFlow](docs/custom_operations.md#tensorflow)
  * [PyTorch](docs/custom_operations.md#pytorch)
  * [ONNX](docs/custom_operations.md#onnx)
* [Client Libraries and Examples](https://github.com/triton-inference-server/client)
  * [C++ HTTP/GRPC Libraries](https://github.com/triton-inference-server/client#client-library-apis)
  * [Python HTTP/GRPC Libraries](https://github.com/triton-inference-server/client#client-library-apis)
  * [Java HTTP Library](https://github.com/triton-inference-server/client/src/java)
  * GRPC Generated Libraries
    * [go](https://github.com/triton-inference-server/client/tree/main/src/grpc_generated/go)
    * [Java/Scala](https://github.com/triton-inference-server/client/tree/main/src/grpc_generated/java)
* [Performance Analysis](docs/optimization.md)
  * [Model Analyzer](docs/model_analyzer.md)
  * [Performance Analyzer](docs/perf_analyzer.md)
  * [Inference Request Tracing](docs/trace.md)
* [Jetson and JetPack](docs/jetson.md)

The [quickstart](docs/quickstart.md) walks you through all the steps
required to install and run Triton with an example image
classification model and then use an example client application to
perform inferencing using that model. The quickstart also demonstrates
how [Triton supports both GPU systems and CPU-only
systems](docs/quickstart.md#run-triton).

The first step in using Triton to serve your models is to place one or
more models into a [model
repository](docs/model_repository.md). Optionally, depending on the type
of the model and on what Triton capabilities you want to enable for
the model, you may need to create a [model
configuration](docs/model_configuration.md) for the model.  If your
model has [custom operations](docs/custom_operations.md) you will need
to make sure they are loaded correctly by Triton.

After you have your model(s) available in Triton, you will want to
send inference and other requests to Triton from your *client*
application. The [Python and C++ client
libraries](https://github.com/triton-inference-server/client) provide
APIs to simplify this communication. There are also a large number of
[client examples](https://github.com/triton-inference-server/client)
that demonstrate how to use the libraries.  You can also send
HTTP/REST requests directly to Triton using the [HTTP/REST JSON-based
protocol](docs/inference_protocols.md#httprest-and-grpc-protocols) or
[generate a GRPC client for many other
languages](https://github.com/triton-inference-server/client).

Understanding and [optimizing performance](docs/optimization.md) is an
important part of deploying your models. The Triton project provides
the [Performance Analyzer](docs/perf_analyzer.md) and the [Model
Analyzer](docs/model_analyzer.md) to help your optimization
efforts. Specifically, you will want to optimize [scheduling and
batching](docs/architecture.md#models-and-schedulers) and [model
instances](docs/model_configuration.md#instance-groups) appropriately
for each model. You can also enable cross-model prioritization using
the [rate limiter](docs/rate_limiter.md) which manages the rate at
which requests are scheduled on model instances. You may also want to
consider combining multiple models and pre/post-processing into a
pipeline using [ensembling](docs/architecture.md#ensemble-models) or
[Business Logic Scripting
(BLS)](https://github.com/triton-inference-server/python_backend#business-logic-scripting). A
[Prometheus metrics endpoint](docs/metrics.md) allows you to visualize
and monitor aggregate inference metrics.

NVIDIA publishes a number of [deep learning
examples](https://github.com/NVIDIA/DeepLearningExamples) that use
Triton.

As part of your deployment strategy you may want to [explicitly manage
what models are available by loading and unloading
models](docs/model_management.md) from a running Triton server. If you
are using Kubernetes for deployment there are simple examples of how
to deploy Triton using Kubernetes and Helm:
[GCP](deploy/gcp/README.md), [AWS](deploy/aws/README.md), and [NVIDIA
FleetCommand](deploy/fleetcommand/README.md)

The [version 1 to version 2 migration
information](docs/v1_to_v2.md) is helpful if you are moving to
version 2 of Triton from previously using version 1.

### Developer Documentation

* [Build](docs/build.md)
* [Protocols and APIs](docs/inference_protocols.md).
* [Backends](https://github.com/triton-inference-server/backend)
* [Repository Agents](docs/repository_agents.md)
* [Test](docs/test.md)

Triton can be [built using
Docker](docs/build.md#building-triton-with-docker) or [built without
Docker](docs/build.md#building-triton-without-docker). After building
you should [test Triton](docs/test.md).

It is also possible to [create a Docker image containing a customized
Triton](docs/compose.md) that contains only a subset of the backends.

The Triton project also provides [client libraries for Python and
C++](https://github.com/triton-inference-server/client) that make it
easy to communicate with the server. There are also a large number of
[example clients](https://github.com/triton-inference-server/client)
that demonstrate how to use the libraries. You can also develop your
own clients that directly communicate with Triton using [HTTP/REST or
GRPC protocols](docs/inference_protocols.md). There is also a [C
API](docs/inference_protocols.md) that allows Triton to be linked
directly into your application.

A [Triton backend](https://github.com/triton-inference-server/backend)
is the implementation that executes a model. A backend can interface
with a deep learning framework, like PyTorch, TensorFlow, TensorRT or
ONNX Runtime; or it can interface with a data processing framework
like [DALI](https://github.com/triton-inference-server/dali_backend);
or you can extend Triton by [writing your own
backend](https://github.com/triton-inference-server/backend) in either
[C/C++](https://github.com/triton-inference-server/backend/blob/main/README.md#triton-backend-api)
or
[Python](https://github.com/triton-inference-server/python_backend).

A [Triton repository agent](docs/repository_agents.md) extends Triton
with new functionality that operates when a model is loaded or
unloaded. You can introduce your own code to perform authentication,
decryption, conversion, or similar operations when a model is loaded.

## Papers and Presentation

* [Maximizing Deep Learning Inference Performance with NVIDIA Model
  Analyzer](https://developer.nvidia.com/blog/maximizing-deep-learning-inference-performance-with-nvidia-model-analyzer/).

* [High-Performance Inferencing at Scale Using the TensorRT Inference
  Server](https://developer.nvidia.com/gtc/2020/video/s22418).

* [Accelerate and Autoscale Deep Learning Inference on GPUs with
  KFServing](https://developer.nvidia.com/gtc/2020/video/s22459).

* [Deep into Triton Inference Server: BERT Practical Deployment on
  NVIDIA GPU](https://developer.nvidia.com/gtc/2020/video/s21736).

* [Maximizing Utilization for Data Center Inference with TensorRT
  Inference Server](https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=s9438-maximizing+utilization+for+data+center+inference+with+tensorrt+inference+server).

* [NVIDIA TensorRT Inference Server Boosts Deep Learning
  Inference](https://devblogs.nvidia.com/nvidia-serves-deep-learning-inference/).

* [GPU-Accelerated Inference for Kubernetes with the NVIDIA TensorRT
  Inference Server and
  Kubeflow](https://www.kubeflow.org/blog/nvidia_tensorrt/).
  
* [Deploying NVIDIA Triton at Scale with MIG and Kubernetes](https://developer.nvidia.com/blog/deploying-nvidia-triton-at-scale-with-mig-and-kubernetes/). 

## Contributing

Contributions to Triton Inference Server are more than welcome. To
contribute make a pull request and follow the guidelines outlined in
[CONTRIBUTING.md](CONTRIBUTING.md). If you have a backend, client,
example or similar contribution that is not modifying the core of
Triton, then you should file a PR in the [contrib
repo](https://github.com/triton-inference-server/contrib).

## Reporting problems, asking questions

We appreciate any feedback, questions or bug reporting regarding this
project. When help with code is needed, follow the process outlined in
the Stack Overflow (<https://stackoverflow.com/help/mcve>)
document. Ensure posted examples are:

* minimal – use as little code as possible that still produces the
  same problem

* complete – provide all parts needed to reproduce the problem. Check
  if you can strip external dependency and still show the problem. The
  less time we spend on reproducing problems the more time we have to
  fix it

* verifiable – test the code you're about to provide to make sure it
  reproduces the problem. Remove all other problems that are not
  related to your request/question.
