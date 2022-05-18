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

# Triton Inference Server

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

**LATEST RELEASE: You are currently on the main branch which tracks
under-development progress towards the next release. The current release is 
version [2.21.0](https://github.com/triton-inference-server/server/tree/r22.04)
and corresponds to the 22.04 container release on 
[NVIDIA GPU Cloud (NGC)](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver).**

----
Triton Inference Server is an inference serving software that streamlines AI 
inferencing. Triton enables teams to deploy any AI model from multiple deep 
learning and machine learning frameworks, including TensorFlow, PyTorch, ONNX, 
OpenVINO, Python and more. Triton supports inference across cloud, data center,
edge and embedded devices on NVIDIA GPUs, x86 and ARM CPU, or AWS Inferentia. 
Triton delivers optimized performance for many query types, including real time,
batched, ensembles and audio/video streaming.

Major features include:

- [Multiple deep learning
  framework support](https://github.com/triton-inference-server/backend).
- [Multiple machine learning
  framework support](https://github.com/triton-inference-server/fil_backend).
- [Concurrent model
  execution](docs/architecture.md#concurrent-model-execution).
- [Dynamic batching](docs/architecture.md#models-and-schedulers).
- Provides [Extensible
  backend](https://github.com/triton-inference-server/backend) with a *backend
  API*.
- Model pipelines using
  [Ensembling](docs/architecture.md#ensemble-models) or [Business
  Logic Scripting
  (BLS)](https://github.com/triton-inference-server/python_backend#business-logic-scripting).
- [HTTP/REST and GRPC inference
  protocols](docs/inference_protocols.md) based on the community
  developed [KServe
  protocol](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2).
- A [C API](docs/inference_protocols.md#in-process-triton-server-api) and
  [Java API](docs/inference_protocols.md#java-bindings-for-in-process-triton-server-api) allow Triton to link directly into your application for edge and other in-process use cases.
- [Metrics](docs/metrics.md) indicating GPU utilization, server
  throughput, and server latency. 

## Serve a Model in 3 Easy Steps

```bash
# Step 1: Create the example model repository 
git clone -b r22.04 https://github.com/triton-inference-server/server.git

cd server/docs/examples

./fetch_models.sh

# Step 2: Launch triton from the NGC Triton container
docker pull nvcr.io/nvidia/tritonserver:22.04-py3

docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:22.04-py3 tritonserver --model-repository=/models

# Step 3: In a separate console, launch the image_client example from the NGC Triton SDK container
docker pull nvcr.io/nvidia/tritonserver:22.04-py3-sdk

docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:22.04-py3-sdk

/workspace/install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg

# Inference should return the following
Image '/workspace/images/mug.jpg':
    15.346230 (504) = COFFEE MUG
    13.224326 (968) = CUP
    10.422965 (505) = COFFEEPOT
```
Please read the [QuickStart](docs/quickstart.md) guide for additional information
regarding this example. The quickstart guide also contains an example of how to launch Triton on [CPU-only systems](docs/quickstart.md#run-on-cpu-only-system).

## Examples and Tutorials

Specific end-to-end examples for popular models, such as ResNet, BERT, and DLRM 
are located in the 
[NVIDIA Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples)
page on GitHub. The 
[NVIDIA Developer Zone](https://developer.nvidia.com/nvidia-triton-inference-server) 
contains additional documentation, presentations, and examples.
 
## Documentation

### Build and Deploy

The recommended way to build and use Triton Inference Server is with Docker
images.

- [Install Triton Inference Server with Docker containers](docs/build.md#building-triton-with-docker) (*Recommended*)
- [Install Triton Inference Server without Docker containers](docs/build.md#building-triton-without-docker)
- [Build a custom Triton Inference Server Docker container](docs/compose.md)
- [Build Triton Inference Server from source](docs/build.md#building-on-unsupported-platforms)
- [Build Triton Inference Server for Windows 10](docs/build.md#building-for-windows-10)
- Examples for deploying Triton Inference Server with Kubernetes and Helm on [GCP](deploy/gcp/README.md), 
  [AWS](deploy/aws/README.md), and [NVIDIA FleetCommand](deploy/fleetcommand/README.md)

### Using Triton

#### Preparing Models for Triton Inferencing

The first step in using Triton to serve your models is to place one or
more models into a [model repository](docs/model_repository.md). Depending on 
the type of the model and on what Triton capabilities you want to enable for
the model, you may need to create a [model
configuration](docs/model_configuration.md) for the model.  

- [Use the Model Analyzer to optimize your model 
  configuration](https://github.com/triton-inference-server/model_analyzer)
- [Add custom operations to Triton if needed by your model](docs/custom_operations.md)
- Enable model pipelining with [Model Ensemble](docs/architecture.md#ensemble-models)
  and [Business Logic Scripting (BLS)](https://github.com/triton-inference-server/python_backend#business-logic-scripting)

#### Configure and Use Triton Inference Server

- [TBD] Read tutorial on using Triton Inference Server
- [TBD] Customize Triton Inference Server for your use case
- Learn how to [optimizing performance](docs/optimization.md) using the 
  [Performance Analyzer](docs/perf_analyzer.md) and 
  [Model Analyzer](https://github.com/triton-inference-server/model_analyzer)
- Learn how to [manage loading and unloading models](model_management.md)
- Send requests directly to Triton with the [HTTP/REST JSON-based
  protocol](docs/inference_protocols.md#httprest-and-grpc-protocols)

#### Client Support and Examples

After you have your model(s) available in Triton, you will want to send 
inference and other requests to Triton from your *client* application. The 
[Python and C++ client libraries](https://github.com/triton-inference-server/client)
provide APIs to simplify this communication.

- Review client examples for [C++](https://github.com/triton-inference-server/client/blob/main/src/c%2B%2B/examples),
  [Python](https://github.com/triton-inference-server/client/blob/main/src/python/examples),
  and [Java](https://github.com/triton-inference-server/client/blob/main/src/java/src/main/java/triton/client/examples)
- Configure [HTTP](https://github.com/triton-inference-server/client#http-options)
  and [gRPC](https://github.com/triton-inference-server/client#grpc-options)
  client options
- Send input data (e.g. a jpeg image) directly to Triton in the [body of an HTTP 
  request without any additional metadata](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_binary_data.md#raw-binary-request)

### Extend Triton

TBD

### Overview

[Triton Architecture](docs/architecture.md) gives a high-level
overview of the structure and capabilities of the inference
server. Additional Triton Inference Server documentation include:
- [FAQ](docs/faq.md)
- [User Guide](docs#user-guide)
- [Developer Guide](docs#developer-guide)
- [Release Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html)
- [GPU, Driver, and CUDA Support
Matrix](https://docs.nvidia.com/deeplearning/dgx/support-matrix/index.html)

### OLD Documentation
----

Items in this section will be removed
### User Documentation



The [quickstart](docs/quickstart.md) walks you through all the steps
required to install and run Triton with an example image
classification model and then use an example client application to
perform inferencing using that model. The quickstart also demonstrates
how [Triton supports both GPU systems and CPU-only
systems](docs/quickstart.md#run-triton).



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
languages](https://github.com/triton-inference-server/client). For
certain types of models you can also send input data (e.g. a jpeg
image) directly to Triton in the [body of an HTTP request without any
additional
metadata](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_binary_data.md#raw-binary-request).

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
Triton also allows developers to write backends that can send multiple
responses for a request or not send any responses for a request. Backends
and models that operate in this way are referred to as [decoupled
backends and models](docs/decoupled_models.md).

A [Triton repository agent](docs/repository_agents.md) extends Triton
with new functionality that operates when a model is loaded or
unloaded. You can introduce your own code to perform authentication,
decryption, conversion, or similar operations when a model is loaded.

----

## Contributing

Contributions to Triton Inference Server are more than welcome. To
contribute make please review the [contribution 
guidelines](CONTRIBUTING.md). If you have a backend, client,
example or similar contribution that is not modifying the core of
Triton, then you should file a PR in the [contrib
repo](https://github.com/triton-inference-server/contrib).

## Reporting problems, asking questions

We appreciate any feedback, questions or bug reporting regarding this project. 
When posting [issues in GitHub](https://github.com/triton-inference-server/server/issues),
follow the process outlined in the [Stack Overflow document](https://stackoverflow.com/help/mcve).
Ensure posted examples are:
- minimal – use as little code as possible that still produces the
  same problem
- complete – provide all parts needed to reproduce the problem. Check
  if you can strip external dependency and still show the problem. The
  less time we spend on reproducing problems the more time we have to
  fix it
- verifiable – test the code you're about to provide to make sure it
  reproduces the problem. Remove all other problems that are not
  related to your request/question.

## For more information

Please refer to the [NVIDIA Developer Triton page](https://developer.nvidia.com/nvidia-triton-inference-server)
for more information.
