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

# Triton Inference Server

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

> [!WARNING]
> ##### LATEST RELEASE
> You are currently on the `main` branch which tracks under-development progress towards the next release.
> The current release is version [2.41.0](https://github.com/triton-inference-server/server/releases/latest) and corresponds to the 23.12 container release on NVIDIA GPU Cloud (NGC).

Triton Inference Server is an open source inference serving software that
streamlines AI inferencing. Triton enables teams to deploy any AI model from
multiple deep learning and machine learning frameworks, including TensorRT,
TensorFlow, PyTorch, ONNX, OpenVINO, Python, RAPIDS FIL, and more. Triton
Inference Server supports inference across cloud, data center, edge and embedded
devices on NVIDIA GPUs, x86 and ARM CPU, or AWS Inferentia. Triton Inference
Server delivers optimized performance for many query types, including real time,
batched, ensembles and audio/video streaming. Triton inference Server is part of
[NVIDIA AI Enterprise](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/),
a software platform that accelerates the data science pipeline and streamlines
the development and deployment of production AI.

Major features include:

- [Supports multiple deep learning
  frameworks](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton)
- [Supports multiple machine learning
  frameworks](https://github.com/triton-inference-server/fil_backend)
- [Concurrent model
  execution](docs/user_guide/architecture.md#concurrent-model-execution)
- [Dynamic batching](docs/user_guide/model_configuration.md#dynamic-batcher)
- [Sequence batching](docs/user_guide/model_configuration.md#sequence-batcher) and
  [implicit state management](docs/user_guide/architecture.md#implicit-state-management)
  for stateful models
- Provides [Backend API](https://github.com/triton-inference-server/backend) that
  allows adding custom backends and pre/post processing operations
- Supports writing custom backends in python, a.k.a.
  [Python-based backends.](https://github.com/triton-inference-server/backend/blob/main/docs/python_based_backends.md#python-based-backends)
- Model pipelines using
  [Ensembling](docs/user_guide/architecture.md#ensemble-models) or [Business
  Logic Scripting
  (BLS)](https://github.com/triton-inference-server/python_backend#business-logic-scripting)
- [HTTP/REST and GRPC inference
  protocols](docs/customization_guide/inference_protocols.md) based on the community
  developed [KServe
  protocol](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2)
- A [C API](docs/customization_guide/inference_protocols.md#in-process-triton-server-api) and
  [Java API](docs/customization_guide/inference_protocols.md#java-bindings-for-in-process-triton-server-api)
  allow Triton to link directly into your application for edge and other in-process use cases
- [Metrics](docs/user_guide/metrics.md) indicating GPU utilization, server
  throughput, server latency, and more

**New to Triton Inference Server?** Make use of
[these tutorials](https://github.com/triton-inference-server/tutorials)
to begin your Triton journey!

Join the [Triton and TensorRT community](https://www.nvidia.com/en-us/deep-learning-ai/triton-tensorrt-newsletter/) and
stay current on the latest product updates, bug fixes, content, best practices,
and more.  Need enterprise support?  NVIDIA global support is available for Triton
Inference Server with the
[NVIDIA AI Enterprise software suite](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/).

## Serve a Model in 3 Easy Steps

```bash
# Step 1: Create the example model repository
git clone -b r23.12 https://github.com/triton-inference-server/server.git
cd server/docs/examples
./fetch_models.sh

# Step 2: Launch triton from the NGC Triton container
docker run --gpus=1 --rm --net=host -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:23.12-py3 tritonserver --model-repository=/models

# Step 3: Sending an Inference Request
# In a separate console, launch the image_client example from the NGC Triton SDK container
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.12-py3-sdk
/workspace/install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg

# Inference should return the following
Image '/workspace/images/mug.jpg':
    15.346230 (504) = COFFEE MUG
    13.224326 (968) = CUP
    10.422965 (505) = COFFEEPOT
```
Please read the [QuickStart](docs/getting_started/quickstart.md) guide for additional information
regarding this example. The quickstart guide also contains an example of how to launch Triton on [CPU-only systems](docs/getting_started/quickstart.md#run-on-cpu-only-system). New to Triton and wondering where to get started? Watch the [Getting Started video](https://youtu.be/NQDtfSi5QF4).

## Examples and Tutorials

Check out [NVIDIA LaunchPad](https://www.nvidia.com/en-us/data-center/products/ai-enterprise-suite/trial/)
for free access to a set of hands-on labs with Triton Inference Server hosted on
NVIDIA infrastructure.

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

- [Install Triton Inference Server with Docker containers](docs/customization_guide/build.md#building-with-docker) (*Recommended*)
- [Install Triton Inference Server without Docker containers](docs/customization_guide/build.md#building-without-docker)
- [Build a custom Triton Inference Server Docker container](docs/customization_guide/compose.md)
- [Build Triton Inference Server from source](docs/customization_guide/build.md#building-on-unsupported-platforms)
- [Build Triton Inference Server for Windows 10](docs/customization_guide/build.md#building-for-windows-10)
- Examples for deploying Triton Inference Server with Kubernetes and Helm on [GCP](deploy/gcp/README.md),
  [AWS](deploy/aws/README.md), and [NVIDIA FleetCommand](deploy/fleetcommand/README.md)
- [Secure Deployment Considerations](docs/customization_guide/deploy.md)

### Using Triton

#### Preparing Models for Triton Inference Server

The first step in using Triton to serve your models is to place one or
more models into a [model repository](docs/user_guide/model_repository.md). Depending on
the type of the model and on what Triton capabilities you want to enable for
the model, you may need to create a [model
configuration](docs/user_guide/model_configuration.md) for the model.

- [Add custom operations to Triton if needed by your model](docs/user_guide/custom_operations.md)
- Enable model pipelining with [Model Ensemble](docs/user_guide/architecture.md#ensemble-models)
  and [Business Logic Scripting (BLS)](https://github.com/triton-inference-server/python_backend#business-logic-scripting)
- Optimize your models setting [scheduling and batching](docs/user_guide/architecture.md#models-and-schedulers)
  parameters and [model instances](docs/user_guide/model_configuration.md#instance-groups).
- Use the [Model Analyzer tool](https://github.com/triton-inference-server/model_analyzer)
  to help optimize your model configuration with profiling
- Learn how to [explicitly manage what models are available by loading and
  unloading models](docs/user_guide/model_management.md)

#### Configure and Use Triton Inference Server

- Read the [Quick Start Guide](docs/getting_started/quickstart.md) to run Triton Inference
  Server on both GPU and CPU
- Triton supports multiple execution engines, called
  [backends](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton), including
  [TensorRT](https://github.com/triton-inference-server/tensorrt_backend),
  [TensorFlow](https://github.com/triton-inference-server/tensorflow_backend),
  [PyTorch](https://github.com/triton-inference-server/pytorch_backend),
  [ONNX](https://github.com/triton-inference-server/onnxruntime_backend),
  [OpenVINO](https://github.com/triton-inference-server/openvino_backend),
  [Python](https://github.com/triton-inference-server/python_backend), and more
- Not all the above backends are supported on every platform supported by Triton.
  Look at the
  [Backend-Platform Support Matrix](https://github.com/triton-inference-server/backend/blob/main/docs/backend_platform_support_matrix.md)
  to learn which backends are supported on your target platform.
- Learn how to [optimize performance](docs/user_guide/optimization.md) using the
  [Performance Analyzer](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md)
  and
  [Model Analyzer](https://github.com/triton-inference-server/model_analyzer)
- Learn how to [manage loading and unloading models](docs/user_guide/model_management.md) in
  Triton
- Send requests directly to Triton with the [HTTP/REST JSON-based
  or gRPC protocols](docs/customization_guide/inference_protocols.md#httprest-and-grpc-protocols)

#### Client Support and Examples

A Triton *client* application sends inference and other requests to Triton. The
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

[Triton Inference Server's architecture](docs/user_guide/architecture.md) is specifically
designed for modularity and flexibility

- [Customize Triton Inference Server container](docs/customization_guide/compose.md) for your use case
- [Create custom backends](https://github.com/triton-inference-server/backend)
  in either [C/C++](https://github.com/triton-inference-server/backend/blob/main/README.md#triton-backend-api)
  or [Python](https://github.com/triton-inference-server/python_backend)
- Create [decoupled backends and models](docs/user_guide/decoupled_models.md) that can send
  multiple responses for a request or not send any responses for a request
- Use a [Triton repository agent](docs/customization_guide/repository_agents.md) to add functionality
  that operates when a model is loaded and unloaded, such as authentication,
  decryption, or conversion
- Deploy Triton on [Jetson and JetPack](docs/user_guide/jetson.md)
- [Use Triton on AWS
   Inferentia](https://github.com/triton-inference-server/python_backend/tree/main/inferentia)

### Additional Documentation

- [FAQ](docs/user_guide/faq.md)
- [User Guide](docs/README.md#user-guide)
- [Customization Guide](docs/README.md#customization-guide)
- [Release Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html)
- [GPU, Driver, and CUDA Support
Matrix](https://docs.nvidia.com/deeplearning/dgx/support-matrix/index.html)

## Contributing

Contributions to Triton Inference Server are more than welcome. To
contribute please review the [contribution
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
  if you can strip external dependencies and still show the problem. The
  less time we spend on reproducing problems the more time we have to
  fix it
- verifiable – test the code you're about to provide to make sure it
  reproduces the problem. Remove all other problems that are not
  related to your request/question.

For issues, please use the provided bug report and feature request templates.

For questions, we recommend posting in our community
[GitHub Discussions.](https://github.com/triton-inference-server/server/discussions)

## For more information

Please refer to the [NVIDIA Developer Triton page](https://developer.nvidia.com/nvidia-triton-inference-server)
for more information.
