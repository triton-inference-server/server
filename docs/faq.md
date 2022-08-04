<!--
# Copyright 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# FAQ

## What are the advantages of running a model with Triton Inference Server compared to running directly using the model's framework API?

When using Triton Inference Server the inference result will be the
same as when using the model's framework directly. However, with
Triton you get benefits like [concurrent model
execution](architecture.md#concurrent-model-execution) (the ability to
run multiple models at the same time on the same GPU) and [dynamic
batching](architecture.md#dynamic-batcher) to get better
throughput. You can also [replace or upgrade models while Triton and
client application are running](model_management.md). Another benefit
is that Triton can be deployed as a Docker container, anywhere – on
premises and on public clouds. Triton Inference Server also [supports
multiple
frameworks](https://github.com/triton-inference-server/backend) such
as TensorRT, TensorFlow, PyTorch, and ONNX on both GPUs and CPUs
leading to a streamlined deployment.

## Can Triton Inference Server run on systems that don't have GPUs?

Yes, the QuickStart guide describes how to [run Triton on a CPU-Only
System](quickstart.md#run-on-cpu-only-system).

## Can Triton Inference Server be used in non-Docker environments?

Yes. Triton Inference Server can also be [built from
source](build.md#building-without-docker) on your "bare metal"
system.

## Do you provide client libraries for languages other than C++ and Python?

We provide C++ and Python client libraries to make it easy for users
to write client applications that communicate with Triton. We chose
those languages because they were likely to be popular and performant
in the ML inference space, but in the future we can possibly add other
languages if there is a need.

We provide the GRPC API as a way to generate your own client library
for a large number of languages. By following the official GRPC
documentation and using
[grpc_service.proto](https://github.com/triton-inference-server/common/blob/main/protobuf/grpc_service.proto)
you can generate language bindings for all the languages supported by
GRPC. We provide three examples of this for
[Go](https://github.com/triton-inference-server/client/blob/main/src/grpc_generated/go), 
[Python](https://github.com/triton-inference-server/client/blob/main/src/python/examples/grpc_client.py) and
[Java](https://github.com/triton-inference-server/client/blob/main/src/grpc_generated/java).

In general the client libraries (and client examples) are meant to be
just that, examples. We feel the client libraries are well written and
well tested, but they are not meant to serve every possible use
case. In some cases you may want to develop your own customized
library to suit your specific needs.

## How would you use Triton Inference Server within the AWS environment?

In an AWS environment, the Triton Inference Server docker container
can run on [CPU-only instances or GPU compute
instances](quickstart.md#run-triton). Triton can run directly on the
compute instance or inside Elastic Kubernetes Service (EKS). In
addition, other AWS services such as Elastic Load Balancer (ELB) can
be used for load balancing traffic among multiple Triton
instances. Elastic Block Store (EBS) or S3 can be used for storing
deep-learning models loaded by the inference server.

## How do I measure the performance of my model running in the Triton Inference Server?

The Triton Inference Server exposes performance information in two
ways: by [Prometheus metrics](metrics.md) and by the statistics
available through the [HTTP/REST, GRPC, and C
APIs](inference_protocols.md).

A client application, [perf_analyzer](perf_analyzer.md), allows you to
measure the performance of an individual model using a synthetic
load. The perf_analyzer application is designed to show you the
tradeoff of latency vs. throughput.

## How can I fully utilize the GPU with Triton Inference Server?

Triton Inference Server has several features designed to increase
GPU utilization:

* Triton can [simultaneously perform inference for multiple
  models](architecture.md#concurrent-model-execution) (using either
  the same or different frameworks) using the same GPU.

* Triton can increase inference throughput by using [multiple
instances of the same
model](architecture.md#concurrent-model-execution) to handle multiple
simultaneous inferences requests to that model. Triton chooses
reasonable defaults but [you can also control the exact level of
concurrency](model_configuration.md#instance-groups) on a
model-by-model basis.

* Triton can [batch together multiple inference requests into a single
  inference execution](architecture.md#dynamic-batcher). Typically,
  batching inference requests leads to much higher thoughput with only
  a relatively small increase in latency.

As a general rule, batching is the most beneficial way to increase GPU
utilization. So you should always try enabling the [dynamic
batcher](architecture.md#dynamic-batcher) with your models. Using
multiple instances of a model can also provide some benefit but is
typically most useful for models that have small compute
requirements. Most models will benefit from using two instances but
more than that is often not useful.

## If I have a server with multiple GPUs should I use one Triton Inference Server to manage all GPUs or should I use multiple inference servers, one for each GPU?

Triton Inference Server will take advantage of all GPUs that it has
access to on the server. You can limit the GPUs available to Triton by
using the CUDA_VISIBLE_DEVICES environment variable (or with Docker
you can also use NVIDIA_VISIBLE_DEVICES or --gpus flag when launching
the container). When using multiple GPUs, Triton will distribute
inference request across the GPUs to keep them all equally
utilized. You can also [control more explicitly which models are
running on which GPUs](model_configuration.md#instance-groups).

In some deployment and orchestration environments (for example,
Kubernetes) it may be more desirable to partition a single multi-GPU
server into multiple *nodes*, each with one GPU. In this case the
orchestration environment will run a different Triton for each GPU and
an load balancer will be used to divide inference requests across the
available Triton instances.

## If the server segfaults, how can I debug it?

The NGC build is a Release build and does not contain Debug symbols. 
The build.py as well defaults to a Release build. Refer to the instructions
in [build.md](build.md#building-with-debug-symbols) to create a Debug build
of Triton. This will help find the cause of the segmentation fault when
looking at the gdb trace for the segfault.

When opening a GitHub issue for the segfault with Triton, please include
the backtrace to better help us resolve the problem.
