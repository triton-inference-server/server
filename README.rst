..
  # Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

|License|

NVIDIA TensorRT Inference Server
================================

    **LATEST RELEASE: You are currently on the master branch which
    tracks under-development progress towards the next release. The
    latest release of the TensorRT Inference Server is 1.5.0 and
    is available on branch** `r19.08
    <https://github.com/NVIDIA/tensorrt-inference-server/tree/r19.08>`_.

.. overview-begin-marker-do-not-remove

The NVIDIA TensorRT Inference Server provides a cloud inferencing
solution optimized for NVIDIA GPUs. The server provides an inference
service via an HTTP or GRPC endpoint, allowing remote clients to
request inferencing for any model being managed by the server. The
inference server provides the following features:

* `Multiple framework support
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/model_repository.html#framework-model-definition>`_. The
  server can manage any number and mix of models (limited by system
  disk and memory resources). Supports TensorRT, TensorFlow GraphDef,
  TensorFlow SavedModel, ONNX, PyTorch, and Caffe2 NetDef model
  formats. Also supports TensorFlow-TensorRT integrated
  models. Variable-size input and output tensors are allowed if
  supported by the framework. See `Capabilities
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/capabilities.html#capabilities>`_
  for detailed support information for each framework.

* `Concurrent model execution support
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/model_configuration.html#instance-groups>`_. Multiple
  models (or multiple instances of the same model) can run
  simultaneously on the same GPU.

* Batching support. For models that support batching, the server can
  accept requests for a batch of inputs and respond with the
  corresponding batch of outputs. The inference server also supports
  multiple `scheduling and batching
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/model_configuration.html#scheduling-and-batching>`_
  algorithms that combine individual inference requests together to
  improve inference throughput. These scheduling and batching
  decisions are transparent to the client requesting inference.

* `Custom backend support
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/model_repository.html#custom-backends>`_. The inference server
  allows individual models to be implemented with custom backends
  instead of by a deep-learning framework. With a custom backend a
  model can implement any logic desired, while still benefiting from
  the GPU support, concurrent execution, dynamic batching and other
  features provided by the server.

* `Ensemble support
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/models_and_schedulers.html#ensemble-models>`_. An
  ensemble represents a pipeline of one or more models and the
  connection of input and output tensors between those models. A
  single inference request to an ensemble will trigger the execution
  of the entire pipeline.

* Multi-GPU support. The server can distribute inferencing across all
  system GPUs.

* The inference server provides `multiple modes for model management
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/model_management.html>`_. These
  model management modes allow for both implicit and explicit loading
  and unloading of models without requiring a server restart.

* `Model repositories
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/model_repository.html#>`_
  may reside on a locally accessible file system (e.g. NFS), in Google
  Cloud Storage or in Amazon S3.

* Readiness and liveness `health endpoints
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/http_grpc_api.html#health>`_
  suitable for any orchestration or deployment framework, such as
  Kubernetes.

* `Metrics
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/metrics.html>`_
  indicating GPU utilization, server throughput, and server latency.

* `C library inferface
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/library_api.html>`_
  allows the full functionality of the inference server to be included
  directly in an application.

.. overview-end-marker-do-not-remove

The current release of the TensorRT Inference Server is 1.5.0 and
corresponds to the 19.08 release of the tensorrtserver container on
`NVIDIA GPU Cloud (NGC) <https://ngc.nvidia.com>`_. The branch for
this release is `r19.08
<https://github.com/NVIDIA/tensorrt-inference-server/tree/r19.08>`_.

Backwards Compatibility
-----------------------

Continuing in the latest version the following interfaces maintain
backwards compatibilty with the 1.0.0 release. If you have model
configuration files, custom backends, or clients that use the
inference server HTTP or GRPC APIs (either directly or through the
client libraries) from releases prior to 1.0.0 you should edit
and rebuild those as necessary to match the version 1.0.0 APIs.

These inferfaces will maintain backwards compatibility for all future
1.x.y releases (see below for exceptions):

* Model configuration as defined in `model_config.proto
  <https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/core/model_config.proto>`_.

* The inference server HTTP and GRPC APIs as defined in `api.proto
  <https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/core/api.proto>`_
  and `grpc_service.proto
  <https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/core/grpc_service.proto>`_.

* The custom backend interface as defined in `custom.h
  <https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/backends/custom/custom.h>`_.

As new features are introduced they may temporarily have beta status
where they are subject to change in non-backwards-compatible
ways. When they exit beta they will conform to the
backwards-compatibility guarantees described above. Currently the
following features are in beta:

* The inference server library API as defined in `trtserver.h
  <https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/core/trtserver.h>`_
  is currently in beta and may undergo non-backwards-compatible
  changes.

Documentation
-------------

The User Guide, Developer Guide, and API Reference `documentation for
the current release
<https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/index.html>`_
provide guidance on installing, building and running the TensorRT
Inference Server.

You can also view the `documentation for the master branch
<https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/index.html>`_
and for `earlier releases
<https://docs.nvidia.com/deeplearning/sdk/inference-server-archived/index.html>`_.

READMEs for deployment examples can be found in subdirectories of
deploy/, for example, `deploy/single_server/README.rst
<https://github.com/NVIDIA/tensorrt-inference-server/tree/master/deploy/single_server/README.rst>`_.

The `Release Notes
<https://docs.nvidia.com/deeplearning/sdk/inference-release-notes/index.html>`_
and `Support Matrix
<https://docs.nvidia.com/deeplearning/dgx/support-matrix/index.html>`_
indicate the required versions of the NVIDIA Driver and CUDA, and also
describe which GPUs are supported by the inference server.

Other Documentation
^^^^^^^^^^^^^^^^^^^

* `Maximizing Utilization for Data Center Inference with TensorRT
  Inference Server
  <https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=s9438-maximizing+utilization+for+data+center+inference+with+tensorrt+inference+server>`_.

* `NVIDIA TensorRT Inference Server Boosts Deep Learning Inference
  <https://devblogs.nvidia.com/nvidia-serves-deep-learning-inference/>`_.

* `GPU-Accelerated Inference for Kubernetes with the NVIDIA TensorRT
  Inference Server and Kubeflow
  <https://www.kubeflow.org/blog/nvidia_tensorrt/>`_.

Contributing
------------

Contributions to TensorRT Inference Server are more than welcome. To
contribute make a pull request and follow the guidelines outlined in
the `Contributing <CONTRIBUTING.md>`_ document.

Reporting problems, asking questions
------------------------------------

We appreciate any feedback, questions or bug reporting regarding this
project. When help with code is needed, follow the process outlined in
the Stack Overflow (https://stackoverflow.com/help/mcve)
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

.. |License| image:: https://img.shields.io/badge/License-BSD3-lightgrey.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
