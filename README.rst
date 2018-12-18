..
  # Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

.. overview-begin-marker-do-not-remove

The NVIDIA TensorRT Inference Server (TRTIS) provides a cloud
inferencing solution optimized for NVIDIA GPUs. The server provides an
inference service via an HTTP or GRPC endpoint, allowing remote
clients to request inferencing for any model being managed by the
server.

What's New in 0.9.0 Beta
------------------------

* TRTIS now `monitors the model repository
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/model_repository.html#modifying-the-model-repository>`_
  for any change and dynamically reloads the model when necessary,
  without requiring a server restart. It is now possible to add and
  remove model versions, add/remove entire models, modify the model
  configuration, and modify the model labels while the server is
  running.
* Added a `model priority
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/model_configuration.html#optimization-policy>`_
  parameter to the model configuration. Currently the model priority
  controls the CPU thread priority when executing the model and for
  TensorRT models also controls the CUDA stream priority.
* Fixed a bug in GRPC API: changed the model version parameter from
  string to int. This is a non-backwards compatible change.
* Added --strict-model-config=false option to allow some `model
  configuration properties to be derived automatically
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/model_configuration.html#generated-model-configuration>`_. For
  some model types, this removes the need to specify the config.pbtxt
  file.
* Improved performance from an asynchronous GRPC frontend.

Features
--------

TRTIS provides the following features:

* `Multiple framework support
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/model_repository.html#framework-model-definition>`_. The
  server can manage any number and mix of models (limited by system
  disk and memory resources). Supports TensorRT, TensorFlow GraphDef,
  TensorFlow SavedModel and Caffe2 NetDef model formats. Also supports
  TensorFlow-TensorRT integrated models.
* Multi-GPU support. TRTIS can distribute inferencing across all
  system GPUs.
* `Concurrent model execution support
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/model_configuration.html?highlight=batching#instance-groups>`_. Multiple
  models (or multiple instances of the same model) can run
  simultaneously on the same GPU.
* Batching support. For models that support batching, the server can
  accept requests for a batch of inputs and respond with the
  corresponding batch of outputs. TRTIS also supports `dynamic
  batching
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/model_configuration.html?highlight=batching#dynamic-batching>`_
  where individual inference requests are dynamically combined
  together to improve inference throughput. Dynamic batching is
  transparent to the client requesting inference.
* `Model repositories
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/model_repository.html#>`_
  may reside on a locally accessible file system (e.g. NFS) or in
  Google Cloud Storage.
* Readiness and liveness `health endpoints
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/http_grpc_api.html#health>`_
  suitable for any orchestration or deployment framework, such as
  Kubernetes.
* `Metrics
  <https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/metrics.html>`_
  indicating GPU utiliization, server throughput, and server latency.

.. overview-end-marker-do-not-remove

The current release of the TensorRT Inference Server is 0.9.0 beta and
corresponds to the 18.12 release of the tensorrtserver container on
`NVIDIA GPU Cloud (NGC) <https://ngc.nvidia.com>`_. The branch for
this release is `r18.12
<https://github.com/NVIDIA/tensorrt-inference-server/tree/r18.12>`_.

Documentation
-------------

The User Guide, Developer Guide, and API Reference `documentation
<https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-guide/docs/index.html>`_
provide guidance on installing, building and running the latest
release of the TensorRT Inference Server.

You can also view the documentation for the `master branch
<https://docs.nvidia.com/deeplearning/sdk/tensorrt-inference-server-master-branch-guide/docs/index.html>`_
and for `earlier releases
<https://docs.nvidia.com/deeplearning/sdk/inference-server-archived/index.html>`_.

The `Release Notes
<https://docs.nvidia.com/deeplearning/sdk/inference-release-notes/index.html>`_
and `Support Matrix
<https://docs.nvidia.com/deeplearning/dgx/support-matrix/index.html>`_
indicate the required versions of the NVIDIA Driver and CUDA, and also
describe which GPUs are supported by TRTIS.

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
