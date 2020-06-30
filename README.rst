..
  # Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

NVIDIA Triton Inference Server
==============================

    **LATEST RELEASE: You are currently on the master branch which
    tracks under-development progress towards the next release. The
    latest release of the Triton Inference Server is 2.0.0 and
    is available on branch** `r20.06
    <https://github.com/NVIDIA/triton-inference-server/tree/r20.06>`_.

    **Triton V2: Starting with the 20.06 release, Triton moves to
    version 2. The master branch currently tracks V2 development and
    is likely to be more unstable than usual due to the significant
    changes during the transition from V1 to V2. A legacy V1 version
    of Triton will be released from the master-v1 branch. The V1
    version of Triton is deprecated and no releases beyond 20.06 are
    planned. More information on the V1 and V2 transition is available
    in** `Roadmap
    <https://github.com/NVIDIA/triton-inference-server/blob/master/README.rst#roadmap>`_.

.. overview-begin-marker-do-not-remove

NVIDIA Triton Inference Server provides a cloud inferencing solution
optimized for NVIDIA GPUs. The server provides an inference service
via an HTTP/REST or GRPC endpoint, allowing remote clients to request
inferencing for any model being managed by the server. For edge
deployments, Triton Server is also available as a shared library with
an API that allows the full functionality of the server to be included
directly in an application. Triton Server provides the following
features:

* `Multiple framework support
  <https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/model_repository.html#framework-model-definition>`_. The
  server can manage any number and mix of models (limited by system
  disk and memory resources). Supports TensorRT, TensorFlow GraphDef,
  TensorFlow SavedModel, ONNX, PyTorch, and Caffe2 NetDef model
  formats. Also supports TensorFlow-TensorRT and ONNX-TensorRT
  integrated models. Variable-size input and output tensors are
  allowed if supported by the framework. See `Capabilities
  <https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/capabilities.html#capabilities>`_
  for detailed support information for each framework.

* `Concurrent model execution support
  <https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/model_configuration.html#instance-groups>`_. Multiple
  models (or multiple instances of the same model) can run
  simultaneously on the same GPU.

* Batching support. For models that support batching, Triton Server
  can accept requests for a batch of inputs and respond with the
  corresponding batch of outputs. Triton Server also supports multiple
  `scheduling and batching
  <https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/model_configuration.html#scheduling-and-batching>`_
  algorithms that combine individual inference requests together to
  improve inference throughput. These scheduling and batching
  decisions are transparent to the client requesting inference.

* `Custom backend support
  <https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/model_repository.html#custom-backends>`_. Triton
  Server allows individual models to be implemented with custom
  backends instead of by a deep-learning framework. With a custom
  backend a model can implement any logic desired, while still
  benefiting from the GPU support, concurrent execution, dynamic
  batching and other features provided by the server.

* `Ensemble support
  <https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/models_and_schedulers.html#ensemble-models>`_. An
  ensemble represents a pipeline of one or more models and the
  connection of input and output tensors between those models. A
  single inference request to an ensemble will trigger the execution
  of the entire pipeline.

* Multi-GPU support. Triton Server can distribute inferencing across
  all system GPUs.

* Triton Server provides `multiple modes for model management
  <https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/model_management.html>`_. These
  model management modes allow for both implicit and explicit loading
  and unloading of models without requiring a server restart.

* `Model repositories
  <https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/model_repository.html#>`_
  may reside on a locally accessible file system (e.g. NFS), in Google
  Cloud Storage or in Amazon S3.

* HTTP/REST and GRPC `inference protocols
  <https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/http_grpc_api.html>`_
  based on the community developed `KFServing protocol
  <https://github.com/kubeflow/kfserving/tree/master/docs/predict-api/v2>`_.

* Readiness and liveness `health endpoints
  <https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/http_grpc_api.html>`_
  suitable for any orchestration or deployment framework, such as
  Kubernetes.

* `Metrics
  <https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/metrics.html>`_
  indicating GPU utilization, server throughput, and server latency.

* `C library inferface
  <https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/library_api.html>`_
  allows the full functionality of Triton Server to be included
  directly in an application.

.. overview-end-marker-do-not-remove

The current release of the Triton Inference Server is 2.0.0 and
corresponds to the 20.06 release of the tensorrtserver container on
`NVIDIA GPU Cloud (NGC) <https://ngc.nvidia.com>`_. The branch for
this release is `r20.06
<https://github.com/NVIDIA/triton-inference-server/tree/r20.06>`_.

Backwards Compatibility
-----------------------

Version 2 of Triton is beta quality, so you should expect some changes
to the server and client protocols and APIs. Version 2 of Triton does
not generally maintain backwards compatibility with version 1.
Specifically, you should take the following items into account when
transitioning from version 1 to version 2:

* The Triton executables and libraries are in /opt/tritonserver. The
  Triton executable is /opt/tritonserver/bin/tritonserver.

* Some *tritonserver* command-line arguments are removed or have
  different default behavior in version 2.

  * --api-version, --http-health-port, --grpc-infer-thread-count,
    --grpc-stream-infer-thread-count,--allow-poll-model-repository
    and --allow-model-control are removed.

  * The default for --model-control-mode is changed to *none*.

* The HTTP/REST and GRPC protocols, while conceptually similar to
  version 1, are completely changed in version 2. See the `inference
  protocols
  <https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/http_grpc_api.html>`_
  section of the documentation for more information.

* Python and C++ client libraries are re-implemented to match the new
  HTTP/REST and GRPC protocols. The Python client no longer depends on
  a C++ shared library and so should be usable on any platform that
  supports Python. See the `client libraries
  <https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/client_library.html>`_
  section of the documentaion for more information.

* The version 2 cmake build requires these changes:

  * The cmake flag names have changed from having a TRTIS prefix to
    having a TRITON prefix. For example, TRITON_ENABLE_TENSORRT.

  * The build targets are *server*, *client* and *custom-backend* to
    build the server, client libraries and examples, and custom
    backend SDK, respectively.

* In the Docker containers the environment variables indicating the
  Triton version have changed to have a TRITON prefix, for example,
  TRITON_SERVER_VERSION.

Roadmap
-------

The 20.03.1 release of Triton consists of a single server/container
that supports both the existing version 1 APIs and protocols and the
new version 2 APIs and protocols. For version 2 the release is beta
quality and includes the new `HTTP/REST and GRPC protocols
<https://github.com/kubeflow/kfserving/tree/master/docs/predict-api/v2>`_
and corresponding new C++ and Python client libraries. Version 2 also
includes a beta release of the new server C API defined in
tritionserver.h.

The upcoming 20.06 release of Triton will include two separate server
containers:

* A legacy V1 version of Triton will be released from the master-v1
  branch. The NGC container for the V1 version of Triton will be
  called tritonserver:20.06-v1-py3. The V1 version of Triton is
  deprecated and no releases beyond 20.06 are planned. The V1 version
  of Triton maintains backwards compatibility with prior V1 versions
  in both the server APIs and in the C++ and Python libraries. See the
  `master-v1 branch README
  <https://github.com/NVIDIA/triton-inference-server/tree/master-v1>`_
  for more information.

* The new V2 version of Triton will be released from the master branch
  and will include the new GRPC and HTTP protocols based on `inference
  protocols
  <https://github.com/kubeflow/kfserving/tree/master/docs/predict-api/v2>`_
  that have been proposed by the `KFServing project
  <https://github.com/kubeflow/kfserving>`_. Version 2 of Triton will
  also have a new C API and new C++ and Python client libraries. The
  NGC container for version 2 of Triton will be called
  tritonserver:20.06-py3.

For both V1 and V2 the model repository struture and custom backend
APIs will remain unchanged so that any existing model repository and
custom backends will continue to work with Triton Server.

In the 20.06 V2 release there will be some changes to the tritonserver
command-line executable arguments to remove deprecated arguments and
adjust defaults. The changes will be detailed as part of the 20.06
release. It will be necessary to revisit and possibly adjust
invocations of tritonserver executable.

Documentation
-------------

The User Guide, Developer Guide, and API Reference `documentation for
the current release
<https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html>`_
provide guidance on installing, building, and running Triton Inference
Server.

You can also view the `documentation for the master branch
<https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/index.html>`_
and for `earlier releases
<https://docs.nvidia.com/deeplearning/triton-inference-server/archives/index.html>`_.

NVIDIA publishes a number of `deep learning examples that use Triton
<https://github.com/NVIDIA/DeepLearningExamples>`_.

An `FAQ
<https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/faq.html>`_
provides answers for frequently asked questions.

READMEs for deployment examples can be found in subdirectories of
deploy/, for example, `deploy/single_server/README.rst
<https://github.com/NVIDIA/triton-inference-server/tree/master/deploy/single_server/README.rst>`_.

The `Release Notes
<https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html>`_
and `Support Matrix
<https://docs.nvidia.com/deeplearning/dgx/support-matrix/index.html>`_
indicate the required versions of the NVIDIA Driver and CUDA, and also
describe which GPUs are supported by Triton Server.

Presentations and Papers
^^^^^^^^^^^^^^^^^^^^^^^^

* `High-Performance Inferencing at Scale Using the TensorRT Inference Server <https://developer.nvidia.com/gtc/2020/video/s22418>`_.

* `Accelerate and Autoscale Deep Learning Inference on GPUs with KFServing <https://developer.nvidia.com/gtc/2020/video/s22459>`_.

* `Deep into Triton Inference Server: BERT Practical Deployment on NVIDIA GPU <https://developer.nvidia.com/gtc/2020/video/s21736>`_.

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

Contributions to Triton Inference Server are more than welcome. To
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
