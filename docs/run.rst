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

.. _section-running-triton:

Running Triton
==============

For best performance the Triton Inference Server should be run on a
system that contains Docker, nvidia-docker, CUDA and one or more
supported GPUs, as explained in
:ref:`section-running-triton-with-gpu`. Triton can also be run on
non-CUDA, non-GPU systems as described in
:ref:`section-running-triton-without-gpu`.

If you :ref:`build Triton outside of Docker
<section-building-the-server-with-cmake>`, you can then run Triton
without Docker, as explained in
:ref:`section-running-triton-without-docker`.

.. _section-example-model-repository:

Example Model Repository
------------------------

Before running the Triton, you must first set up a model repository
containing the models that the server will make available for
inferencing.

An example model repository containing a Caffe2 ResNet50, a TensorFlow
Inception model and an ONNX densenet model. The example repository
also contains two simple TensorFlow GraphDef models that are used by
the :ref:`example client applications
<section-client-examples>`. These models are provided in the
`docs/examples/model_repository
<https://github.com/triton-inference-server/server/tree/master/docs/examples/model_repository>`_
directory. Before using the example model repository you must fetch
any missing model definition files from their public model zoos. Be
sure to checkout the release version of the branch that corresponds to
the server you are using (or the master branch if you are using a
server build from master)::

  $ git checkout r20.08
  $ cd docs/examples
  $ ./fetch_models.sh

An example ensemble model repository is also provided in the
`docs/examples/ensemble_model_repository
<https://github.com/triton-inference-server/server/tree/master/docs/examples/ensemble_model_repository>`_
directory. It contains a custom image preprocess model, Caffe2
ResNet50, and an ensemble model that are used by the :ref:`ensemble
example <section-ensemble-image-classification-example>`.

Before using the example ensemble model repository, in addition to
fetching public model definition files as mentioned above, you must
build the custom backend for the custom image preprocess model (see
:ref:`section-building-a-custom-backend` for instructions). Also note
that although ensemble models are fully specified in their model
configuration, empty version directories are required for them to be
recognized as valid model directories::

  $ cd docs/examples
  $ mkdir -p ensemble_model_repository/preprocess_resnet50_ensemble/1

.. _section-running-triton-with-gpu:

Running Triton On A System With A GPU
-------------------------------------

Before running Triton you must first set up a model repository
containing the models that you want to be available for
inferencing. Section :ref:`section-model-repository` describes how to
create your own model repository. You can also follow the steps above
in :ref:`section-example-model-repository` to set up an example model
repository.

Assuming the model repository is available in
/path/to/model/repository, the following command runs the container
you pulled from NGC or built locally::

  $ docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/path/to/model/repository:/models <tritonserver image name> tritonserver --model-repository=/models

Where *<tritonserver image name>* will be something like
**nvcr.io/nvidia/tritonserver:20.08-py3** if you :ref:`pulled the
container from the NGC registry <section-installing-triton>`, or
**tritonserver** if you :ref:`built it from source
<section-building>`.

The docker -v option maps /path/to/model/repository on the host into
the container at /models, and the -\\-model-repository option to
Triton is used to point to /models as the model repository.

The -p flags expose the container ports where Triton listens for HTTP
requests (port 8000), listens for GRPC requests (port 8001), and
reports Prometheus metrics (port 8002).

You may also want to use the -\\-shm-size and -\\-ulimit flags to
improve the server's performance depending on how you are using
Triton.

For more information on the Prometheus metrics provided by Triton see
:ref:`section-metrics`.

.. _section-running-triton-without-gpu:

Running Triton On A System Without A GPU
----------------------------------------

On a system without GPUs, Triton should be run without using the
-\\-gpus flag to docker, but is otherwise identical to what is
described in :ref:`section-running-triton-with-gpu`::

  $ docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/path/to/model/repository:/models <tritonserver image name> tritonserver --model-repository=/models

Because the -\\-gpus flag is not used, a GPU is not available and
Triton will therefore be unable to load any model configuration that
requires a GPU or that specifies a GPU instance by an
:ref:`instance-group <section-instance-groups>` configuration.

.. _section-running-triton-without-docker:

Running Triton Without Docker
-----------------------------

After :ref:`building Triton outside of Docker
<section-building-the-server-with-cmake>`, the *tritonserver*
executable will be in builddir/server/install/bin and the required
shared libraries will be in builddir/server/install/lib. The
*tritonserver* executable and libraries are configured to be installed
and executed from the /opt/tritonserver directory, so copy
builddir/server/install/* to /opt/tritonserver/. . Then execute
*tritonserver* with the desired arguments::

  $ /opt/tritonserver/bin/tritonserver --model-repository=/models

.. _section-checking-inference-server-status:

Checking Triton Status
----------------------

The simplest way to verify that the inference server is running and
ready to perform inference is to use the server *ready* API to query
the server’s status. From the host system use curl to access the HTTP
endpoint that indicates server status. For example::

  $ curl -v localhost:8000/v2/health/ready
  ...
  < HTTP/1.1 200 OK
  < Content-Length: 0
  < Content-Type: text/plain

The HTTP request returns status 200 if Triton is ready and non-200 if
it is not ready.

Triton provides several additional ready, status and metadata
APIs. See :ref:`section-http-grpc-protocol` for more information on
the HTTP/REST and GRPC protocols that are supported by Triton.
