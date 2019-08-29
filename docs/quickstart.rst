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

Quickstart
==========

The TensorRT Inference Server is available in two ways:

* As a pre-built Docker container available from the `NVIDIA GPU Cloud
  (NGC) <https://ngc.nvidia.com>`_. For more information, see
  :ref:`section-using-a-prebuilt-docker-container`.

* As buildable source code located in GitHub. You can :ref:`build your
  own container using Docker<section-building-with-docker>` or you can
  :ref:`build using CMake<section-building-with-cmake>`.

.. _section-prerequisites:

Prerequisites
-------------

Regardless of which method you choose (starting with a pre-built
container from NGC or building from source), you must perform the
following prerequisite steps:

* Clone the TensorRT Inference Server GitHub repo. Even if you choose
  to get the pre-built inference server from NGC, you need the GitHub
  repo for the example model repository and to build the example
  applications. Go to
  https://github.com/NVIDIA/tensorrt-inference-server and then select
  the *clone* or *download* drop down button. After clone the repo be
  sure to select the r<xx.yy> release branch that corresponds to the
  version of the server you want to use::

  $ git checkout r19.07

* Create a model repository containing one or more models that you
  want the inference server to serve. An example model repository is
  included in the docs/examples/model_repository directory of the
  GitHub repo. Before using the repository, you must fetch any missing
  model definition files from their public model zoos via the provided
  docs/examples/fetch_models.sh script::

  $ cd docs/examples
  $ ./fetch_models.sh

If you are starting with a pre-built NGC container perform these
additional steps:

* Ensure you have access and are logged into NGC.  For step-by-step
  instructions, see the `NGC Getting Started Guide
  <http://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html>`_.

* Install Docker and nvidia-docker.  For DGX users, see `Preparing to
  use NVIDIA Containers
  <http://docs.nvidia.com/deeplearning/dgx/preparing-containers/index.html>`_.
  For users other than DGX, see the `nvidia-docker installation
  documentation <https://github.com/NVIDIA/nvidia-docker>`_.

.. _section-using-a-prebuilt-docker-container:

Using A Prebuilt Docker Container
---------------------------------

Make sure you log into NGC as described in
:ref:`section-prerequisites` before attempting the steps in this
section.  Use docker pull to get the TensorRT Inference Server
container from NGC::

  $ docker pull nvcr.io/nvidia/tensorrtserver:<xx.yy>-py3

Where <xx.yy> is the version of the inference server that you want to
pull. Once you have the container follow these steps to run the server
and the example client applications.

#. :ref:`Run the inference server <section-run-tensorrt-inference-server>`.
#. :ref:`Verify that the server is running correct <section-verify-inference-server-status>`.
#. :ref:`Get the example client applications <section-getting-the-client-examples>`.
#. :ref:`Run the image classification example <section-running-the-image-classification-example>`.

.. _section-building-with-docker:

Building With Docker
--------------------

Make sure you complete the steps in :ref:`section-prerequisites`
before attempting to build the inference server. To build the
inference server from source, change to the root directory of the
GitHub repo and checkout the release version of the branch that you
want to build (or the master branch if you want to build the
under-development version)::

  $ git checkout r19.07

Then use docker to build::

  $ docker build --pull -t tensorrtserver .

After the build completes follow these steps to run the server and the
example client applications.

#. :ref:`Run the inference server <section-run-tensorrt-inference-server>`.
#. :ref:`Verify that the server is running correct <section-verify-inference-server-status>`.
#. :ref:`Get the example client applications <section-getting-the-client-examples>`.
#. :ref:`Run the image classification example <section-running-the-image-classification-example>`.

.. _section-building-with-cmake:

Building With CMake
-------------------

Make sure you complete the steps in :ref:`section-prerequisites`
before attempting to build the inference server. To build with CMake
you must decide which features of the inference server you want, build
any required dependencies, and the lastly build the TensorRT Inference
Server itself. See :ref:`section-building-the-server-with-cmake` for
details on how to build with CMake.

After the build completes follow these steps to run the server and the
example client applications.

#. :ref:`Run the inference server <section-run-tensorrt-inference-server>`.
#. :ref:`Verify that the server is running correct <section-verify-inference-server-status>`.
#. :ref:`Get the example client applications <section-getting-the-client-examples>`.
#. :ref:`Run the image classification example <section-running-the-image-classification-example>`.

.. _section-run-tensorrt-inference-server:

Run TensorRT Inference Server
-----------------------------

Assuming the example model repository is available in
/full/path/to/example/model/repository, if you build using Docker use
the following command to run the inference server container::

  $ nvidia-docker run --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/example/model/repository:/models <docker image> trtserver --model-repository=/models

Where <docker image> is *nvcr.io/nvidia/tensorrtserver:<xx.yy>-py3* if
you pulled the inference server container from NGC, or is
*tensorrtserver* if you built the inference server from source.

If you built using CMake run the inference server directly on your host system::

    $ trtserver --model-repository=/full/path/to/example/model/repository

In either case, after you start the inference server you will see
output on the console showing the server starting up and loading the
model. When you see output like the following, the inference server is
ready to accept inference requests::

  I0828 23:42:45.635957 1 main.cc:417] Starting endpoints, 'inference:0' listening on
  I0828 23:42:45.649580 1 grpc_server.cc:1730] Started GRPCService at 0.0.0.0:8001
  I0828 23:42:45.649647 1 http_server.cc:1125] Starting HTTPService at 0.0.0.0:8000
  I0828 23:42:45.693758 1 http_server.cc:1139] Starting Metrics Service at 0.0.0.0:8002

For more information, see :ref:`section-running-the-inference-server`.

.. _section-verify-inference-server-status:

Verify Inference Server Is Running Correctly
--------------------------------------------

Use the server’s *Status* endpoint to verify that the server and the
models are ready for inference.  From the host system use curl to
access the HTTP endpoint to request the server status. For example::

  $ curl localhost:8000/api/status
  id: "inference:0"
  version: "0.6.0"
  uptime_ns: 23322988571
  model_status {
    key: "resnet50_netdef"
    value {
      config {
        name: "resnet50_netdef"
        platform: "caffe2_netdef"
      }
      ...
      version_status {
        key: 1
        value {
          ready_state: MODEL_READY
        }
      }
    }
  }
  ready_state: SERVER_READY

The ready_state field should return SERVER_READY to indicate that the
inference server is online, that models are properly loaded, and that
the server is ready to receive inference requests.

For more information, see
:ref:`section-checking-inference-server-status`.

.. _section-getting-the-client-examples:

Getting The Client Examples
---------------------------

The provided Dockerfile.client can be used to build the client
libraries and examples. First change directory to the root of the repo
and checkout the release version of the branch that you want to build
(or the master branch if you want to build the under-development
version). The branch you use for the client build should match the
version of the inference server you are using::

  $ git checkout r19.07

Then use docker to build the C++ client library, C++ and Python
examples, and a Python wheel file for the Python client library::

  $ docker build -t tensorrtserver_client -f Dockerfile.client .

After the build completes, the tensorrtserver_client Docker image will
contain the built client libraries and examples. Run the client image
so that the client examples can access the inference server::

  $ docker run -it --rm --net=host tensorrtserver_client

It is also possible to build the client examples without Docker and
for some platforms pre-compiled client examples are available. For
more information, see
:ref:`section-getting-the-client-libraries-and-examples`.

.. _section-running-the-image-classification-example:

Running The Image Classification Example
----------------------------------------

From within the tensorrtserver_client image, run the example
image-client application to perform image classification using the
example resnet50_netdef from the example model repository.

To send a request for the resnet50_netdef (Caffe2) model from the
example model repository for an image from the /workspace/images directory::

  $ /workspace/install/bin/image_client -m resnet50_netdef -s INCEPTION /workspace/images/mug.jpg
  Request 0, batch size 1
  Image '../images/mug.jpg':
      504 (COFFEE MUG) = 0.723991

The Python version of the application accepts the same command-line
arguments::

  $ python /workspace/install/python/image_client.py -m resnet50_netdef -s INCEPTION /workspace/images/mug.jpg
  Request 0, batch size 1
  Image '../images/mug.jpg':
      504 (COFFEE MUG) = 0.778078556061

For more information, see :ref:`section-image_classification_example`.
