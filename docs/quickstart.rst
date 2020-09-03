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

.. _section-quickstart:

Quickstart
==========

The Triton Inference Server is available in two ways:

* As a pre-built Docker container available from the `NVIDIA GPU Cloud
  (NGC) <https://ngc.nvidia.com>`_. For more information, see
  :ref:`section-quickstart-using-a-prebuilt-docker-container`.

* As buildable source code located in GitHub. You can :ref:`build your
  own container using Docker<section-quickstart-building-with-docker>` or you can
  :ref:`build using CMake<section-quickstart-building-with-cmake>`.

.. _section-quickstart-prerequisites:

Prerequisites
-------------

Regardless of which method you choose (starting with a pre-built
container from NGC or building from source), you must perform the
following prerequisite steps:

* Clone the Triton Inference Server GitHub repo. Even if you choose to
  get the pre-built Triton from NGC, you need the GitHub repo for the
  example model repository. Go to
  https://github.com/triton-inference-server/server and then select
  the *clone* or *download* drop down button. After cloning the repo
  be sure to select the r<xx.yy> release branch that corresponds to
  the version of Triton you want to use::

  $ git checkout r20.08

* Create a model repository containing one or more models that you
  want Triton to serve. An example model repository is included in the
  docs/examples/model_repository directory of the GitHub repo. Before
  using the repository, you must fetch any missing model definition
  files from their public model zoos via the provided
  docs/examples/fetch_models.sh script::

  $ cd docs/examples
  $ ./fetch_models.sh

If you are starting with a pre-built NGC container perform these
additional steps:

* Install Docker and nvidia-docker.  For DGX users, see `Preparing to
  use NVIDIA Containers
  <http://docs.nvidia.com/deeplearning/dgx/preparing-containers/index.html>`_.
  For users other than DGX, see the `nvidia-docker installation
  documentation <https://github.com/NVIDIA/nvidia-docker>`_.

.. _section-quickstart-using-a-prebuilt-docker-container:

Using A Prebuilt Docker Container
---------------------------------

Use docker pull to get the Triton Inference Server container from
NGC::

  $ docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3

Where <xx.yy> is the version of Triton that you want to pull. Once you
have the container follow these steps to run Triton and the example
client applications.

#. :ref:`Run Triton <section-quickstart-run-triton-inference-server>`.
#. :ref:`Verify that Triton is running correct <section-quickstart-verify-inference-server-status>`.
#. :ref:`Get the example client applications <section-quickstart-getting-the-examples>`.
#. :ref:`Run the image classification example <section-quickstart-running-the-image-classification-example>`.

.. _section-quickstart-building-with-docker:

Building With Docker
--------------------

Make sure you complete the steps in
:ref:`section-quickstart-prerequisites` before attempting to build
Triton. To build Triton from source, change to the root directory of
the GitHub repo and checkout the release version of the branch that
you want to build (or the master branch if you want to build the
under-development version)::

  $ git checkout r20.08

Then use docker to build::

  $ docker build --pull -t tritonserver .

After the build completes follow these steps to run Triton and the
example client applications.

#. :ref:`Run Triton <section-quickstart-run-triton-inference-server>`.
#. :ref:`Verify that Triton is running correct <section-quickstart-verify-inference-server-status>`.
#. :ref:`Get the example client applications <section-quickstart-getting-the-examples>`.
#. :ref:`Run the image classification example <section-quickstart-running-the-image-classification-example>`.

.. _section-quickstart-building-with-cmake:

Building With CMake
-------------------

Make sure you complete the steps in
:ref:`section-quickstart-prerequisites` before attempting to build
Triton. To build with CMake you must decide which features of Triton
you want, build any required dependencies, and the lastly build the
Triton itself. See :ref:`section-building-the-server-with-cmake` for
details on how to build with CMake.

After the build completes follow these steps to run Triton and the
example client applications.

#. :ref:`Run Triton <section-quickstart-run-triton-inference-server>`.
#. :ref:`Verify that Triton is running correct <section-quickstart-verify-inference-server-status>`.
#. :ref:`Get the example client applications <section-quickstart-getting-the-examples>`.
#. :ref:`Run the image classification example <section-quickstart-running-the-image-classification-example>`.

.. _section-quickstart-run-triton-inference-server:

Run Triton Inference Server
---------------------------

Assuming the example model repository is available in
/full/path/to/example/model/repository, if you built using Docker use
the following command to run the Triton container::

  $ docker run --gpus=1 --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/example/model/repository:/models <docker image> tritonserver --model-repository=/models

Where <docker image> is *nvcr.io/nvidia/tritonserver:<xx.yy>-py3* if
you pulled the Triton container from NGC, or is *tritonserver* if you
built Triton from source.

If you built using CMake run Triton directly on your host system::

    $ tritonserver --model-repository=/full/path/to/example/model/repository

In either case, after you start Triton you will see output on the
console showing the server starting up and loading the model. When you
see output like the following, Triton is ready to accept inference
requests::

  I0828 23:42:45.635957 1 main.cc:417] Starting endpoints, 'inference:0' listening on
  I0828 23:42:45.649580 1 grpc_server.cc:1730] Started GRPCInferenceService at 0.0.0.0:8001
  I0828 23:42:45.649647 1 http_server.cc:1125] Started HTTPService at 0.0.0.0:8000
  I0828 23:42:45.693758 1 http_server.cc:1139] Started Metrics Service at 0.0.0.0:8002

For more information, see :ref:`section-running-triton`.

.. _section-quickstart-verify-inference-server-status:

Verify Triton Is Running Correctly
----------------------------------

Use Triton’s *ready* endpoint to verify that the server and the models
are ready for inference. From the host system use curl to access the
HTTP endpoint that indicates server status. For example::

  $ curl -v localhost:8000/v2/health/ready
  ...
  < HTTP/1.1 200 OK
  < Content-Length: 0
  < Content-Type: text/plain

The HTTP request returns status 200 if Triton is ready and non-200 if
it is not ready.  For more information, see
:ref:`section-checking-inference-server-status`.

.. _section-quickstart-getting-the-examples:

Getting The Client Examples
---------------------------

Use docker pull to get the client libraries and examples container
from NGC::

  $ docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3-clientsdk

Where <xx.yy> is the version that you want to pull. Run the client
image::

  $ docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:<xx.yy>-py3-clientsdk

It is also possible to build the client examples with or without
Docker. For more information, see
:ref:`section-getting-the-client-examples`.

.. _section-quickstart-running-the-image-classification-example:

Running The Image Classification Example
----------------------------------------

From within the nvcr.io/nvidia/tritonserver:<xx.yy>-py3-clientsdk
image, run the example image-client application to perform image
classification using the example resnet50_netdef.

To send a request for the resnet50_netdef (Caffe2) model from the
example model repository for an image from the /workspace/images
directory::

  $ /workspace/install/bin/image_client -m resnet50_netdef -s INCEPTION /workspace/images/mug.jpg
  Request 0, batch size 1
  Image 'images/mug.jpg':
      0.723992 (504) = COFFEE MUG

The Python version of image_client accepts the same command-line
arguments::

  $ python /workspace/install/python/image_client.py -m resnet50_netdef -s INCEPTION /workspace/images/mug.jpg
  Request 1, batch size 1
      0.777365 (504) = COFFEE MUG

For more information, see :ref:`section-image-classification-example`.
