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

Running the Server
==================

For best performance the TensorRT Inference Server should be run on a
system that contains Docker, nvidia-docker, CUDA and one or more
supported GPUs, as explained in
:ref:`section-running-the-inference-server`. The inference server can
also be run on non-CUDA, non-GPU systems as described in
:ref:`section-running-the-inference-server-without-gpu`.

If you :ref:`build the inference server outside of Docker
<section-building-the-server-with-cmake>`, you can then run the
inference server without Docker, as explained in
:ref:`section-running-the-inference-server-without-docker`.

.. _section-example-model-repository:

Example Model Repository
------------------------

Before running the TensorRT Inference Server, you must first set up a
model repository containing the models that the server will make
available for inferencing.

An example model repository containing a Caffe2 ResNet50, a TensorFlow
Inception model, an ONNX densenet model, a simple TensorFlow GraphDef
model (used by the :ref:`simple_client example <section-client-api>`),
and a simple TensorFlow GraphDef model using String tensors (used by
the :ref:`simple_string_client example <section-client-api>`) are
provided in the `docs/examples/model_repository
<https://github.com/NVIDIA/tensorrt-inference-server/tree/master/docs/examples/model_repository>`_
directory. Before using the example model repository you must fetch
any missing model definition files from their public model zoos. Be
sure to checkout the release version of the branch that corresponds to
the server you are using (or the master branch if you are using a
server build from master)::

  $ git checkout r19.08
  $ cd docs/examples
  $ ./fetch_models.sh

An example ensemble model repository is also provided in the
`docs/examples/ensemble_model_repository
<https://github.com/NVIDIA/tensorrt-inference-server/tree/master/docs/examples/ensemble_model_repository>`_
directory. It contains a custom image preprocess model, Caffe2
ResNet50, and an ensemble (used by the :ref:`ensemble_image_client
example <section-ensemble_image_classification_example>`).

Before using the example ensemble model repository, in addition to
fetching public model definition files as mentioned above, you must
build the model definition file for the custom image preprocess model
(see :ref:`section-building-a-custom-backend` for instructions on how
to build it). Also note that although ensemble models are fully
specified in their model configuration, empty version directories are
required for them to be recognized as valid model directories::

  $ cd docs/examples
  $ mkdir -p ensemble_model_repository/preprocess_resnet50_ensemble/1

.. _section-running-the-inference-server:

Running The Inference Server
----------------------------

Before running the inference server, you must first set up a model
repository containing the models that the server will make available
for inferencing. Section :ref:`section-model-repository` describes how
to create your own model repository. You can also use
:ref:`section-example-model-repository` to set up an example model
repository.

Assuming the sample model repository is available in
/path/to/model/repository, the following command runs the container
you pulled from NGC or built locally::

  $ nvidia-docker run --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v/path/to/model/repository:/models <tensorrtserver image name> trtserver --model-repository=/models

Where *<tensorrtserver image name>* will be something like
**nvcr.io/nvidia/tensorrtserver:19.08-py3** if you :ref:`pulled the
container from the NGC registry
<section-installing-prebuilt-containers>`, or **tensorrtserver** if
you :ref:`built it from source <section-building>`.

The nvidia-docker -v option maps /path/to/model/repository on the host
into the container at /models, and the -\\-model-repository option to the
server is used to point to /models as the model repository.

The -p flags expose the container ports where the inference server
listens for HTTP requests (port 8000), listens for GRPC requests (port
8001), and reports Prometheus metrics (port 8002).

The -\\-shm-size and -\\-ulimit flags are recommended to improve the
server's performance. For -\\-shm-size the minimum recommended size is
1g but smaller or larger sizes may be used depending on the number and
size of models being served.

For more information on the Prometheus metrics provided by the
inference server see :ref:`section-metrics`.

.. _section-running-the-inference-server-without-gpu:

Running The Inference Server On A System Without A GPU
------------------------------------------------------

On a system without GPUs, the inference server should be run using
docker instead of nvidia-docker, but is otherwise identical to what is
described in :ref:`section-running-the-inference-server`::

  $ docker run --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v/path/to/model/repository:/models <tensorrtserver image name> trtserver --model-repository=/models

Because a GPU is not available, the inference server will be unable to
load any model configuration that requires a GPU or that specifies a
GPU instance by an :ref:`instance-group <section-instance-groups>`
configuration.

.. _section-running-the-inference-server-without-docker:

Running The Inference Server Without Docker
-------------------------------------------

After :ref:`building the inference server outside of Docker
<section-building-the-server-with-cmake>`, the *trtserver* binary will
be in builddir/trtis/install/bin and the required shared libraries
will be in builddir/trtis/install/lib. To run make sure that
builddir/trtis/install/lib is on your library path (for example, by
adding it to LD_LIBRARY_PATH), and then execute *trtserver* with the
desired arguments::

  $ builddir/trtis/install/bin/trtserver --model-repository=/models

.. _section-checking-inference-server-status:

Checking Inference Server Status
--------------------------------

The simplest way to verify that the inference server is running
correctly is to use the Status API to query the server’s status. From
the host system use *curl* to access the HTTP endpoint to request
server status. The response is protobuf text showing the status for
the server and for each model being served, for example::

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

This status shows configuration information as well as indicating that
version 1 of the resnet50_netdef model is MODEL_READY. This means that
the server is ready to accept inferencing requests for version 1 of
that model. A model version ready_state will show up as
MODEL_UNAVAILABLE if the model failed to load for some reason.
