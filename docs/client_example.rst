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

.. _section-client-examples:

Client Examples
===============

The inference server includes a couple of example applications that
show how to use the :ref:`client libraries
<section-client-libraries>`:

* C++ and Python versions of *image\_client*, an example application
  that uses the C++ or Python client library to execute image
  classification models on the TensorRT Inference Server.

* C++ version of *perf\_client*, an application that issues a large
  number of concurrent requests to the inference server to measure
  latency and throughput for a given model. You can use this to
  experiment with different model configuration settings for your
  models.

* A number of simple `C++
  <https://github.com/NVIDIA/tensorrt-inference-server/tree/master/src/clients/c%2B%2B>`_
  and `Python
  <https://github.com/NVIDIA/tensorrt-inference-server/tree/master/src/clients/python>`_
  samples that show various aspects of the inference server. The name
  of these examples begins with *simple_*.

You can also communicate with the inference server by using the
`protoc compiler to generate the GRPC client stub
<https://grpc.io/docs/guides/>`_ in a large number of programming
languages. As an example, *grpc\_image\_client*, is a Python
application that is functionally equivalent to *image\_client* but
that uses a generated GRPC client stub to communicate with the
inference server (instead of the client library).

.. _section-getting-the-client-examples:

Getting the Client Examples
---------------------------

The provided Dockerfile.client and CMake support can be used to build
the examples, or the pre-built examples can be downloaded from GitHub
or a pre-built Docker image containing the client libraries from
`NVIDIA GPU Cloud (NGC) <https://ngc.nvidia.com>`_.

Build Using Dockerfile
^^^^^^^^^^^^^^^^^^^^^^

To build the examples using Docker follow the description in
:ref:`section-client-libraries-build-using-dockerfile`.

After the build completes the tensorrtserver_client docker image will
contain the built client examples, and will also be configured with
all the dependencies required to run those examples within the
container. The easiest way to try the examples described in the
following sections is to run the client image with -\\-net=host so
that the client examples can access the inference server running in
its own container. To be able to use system shared memory you need to run
the client and server image with -\\-ipc=host so that the inference server
can access the system shared memory in the client container. Additionally,
to create system shared memory regions that are larger than 64MB, the
-\\-shm-size=1g flag is needed while running the client image. To be able
to use CUDA shared memory you need to use nvidia-docker instead of Docker
to run the client image. (see :ref:`section-running-the-inference-server`
for more information about running the inference server)::

  $ docker run -it --rm --net=host tensorrtserver_client

In the tensorrtserver_client image you can find the example
executables in /workspace/install/bin, and the
Python examples in /workspace/install/python.

Build Using CMake
^^^^^^^^^^^^^^^^^

To build the examples using CMake follow the description in
:ref:`section-client-libraries-build-using-cmake`.

Ubuntu 16.04 / Ubuntu 18.04
...........................

When the build completes the examples can be found in
trtis-clients/install. To use the examples, you need to include the
path to the client library in environment variable "LD_LIBRARY_PATH",
by default it is
/path/to/tensorrtserver/repo/build/trtis-clients/install/lib. In
addition to that, you also need to install the tensorrtserver Python
package and other packages required by the examples::

  $ pip install trtis-clients/install/python/tensorrtserver-*.whl numpy pillow

Windows 10
..........

When the build completes the examples can be found in
trtis-clients/install. The C++ client examples will not be generated
as those examples have not yet been ported to Windows. However, you
can use the Python examples to test if the build is successful. To use
the Python examples, you need to install the tensorrtserver Python
package and other packages required by the examples::

  > pip install trtis-clients/install/python/tensorrtserver-*.whl numpy pillow

Download From GitHub
^^^^^^^^^^^^^^^^^^^^

To download the examples follow the description in
:ref:`section-client-libraries-download-from-github`.

To use the C++ examples you must install some dependencies. What
dependencies you need to install depends on your OS. For Ubuntu
16.04::

  $ apt-get update
  $ apt-get install curl libcurl3-dev

For Ubuntu 18.04::

  $ apt-get update
  $ apt-get install curl libcurl4-openssl-dev

The Python examples require that you additionally install the wheel
file and some other dependencies::

  $ apt-get install python python-pip
  $ pip install --user --upgrade python/tensorrtserver-*.whl numpy pillow

The C++ image_client example uses OpenCV for image manipulation so for
that example you must install the following::

  $ apt-get install libopencv-dev libopencv-core-dev

Download Docker Image From NGC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To download the Docker image follow the description in
:ref:`section-client-libraries-download-from-ngc`.

The docker image contains the built client examples and will also be
configured with all the dependencies required to run those examples
within the container. The easiest way to try the examples described in
the following sections is to run the client image with -\\-net=host so
that the client examples can access the inference server running in
its own container. To be able to use system shared memory you need to run
the client and server image with -\\-ipc=host so that the inference server
can access the system shared memory in the client container. Additionally,
to create system shared memory regions that are larger than 64MB, the
-\\-shm-size=1g flag is needed while running the client image. To be able
to use CUDA shared memory you need to use nvidia-docker instead of Docker
to run the client image. (see :ref:`section-running-the-inference-server`
for more information about running the inference server)::

  $ docker run -it --rm --net=host nvcr.io/nvidia/tensorrtserver:<xx.yy>-py3-clientsdk

In the image you can find the example executables in
/workspace/install/bin, and the Python examples in
/workspace/install/python.

.. _section-image-classification-example:

Image Classification Example Application
----------------------------------------

The image classification example that uses the C++ client API is
available at `src/clients/c++/examples/image\_client.cc
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/c%2B%2B/examples/image_client.cc>`_. The
Python version of the image classification client is available at
`src/clients/python/image\_client.py
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/python/image_client.py>`_.

To use image\_client (or image\_client.py) you must first have a
running inference server that is serving one or more image
classification models. The image\_client application requires that the
model have a single image input and produce a single classification
output. If you don't have a model repository with image classification
models see :ref:`section-example-model-repository` for instructions on
how to create one.

Follow the instructions in :ref:`section-running-the-inference-server`
to launch the server using the model repository. Once the server is
running you can use the image\_client application to send inference
requests to the server. You can specify a single image or a directory
holding images. Here we send a request for the resnet50_netdef model
from the :ref:`example model repository
<section-example-model-repository>` for an image from the `qa/images
<https://github.com/NVIDIA/tensorrt-inference-server/tree/master/qa/images>`_
directory::

  $ image_client -m resnet50_netdef -s INCEPTION qa/images/mug.jpg
  Request 0, batch size 1
  Image '../qa/images/mug.jpg':
      504 (COFFEE MUG) = 0.723991

The Python version of the application accepts the same command-line
arguments::

  $ python image_client.py -m resnet50_netdef -s INCEPTION qa/images/mug.jpg
  Request 0, batch size 1
  Image '../qa/images/mug.jpg':
      504 (COFFEE MUG) = 0.778078556061

The image\_client and image\_client.py applications use the inference
server client library to talk to the server. By default image\_client
instructs the client library to use HTTP protocol to talk to the
server, but you can use GRPC protocol by providing the \-i flag. You
must also use the \-u flag to point at the GRPC endpoint on the
inference server::

  $ image_client -i grpc -u localhost:8001 -m resnet50_netdef -s INCEPTION qa/images/mug.jpg
  Request 0, batch size 1
  Image '../qa/images/mug.jpg':
      504 (COFFEE MUG) = 0.723991

By default the client prints the most probable classification for the
image. Use the \-c flag to see more classifications::

  $ image_client -m resnet50_netdef -s INCEPTION -c 3 qa/images/mug.jpg
  Request 0, batch size 1
  Image '../qa/images/mug.jpg':
      504 (COFFEE MUG) = 0.723991
      968 (CUP) = 0.270953
      967 (ESPRESSO) = 0.00115996

The \-b flag allows you to send a batch of images for inferencing.
The image\_client application will form the batch from the image or
images that you specified. If the batch is bigger than the number of
images then image\_client will just repeat the images to fill the
batch::

  $ image_client -m resnet50_netdef -s INCEPTION -c 3 -b 2 qa/images/mug.jpg
  Request 0, batch size 2
  Image '../qa/images/mug.jpg':
      504 (COFFEE MUG) = 0.778078556061
      968 (CUP) = 0.213262036443
      967 (ESPRESSO) = 0.00293014757335
  Image '../qa/images/mug.jpg':
      504 (COFFEE MUG) = 0.778078556061
      968 (CUP) = 0.213262036443
      967 (ESPRESSO) = 0.00293014757335

Provide a directory instead of a single image to perform inferencing
on all images in the directory::

  $ image_client -m resnet50_netdef -s INCEPTION -c 3 -b 2 qa/images
  Request 0, batch size 2
  Image '../qa/images/car.jpg':
      817 (SPORTS CAR) = 0.836187
      511 (CONVERTIBLE) = 0.0708251
      751 (RACER) = 0.0597549
  Image '../qa/images/mug.jpg':
      504 (COFFEE MUG) = 0.723991
      968 (CUP) = 0.270953
      967 (ESPRESSO) = 0.00115996
  Request 1, batch size 2
  Image '../qa/images/vulture.jpeg':
      23 (VULTURE) = 0.992326
      8 (HEN) = 0.00231854
      84 (PEACOCK) = 0.00201471
  Image '../qa/images/car.jpg':
      817 (SPORTS CAR) = 0.836187
      511 (CONVERTIBLE) = 0.0708251
      751 (RACER) = 0.0597549

The grpc\_image\_client.py application at available at
`src/clients/python/grpc\_image\_client.py
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/python/grpc_image_client.py>`_
behaves the same as the image\_client except that instead of using the
inference server client library it uses the GRPC generated client
library to communicate with the server.

.. _section-ensemble-image-classification-example:

Ensemble Image Classification Example Application
-------------------------------------------------

In comparison to the image classification example above, this example
uses an ensemble of an image-preprocessing model implemented as a
custom backend and a Caffe2 ResNet50 model. This ensemble allows you
to send the raw image binaries in the request and receive
classification results without preprocessing the images on the
client. The ensemble image classification example that uses the C++
client API is available at `src/clients/c++/examples/ensemble\_image\_client.cc
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/c%2B%2B/examples/ensemble_image_client.cc>`_.
The Python version of the image classification client is available at
`src/clients/python/ensemble\_image\_client.py
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/python/ensemble_image_client.py>`_.

To use ensemble\_image\_client (or ensemble\_image\_client.py) you must first
have a running inference server that is serving the
"preprocess_resnet50_ensemble" model and the models it depends on. The models
are provided in example ensemble model repository see
:ref:`section-example-model-repository` for instructions on how to create one.

Follow the instructions in :ref:`section-running-the-inference-server`
to launch the server using the ensemble model repository. Once the server is
running you can use the ensemble\_image\_client application to send inference
requests to the server. You can specify a single image or a directory
holding images. Here we send a request for the ensemble from the
:ref:`example ensemble model repository <section-example-model-repository>` for
an image from the `qa/images
<https://github.com/NVIDIA/tensorrt-inference-server/tree/master/qa/images>`_
directory::

  $ ensemble_image_client qa/images/mug.jpg
  Image 'qa/images/mug.jpg':
      504 (COFFEE MUG) = 0.723991

The Python version of the application accepts the same command-line
arguments::

  $ python ensemble_image_client.py qa/images/mug.jpg
  Image 'qa/images/mug.jpg':
      504 (COFFEE MUG) = 0.778078556061

Similar to image\_client, by default ensemble\_image\_client
instructs the client library to use HTTP protocol to talk to the
server, but you can use GRPC protocol by providing the \-i flag. You
must also use the \-u flag to point at the GRPC endpoint on the
inference server::

  $ ensemble_image_client -i grpc -u localhost:8001 qa/images/mug.jpg
  Image 'qa/images/mug.jpg':
      504 (COFFEE MUG) = 0.723991

By default the client prints the most probable classification for the
image. Use the \-c flag to see more classifications::

  $ ensemble_image_client -c 3 qa/images/mug.jpg
  Image 'qa/images/mug.jpg':
      504 (COFFEE MUG) = 0.723991
      968 (CUP) = 0.270953
      967 (ESPRESSO) = 0.00115996

Provide a directory instead of a single image to perform inferencing
on all images in the directory. If the number of images exceeds the maximum
batch size of the ensemble, only the images within the maximum batch size
will be sent::

  $ ensemble_image_client -c 3 qa/images
  Image 'qa/images/car.jpg':
      817 (SPORTS CAR) = 0.836187
      511 (CONVERTIBLE) = 0.0708251
      751 (RACER) = 0.0597549
  Image 'qa/images/mug.jpg':
      504 (COFFEE MUG) = 0.723991
      968 (CUP) = 0.270953
      967 (ESPRESSO) = 0.00115996
  Image 'qa/images/vulture.jpeg':
      23 (VULTURE) = 0.992326
      8 (HEN) = 0.00231854
      84 (PEACOCK) = 0.00201471

.. _section-performance-example:

Performance Measurement Application
-----------------------------------

The perf\_client application located at `src/clients/c++/perf\_client
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/c%2B%2B/perf_client>`_
uses the C++ client API to send concurrent requests to the server to
measure latency and inferences-per-second under varying client
loads. See the :ref:`section-perf-client` for a full description.
