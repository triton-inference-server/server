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

.. _section-building:

Building
========

The TensorRT Inference Server can be built in two ways:

* Build using Docker and the TensorFlow and PyTorch containers from
  `NVIDIA GPU Cloud (NGC) <https://ngc.nvidia.com>`_. Before building
  you must install Docker and nvidia-docker and login to the NGC
  registry by following the instructions in
  :ref:`section-installing-prebuilt-containers`.

* Build using CMake and the dependencies (for example, TensorFlow or
  TensorRT library) that you build or install yourself.

.. _section-building-the-server-with-docker:

Building the Server with Docker
-------------------------------

To build a release version of the TensorRT Inference Server container,
change directory to the root of the repo and checkout the release
version of the branch that you want to build (or the master branch if
you want to build the under-development version)::

  $ git checkout r19.08

Then use docker to build::

  $ docker build --pull -t tensorrtserver .

Incremental Builds with Docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For typical development you will want to run the *build* container
with your local repo’s source files mounted so that your local changes
can be incrementally built. This is done by first building the
*tensorrtserver_build* container::

  $ docker build --pull -t tensorrtserver_build --target trtserver_build .

By mounting /path/to/tensorrtserver/src into the container at
/workspace/src, changes to your local repo will be reflected in the
container::

  $ nvidia-docker run -it --rm -v/path/to/tensorrtserver/src:/workspace/src tensorrtserver_build

Within the container you can perform an incremental server build
with::

  # cd /workspace/builddir
  # make -j16 trtis

When the build completes the binary, libraries and headers can be
found in trtis/install. To overwrite the existing versions::

  # cp trtis/install/bin/trtserver /opt/tensorrtserver/bin/.
  # cp trtis/install/lib/libtrtserver.so /opt/tensorrtserver/lib/.

You can reconfigure the build by running *cmake* as described in
:ref:`section-building-the-server-with-cmake`.

.. _section-building-the-server-with-cmake:

Building the Server with CMake
------------------------------

To build a release version of the TensorRT Inference Server with
CMake, change directory to the root of the repo and checkout the
release version of the branch that you want to build (or the master
branch if you want to build the under-development version)::

  $ git checkout r19.08

Next you must build or install each framework backend you want to
enable in the inference server, configure the inference server to
enable the desired features, and finally build the server.

.. _section-cmake-dependencies:

Dependencies
^^^^^^^^^^^^

To include GPU support in the inference server you must install the
necessary CUDA libraries. Similarly, to include support for a
particular framework backend, you must build the appropriate libraries
for that framework and make them available to the inference server
build. In general, the Dockerfile build steps guide how each of these
frameworks can be built for use in the interence server.

CUDA, cuBLAS, cuDNN
...................

For the inference server to support NVIDIA GPUs you must install CUDA,
cuBLAS and cuDNN. These libraries must be installed on system include
and library paths so that they are available for the CMake build. The
version of the libraries used in the Dockerfile build can be found in
the `Framework Containers Support Matrix
<https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html>`_.

For a given version of the inference server you can attempt to build
with non-supported versions of the libraries but you may have build or
execution issues since non-supported versions are not tested.

Once you have CUDA, cuBLAS and cuDNN installed you can enable GPUs
with the CMake option -DTRTIS_ENABLE_GPU=ON as described below.

TensorRT
........

The TensorRT includes and libraries must be installed on system
include and library paths so that they are available for the CMake
build. The version of TensorRT used in the Dockerfile build can be
found in the `Framework Containers Support Matrix
<https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html>`_.

For a given version of the inference server you can attempt to build
with non-supported versions of TensorRT but you may have build or
execution issues since non-supported versions are not tested.

Once you have TensorRT installed you can enable the TensorRT backend
in the inference server with the CMake option
-DTRTIS_ENABLE_TENSORRT=ON as described below. You must also specify
-DTRTIS_ENABLE_GPU=ON because TensorRT requires GPU support.

TensorFlow
..........

The version of TensorFlow used in the Dockerfile build can be found in
the `Framework Containers Support Matrix
<https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html>`_.
The trtserver_tf section of the Dockerfile shows how to build the
required TensorFlow libary from the `NGC <https://ngc.nvidia.com>`_
TensorFlow container.

You can build and install a different version of the TensorFlow
library but you must build with the equivalent options indicated by
the patches used in the Dockerfile, and you must include
tensorflow_backend_tf.cc and tensorflow_backend_tf.h. The patch to
tensorflow/BUILD and the build options shown in nvbuildopts cause
TensorFlow backend to be built into a single library,
libtensorflow_cc.so, that includes all the functionality required by
the inference server.

Once you have the TensorFlow library built and installed you can
enable the TensorFlow backend in the inference server with the CMake
option -DTRTIS_ENABLE_TENSORRT=ON as described below. You must also
specify -DTRTIS_ENABLE_GPU=ON because TensorRT requires GPU support.

You can install the TensorFlow library in a system library path or you
can specify the path with the CMake option
TRTIS_EXTRA_LIB_PATHS. Multiple paths can be specified by separating
them with a semicolon, for example,
-DTRTIS_EXTRA_LIB_PATHS="/path/a;/path/b".

ONNX Runtime
............

The version of the ONNX Runtime used in the Dockerfile build can be
found in the trtserver_onnx section of the Dockerfile. That section
also details the steps that can be used to build the backend. You can
attempt to build a different version of the ONNX Runtime or use a
different build process but you may have build or execution issues.

Your build should produce the ONNX Runtime library, libonnxruntime.so.
You can enable the ONNX Runtime backend in the inference server with
the CMake option -DTRTIS_ENABLE_ONNXRUNTIME=ON as described below.

You can install the library in a system library path or you can
specify the path with the CMake option TRTIS_EXTRA_LIB_PATHS. Multiple
paths can be specified by separating them with a semicolon, for
example, -DTRTIS_EXTRA_LIB_PATHS="/path/a;/path/b".

You must also provide the path to the ONNX Runtime headers using the
-DTRTIS_ONNXRUNTIME_INCLUDE_PATHS option. Multiple paths can be
specified by separating them with a semicolon.

PyTorch and Caffe2
..................

The version of PyTorch and Caffe2 used in the Dockerfile build can be
found in the `Framework Containers Support Matrix
<https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html>`_.
The trtserver_caffe2 section of the Dockerfile shows how to build the
required PyTorch and Caffe2 libaries from the `NGC
<https://ngc.nvidia.com>`_ PyTorch container.

You can build and install a different version of the libraries but if
you want to enable the Caffe2 backend you must include
netdef_backend_c2.cc and netdef_backend.c2.h in the build, as shown in
the Dockerfile.

Once you have the libraries built and installed you can enable the
PyTorch backend in the inference server with the CMake option
-DTRTIS_ENABLE_PYTORCH=ON and the Caffe2 backend with
-DTRTIS_ENABLE_CAFFE2=ON as described below.

You can install the PyTorch library, libtorch.so, and all the required
Caffe2 libraries (see Dockerfile) in a system library path or you can
specify the path with the CMake option TRTIS_EXTRA_LIB_PATHS. Multiple
paths can be specified by separating them with a semicolon, for
example, -DTRTIS_EXTRA_LIB_PATHS="/path/a;/path/b".

For the PyTorch backend you must also provide the path to the PyTorch
headers using the -DTRTIS_PYTORCH_INCLUDE_PATHS option. Multiple paths
can be specified by separating them with a semicolon.

Configure Inference Server
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use cmake to configure the TensorRT Inference Server::

  $ mkdir builddir
  $ cd builddir
  $ cmake -D<option0> ... -D<optionn> ../build

The following options are used to enable and disable the different
backends. To enable a backend set the corresponding option to ON, for
example -DTRTIS_ENABLE_TENSORRT=ON. To disable a backend set the
corresponding option to OFF, for example -DTRTIS_ENABLE_TENSORRT=OFF.
By default no backends are enabled. See the section on
:ref:`dependencies<section-cmake-dependencies>` for information on
additional requirements for enabling a backend.

* **TRTIS_ENABLE_TENSORRT**: Use -DTRTIS_ENABLE_TENSORRT=ON to enable
  the TensorRT backend. The TensorRT libraries must be on your library
  path or you must add the path to TRTIS_EXTRA_LIB_PATHS.

* **TRTIS_ENABLE_TENSORFLOW**: Use -DTRTIS_ENABLE_TENSORFLOW=ON to
  enable the TensorFlow backend. The TensorFlow library
  libtensorflow_cc.so must be built as described above and must be on
  your library path or you must add the path to TRTIS_EXTRA_LIB_PATHS.

* **TRTIS_ENABLE_ONNXRUNTIME**: Use -DTRTIS_ENABLE_ONNXRUNTIME=ON to
  enable the OnnxRuntime backend. The library libonnxruntime.so must
  be built as described above and must be on your library path or you
  must add the path to TRTIS_EXTRA_LIB_PATHS.

* **TRTIS_ENABLE_PYTORCH**: Use -DTRTIS_ENABLE_PYTORCH=ON to enable
  the PyTorch backend. The library libtorch.so must be built as
  described above and must be on your library path or you must add the
  path to TRTIS_EXTRA_LIB_PATHS.

* **TRTIS_ENABLE_CAFFE2**: Use -DTRTIS_ENABLE_CAFFE2=ON to enable the
  Caffe2 backend. The library libcaffe2.so and all the other required
  libraries must be built as described above and must be on your
  library path or you must add the path to TRTIS_EXTRA_LIB_PATHS.

* **TRTIS_ENABLE_CUSTOM**: Use -DTRTIS_ENABLE_CUSTOM=ON to enable
  support for custom backends. See
  :ref:`section-building-a-custom-backend` for information on how to
  build a custom backend.

These additional options may be specified:

* **TRTIS_ENABLE_METRICS**: By default the inference server reports
  :ref:`Prometheus metrics<section-metrics>` on an HTTP endpoint. Use
  -DTRTIS_ENABLE_METRICS=OFF to disable.

* **TRTIS_ENABLE_GPU**: By default the inference server supports
  NVIDIA GPUs. Use -DTRTIS_ENABLE_GPU=OFF to disable GPU support. When
  GPUs are disable the inference server will :ref:`run models on CPU
  when possible<section-running-the-inference-server-without-gpu>`.

* **TRTIS_MIN_COMPUTE_CAPABILITY**: By default, the inference server
  supports NVIDIA GPUs with CUDA compute capability 6.0 or higher. If
  all framework backends included in the inference server are built to
  support a lower compute capability, then TRTIS can be built to support
  that lower compute capability by setting -DTRTIS_MIN_COMPUTE_CAPABILITY
  appropriately. The setting is ignored if -DTRTIS_ENABLE_GPU=OFF.

Build Inference Server
^^^^^^^^^^^^^^^^^^^^^^

After configuring, build the inference server with make::

  $ cd builddir
  $ make -j16 trtis

When the build completes the binary, libraries and headers can be
found in trtis/install.

.. _section-building-a-custom-backend:

Building A Custom Backend
-------------------------

The source repository contains several example custom backends in the
`src/custom directory
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/custom>`_.
These custom backends are built using CMake::

  $ mkdir builddir
  $ cd builddir
  $ cmake ../build
  $ make -j16 trtis-custom-backends

When the build completes the custom backend libraries can be found in
trtis-custom-backends/install.

A custom backend is not built-into the inference server. Instead it is
built as a separate shared library that the inference server
dynamically loads when the model repository contains a model that uses
that custom backend. There are a couple of ways you can build your
custom backend into a shared libary, as described in the following
sections.

Build Using CMake
^^^^^^^^^^^^^^^^^

One way to build your own custom backend is to use the inference
server's CMake build. Simply copy and modify one of the existing
example custom backends and then build your backend using CMake. You
can then use the resulting shared library in your model repository as
described in :ref:`section-custom-backends`.

Build Using Custom Backend SDK
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The custom backend SDK includes all the header files you need to build
your custom backend as well as a static library which provides all the
model configuration and protobuf utility functions you will need. You
can either build the custom backend SDK yourself using
Dockerfile.custombackend::

  docker build -t tensorrtserver_cbe -f Dockerfile.custombackend .

Or you can download a pre-build version of the SDK from the `GitHub
release page
<https://github.com/NVIDIA/tensorrt-inference-server/releases>`_
corresponding to the release you are interested in. The custom backend
SDK is found in the "Assets" section of the release page in a tar file
named after the version of the release and the OS, for example,
v1.2.0_ubuntu1604.custombackend.tar.gz.

Once you have the SDK you can use the include/ directory and static
library when you compile your custom backend source code. For example,
the SDK includes the source for the *param* custom backend in
src/param.cc. You can create a custom backend from that source using
the following command::

  g++ -fpic -shared -std=c++11 -o libparam.so custom-backend-sdk/src/param.cc -Icustom-backend-sdk/include custom-backend-sdk/lib/libcustombackend.a

Using the Custom Instance Wrapper Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The custom backend SDK provides a `CustomInstance Class 
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/custom/sdk/custom_instance.h>`_. 
The CustomInstance class is a C++ wrapper class that abstracts away the 
backend C-API for ease of use. All of the example custom backends in 
`src/custom directory
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/custom>`_
derive from the CustomInstance class and can be referenced for usage.

Building the Client Libraries and Examples
------------------------------------------

The provided Dockerfile.client and CMake support can be used to build
the client libraries and examples.

.. include:: client.rst
   :start-after: build-client-begin-marker-do-not-remove
   :end-before: build-client-end-marker-do-not-remove

Building the Documentation
--------------------------

The inference server documentation is found in the docs/ directory and
is based on `Sphinx <http://www.sphinx-doc.org>`_. `Doxygen
<http://www.doxygen.org/>`_ integrated with `Exhale
<https://github.com/svenevs/exhale>`_ is used for C++ API
docuementation.

To build the docs install the required dependencies::

  $ apt-get update
  $ apt-get install -y --no-install-recommends doxygen
  $ pip install --upgrade sphinx sphinx-rtd-theme nbsphinx exhale

To get the Python client library API docs the TensorRT Inference
Server Python package must be installed::

  $ pip install --upgrade tensorrtserver-*.whl

Then use Sphinx to build the documentation into the build/html
directory::

  $ cd docs
  $ make clean html
