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

.. _section-building:

Building
========

The Triton Inference Server and the client libraries and examples can
be built using either Docker or CMake. The procedure for each is
different and is detailed in the corresponding sections below.

Building Triton
---------------

Triton can be built in two ways:

* Build using Docker and the TensorFlow and PyTorch containers from
  `NVIDIA GPU Cloud (NGC) <https://ngc.nvidia.com>`_. Before building
  you must install Docker.

* Build using CMake and the dependencies (for example, TensorFlow or
  TensorRT library) that you build or install yourself.

.. _section-building-the-server-with-docker:

Building Triton with Docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build a release version of the Triton container, change directory
to the root of the repo and checkout the release version of the branch
that you want to build (or the master branch if you want to build the
under-development version)::

  $ git checkout r20.08

Then use docker to build::

  $ docker build --pull -t tritonserver .

Incremental Builds with Docker
..............................

For typical development you will want to run the *build* container
with your local repo’s source files mounted so that your local changes
can be incrementally built. This is done by first building the
*tritonserver_build* container::

  $ docker build --pull -t tritonserver_build --target tritonserver_build .

By mounting /path/to/tritonserver/src into the container at
/workspace/src, changes to your local repo will be reflected in the
container::

  $ docker run -it --rm -v/path/to/tritonserver/src:/workspace/src tritonserver_build

Within the container you can perform an incremental server build
with::

  # cd /workspace/builddir
  # make -j16 server

When the build completes the binary, libraries and headers can be
found in server/install. To overwrite the existing versions::

  # cp server/install/bin/tritonserver /opt/tritonserver/bin/.
  # cp server/install/lib/libtritonserver.so /opt/tritonserver/lib/.

You can reconfigure the build by running *cmake* as described in
:ref:`section-building-the-server-with-cmake`.

.. _section-building-the-server-with-cmake:

Building Triton with CMake
^^^^^^^^^^^^^^^^^^^^^^^^^^

To build a release version of Triton with CMake, change directory to
the root of the repo and checkout the release version of the branch
that you want to build (or the master branch if you want to build the
under-development version)::

  $ git checkout r20.08

Next you must build or install each framework backend you want to
enable in Triton, configure the build to enable the desired features,
and finally build Triton.

.. _section-cmake-dependencies:

Dependencies
............

To include GPU support in Triton you must install the necessary CUDA
libraries. Similarly, to include support for a particular framework
backend, you must build the appropriate libraries for that framework
and make them available to the Triton build. In general, the
Dockerfile build steps guide how each of these frameworks can be built
for use in Triton.

CUDA, cuBLAS, cuDNN
~~~~~~~~~~~~~~~~~~~

For Triton to support NVIDIA GPUs you must install CUDA, cuBLAS and
cuDNN. These libraries must be installed on system include and library
paths so that they are available for the CMake build. The version of
the libraries used in the Dockerfile build can be found in the
`Framework Containers Support Matrix
<https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html>`_.

For a given version of Triton you can attempt to build with
non-supported versions of the libraries but you may have build or
execution issues since non-supported versions are not tested.

Once you have CUDA, cuBLAS and cuDNN installed you can enable GPUs
with the CMake option -DTRITON_ENABLE_GPU=ON as described below.

TensorRT
~~~~~~~~

The TensorRT includes and libraries must be installed on system
include and library paths so that they are available for the CMake
build. The version of TensorRT used in the Dockerfile build can be
found in the `Framework Containers Support Matrix
<https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html>`_.

For a given version of Triton you can attempt to build with
non-supported versions of TensorRT but you may have build or execution
issues since non-supported versions are not tested.

Once you have TensorRT installed you can enable the TensorRT backend
in Triton with the CMake option -DTRITON_ENABLE_TENSORRT=ON as
described below. You must also specify -DTRITON_ENABLE_GPU=ON because
TensorRT requires GPU support.

TensorFlow
~~~~~~~~~~

The version of TensorFlow used in the Dockerfile build can be found in
the `Framework Containers Support Matrix
<https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html>`_.
The tritonserver_tf section of the Dockerfile shows the required
TensorFlow V1 container pulled from `NGC <https://ngc.nvidia.com>`_.

You can modify and rebuild this TensorFlow container to generate the
libtensorflow_trtis.so shared library needed by the Triton build. For
example, in the TensorFlow container
/workspace/docker-examples/Dockerfile.customtensorflow shows a
Dockerfile that applies a patch to TensorFlow and then rebuilds. For
Triton you need to replace the nvbuild commands in that file with::

  RUN ./nvbuild.sh --python3.6 --trtis

In the newly build container the required TensorFlow library is
/usr/local/lib/tensorflow/libtensorflow_trtis.so.1. On your build
system you must place libtensorflow_trtis.so.1 in a system library
path or you can specify the path with the CMake option
TRITON_EXTRA_LIB_PATHS. Multiple paths can be specified by separating
them with a semicolon, for example,
-DTRITON_EXTRA_LIB_PATHS="/path/a;/path/b". Also create a soft link to
the library as follows::

  ln -s libtensorflow_trtis.so.1 libtensorflow_trtis.so

Lastly, you must enable the TensorFlow backend in the inference server
with the CMake option -DTRITON_ENABLE_TENSORFLOW=ON as described below.

ONNX Runtime
~~~~~~~~~~~~

The version of the ONNX Runtime used in the Dockerfile build can be
found in the tritonserver_onnx section of the Dockerfile. That section
also details the steps that can be used to build the backend. You can
attempt to build a different version of the ONNX Runtime or use a
different build process but you may have build or execution issues.

Your build should produce the ONNX Runtime library, libonnxruntime.so.
You can enable the ONNX Runtime backend in Triton with the CMake
option -DTRITON_ENABLE_ONNXRUNTIME=ON as described below. If you want
to enable TensorRT within the ONNX Runtime you must also specify the
CMake option TRITON_ENABLE_ONNXRUNTIME_TENSORRT=ON and provide the
necessary TensorRT dependencies. If you want to enable OpenVino within
the ONNX Runtime you must also specify the CMake option
TRITON_ENABLE_ONNXRUNTIME_OPENVINO=ON and provide the necessary
OpenVino dependencies.

You can install the library in a system library path or you can
specify the path with the CMake option TRITON_EXTRA_LIB_PATHS. Multiple
paths can be specified by separating them with a semicolon, for
example, -DTRITON_EXTRA_LIB_PATHS="/path/a;/path/b".

You must also provide the path to the ONNX Runtime headers using the
-DTRITON_ONNXRUNTIME_INCLUDE_PATHS option. Multiple paths can be
specified by separating them with a semicolon.

PyTorch and Caffe2
~~~~~~~~~~~~~~~~~~

The version of PyTorch and Caffe2 used in the Dockerfile build can be
found in the `Framework Containers Support Matrix
<https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html>`_.
The tritonserver_pytorch section of the Dockerfile shows how to build
the required PyTorch and Caffe2 libraries from the `NGC
<https://ngc.nvidia.com>`_ PyTorch container.

You can build and install a different version of the libraries but if
you want to enable the Caffe2 backend you must include
netdef_backend_c2.cc and netdef_backend.c2.h in the build, as shown in
the Dockerfile.

Once you have the libraries built and installed you can enable the
PyTorch backend in Triton with the CMake option
-DTRITON_ENABLE_PYTORCH=ON and the Caffe2 backend with
-DTRITON_ENABLE_CAFFE2=ON as described below.

You can install the PyTorch library, libtorch.so, and all the required
Caffe2 libraries (see Dockerfile) in a system library path or you can
specify the path with the CMake option TRITON_EXTRA_LIB_PATHS. Multiple
paths can be specified by separating them with a semicolon, for
example, -DTRITON_EXTRA_LIB_PATHS="/path/a;/path/b".

For the PyTorch backend you must also provide the path to the PyTorch
and Torchvision headers using the -DTRITON_PYTORCH_INCLUDE_PATHS option.
Multiple paths can be specified by separating them with a semicolon.

Configure Triton Build
......................

Use cmake to configure the Triton build::

  $ mkdir builddir
  $ cd builddir
  $ cmake -D<option0> ... -D<optionn> ../build

The following options are used to enable and disable the different
backends. To enable a backend set the corresponding option to ON, for
example -DTRITON_ENABLE_TENSORRT=ON. To disable a backend set the
corresponding option to OFF, for example -DTRITON_ENABLE_TENSORRT=OFF.
By default no backends are enabled. See the section on
:ref:`dependencies<section-cmake-dependencies>` for information on
additional requirements for enabling a backend.

* **TRITON_ENABLE_TENSORRT**: Use -DTRITON_ENABLE_TENSORRT=ON to enable
  the TensorRT backend. The TensorRT libraries must be on your library
  path or you must add the path to TRITON_EXTRA_LIB_PATHS.

* **TRITON_ENABLE_TENSORFLOW**: Use -DTRITON_ENABLE_TENSORFLOW=ON to
  enable the TensorFlow backend. The TensorFlow library
  libtensorflow_cc.so must be built as described above and must be on
  your library path or you must add the path to TRITON_EXTRA_LIB_PATHS.

* **TRITON_ENABLE_ONNXRUNTIME**: Use -DTRITON_ENABLE_ONNXRUNTIME=ON to
  enable the OnnxRuntime backend. The library libonnxruntime.so must
  be built as described above and must be on your library path or you
  must add the path to TRITON_EXTRA_LIB_PATHS.

* **TRITON_ENABLE_PYTORCH**: Use -DTRITON_ENABLE_PYTORCH=ON to enable
  the PyTorch backend. The library libtorch.so must be built as
  described above and must be on your library path or you must add the
  path to TRITON_EXTRA_LIB_PATHS.

* **TRITON_ENABLE_CAFFE2**: Use -DTRITON_ENABLE_CAFFE2=ON to enable the
  Caffe2 backend. The library libcaffe2.so and all the other required
  libraries must be built as described above and must be on your
  library path or you must add the path to TRITON_EXTRA_LIB_PATHS.

* **TRITON_ENABLE_CUSTOM**: Use -DTRITON_ENABLE_CUSTOM=ON to enable
  support for custom backends. See
  :ref:`section-building-a-custom-backend` for information on how to
  build a custom backend.

* **TRITON_ENABLE_ENSEMBLE**: Use -DTRITON_ENABLE_ENSEMBLE=ON to enable
  support for ensembles.

These additional options may be specified:

* **TRITON_ENABLE_GRPC**: By default Triton accepts inference, status,
  health and other requests via the GRPC protocol. Use
  -DTRITON_ENABLE_GRPC=OFF to disable.

* **TRITON_ENABLE_HTTP**: By default Triton accepts inference, status,
  health and other requests via the HTTP protocol. Use
  -DTRITON_ENABLE_HTTP=OFF to disable.

* **TRITON_ENABLE_STATS**: By default Triton collects statistics for
each model that can be queried using the statistics endpoint. Use
-DTRITON_ENABLE_STATS=OFF to disable statistics.

* **TRITON_ENABLE_METRICS**: By default Triton reports
:ref:`Prometheus metrics<section-metrics>` on an HTTP endpoint. Use
-DTRITON_ENABLE_METRICS=OFF to disable both CPU and GPU metrics.  When
disabling metrics must use -DTRITON_ENABLE_METRICS_GPU=OFF to disable
GPU metrics.

* **TRITON_ENABLE_METRICS_GPU**: By default Triton reports
:ref:`Prometheus GPU metrics<section-metrics>` on an HTTP
endpoint. Use -DTRITON_ENABLE_METRICS_GPU=OFF to disable GPU metrics.

* **TRITON_ENABLE_TRACING**: By default Triton does not enable
  detailed :ref:`tracing of individual inference requests
  <section-trace>`. Use -DTRITON_ENABLE_TRACING=ON to enable.

* **TRITON_ENABLE_GCS**: Use -DTRITON_ENABLE_GCS=ON to enable the
  inference server to read model repositories from Google Cloud
  Storage.

* **TRITON_ENABLE_S3**: Use -DTRITON_ENABLE_S3=ON to enable the
  inference server to read model repositories from Amazon S3.

* **TRITON_ENABLE_GPU**: By default Triton supports NVIDIA GPUs. Use
  -DTRITON_ENABLE_GPU=OFF to disable GPU support. When GPUs are
  disable Triton will :ref:`run models on CPU when possible
  <section-running-the-inference-server-without-gpu>`.  When disabling
  GPU support must use -DTRITON_ENABLE_METRICS_GPU=OFF to disable GPU
  metrics.

* **TRITON_MIN_COMPUTE_CAPABILITY**: By default, Triton supports
  NVIDIA GPUs with CUDA compute capability 6.0 or higher. If all
  framework backends included in Triton are built to support a lower
  compute capability, then Triton Inference Server can be built to
  support that lower compute capability by setting
  -DTRITON_MIN_COMPUTE_CAPABILITY appropriately. The setting is
  ignored if -DTRITON_ENABLE_GPU=OFF.

* **TRITON_EXTRA_LIB_PATHS**: Extra paths that are searched for
  framework libraries as described above. Multiple paths can be
  specified by separating them with a semicolon, for example,
  -DTRITON_EXTRA_LIB_PATHS="/path/a;/path/b".

Build Triton
............

After configuring, build Triton with make::

  $ cd builddir
  $ make -j16 server

When the build completes the binary, libraries and headers can be
found in server/install.

.. _section-building-a-custom-backend:

Building A Custom Backend
-------------------------

The source repository contains several example custom backends in the
`src/backends/backend/examples directory
<https://github.com/NVIDIA/triton-inference-server/tree/master/src/backends/backend/examples>`_.
These example backends are built as part of the Triton server build described above.

When the build completes the example backends can be found in
server/install/backends.

See :ref:`section-backends` or more information on custom backends.

Building the Client Libraries and Examples
------------------------------------------

The provided Dockerfile.client and CMake support can be used to build
the client libraries and examples.

.. include:: client.rst
   :start-after: build-client-begin-marker-do-not-remove
   :end-before: build-client-end-marker-do-not-remove

Building the Documentation
--------------------------

The Triton documentation is found in the docs/ directory and is based
on `Sphinx <http://www.sphinx-doc.org>`_. `Doxygen
<http://www.doxygen.org/>`_ integrated with `Exhale
<https://github.com/svenevs/exhale>`_ is used for C++ API
docuementation.

To build the docs install the required dependencies::

  $ apt-get update
  $ apt-get install -y --no-install-recommends python3-pip doxygen
  $ pip3 install --upgrade setuptools
  $ pip3 install --upgrade sphinx sphinx-rtd-theme nbsphinx exhale

To get the Python client library API docs the client library models
must be installed::

  $ pip3 install --upgrade triton*.whl

Then use Sphinx to build the documentation into the build/html
directory::

  $ cd docs
  $ make clean html
