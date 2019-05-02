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

Building
========

The TensorRT Inference Server is built using Docker and the TensorFlow
and PyTorch containers from `NVIDIA GPU Cloud (NGC)
<https://ngc.nvidia.com>`_. Before building you must install Docker
and nvidia-docker and login to the NGC registry by following the
instructions in :ref:`section-installing-prebuilt-containers`.

.. _section-building-the-server:

Building the Server
-------------------

To build a release version of the TensorRT Inference Server container,
change directory to the root of the repo and checkout the release
version of the branch that you want to build (or the master branch if
you want to build the under-development version)::

  $ git checkout r19.05

Then use docker to build::

  $ docker build --pull -t tensorrtserver .

Incremental Builds
^^^^^^^^^^^^^^^^^^

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

  # cd /workspace
  # bazel build -c opt src/servers/trtserver
  # cp /workspace/bazel-bin/src/servers/trtserver /opt/tensorrtserver/bin/.
  # cp /workspace/bazel-bin/src/core/libtrtserver.so /opt/tensorrtserver/lib/.

Some source changes seem to cause bazel to get confused and not
correctly rebuild all required sources. You can force bazel to rebuild
all of the inference server source without requiring a complete
rebuild of the TensorFlow and Caffe2 components by doing the following
before issuing the above build command::

  # rm -fr bazel-bin/src

.. _section-building-the-example-custom-backends:

Building the Example Custom Backends
------------------------------------

Some examples of the custom backends can be found in the `custom directory
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/custom>`_.
These backends are built as part of the *build* container, so within the
container, you can find the corresponding model files in
/opt/tensorrtserver/custom. Or you can perform an incremental build with::

  # cd /workspace
  # bazel build -c opt src/custom/...
  # cp /workspace/bazel-bin/src/custom/* /opt/tensorrtserver/custom/.

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
