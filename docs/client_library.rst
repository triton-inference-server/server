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

.. _section-client-libraries:

Client Libraries
================

The *client libraries* make it easy to communicate with Triton from
your C++ or Python application. Using these libraries you can send
either HTTP/REST or GRPC requests to Triton to check status or health
and to make inference requests. These libraries also support using
system and CUDA shared memory for passing inputs to and receiving
outputs from Triton.  :ref:`section-client-examples` describes
examples that show the use of both the C++ and Python libraries.

This section shows how to get the client libraries by either building
or downloading, and also describes how to :ref:`build your own client
<section-building-your-own-client>` using these libraries.

.. _section-getting-the-client-libraries:

Getting the Client Libraries
----------------------------

The provided Dockerfile.client and CMake support can be used to build
the client libraries. As an alternative to building, it is also
possible to download the pre-build client libraries from GitHub or a
pre-built Docker image containing the client libraries from `NVIDIA
GPU Cloud (NGC) <https://ngc.nvidia.com>`_.

.. build-client-begin-marker-do-not-remove

.. _section-client-libraries-build-using-dockerfile:

Build Using Dockerfile
^^^^^^^^^^^^^^^^^^^^^^

To build the libraries using Docker, first change directory to the
root of the repo and checkout the release version of the branch that
you want to build (or the master branch if you want to build the
under-development version). The branch you use for the client build
should match the version of Triton you are using::

  $ git checkout r20.08

Then, issue the following command to build the C++ client library and
the Python wheel files for the Python client library::

  $ docker build -t tritonserver_client -f Dockerfile.client .

You can optionally add *-\\-build-arg "BASE_IMAGE=<base_image>"* to
set the base image that you want the client library built
against. This base image must be a Ubuntu CUDA image to be able to
build CUDA shared memory support. If CUDA shared memory support is not
required, you can use Ubuntu 18.04 as the base image.

After the build completes the tritonserver_client docker image will
contain the built client libraries in /workspace/install/lib, the
corresponding headers in /workspace/install/include, and the Python
wheel files in /workspace/install/python. The image will also contain
the built client examples that you can learn more about in
:ref:`section-client-examples`.

.. _section-client-libraries-build-using-cmake:

Build Using CMake
^^^^^^^^^^^^^^^^^

The client library build is performed using CMake. The build
dependencies and requirements are shown in Dockerfile.client. To build
without Docker you must first install those dependencies. This section
describes the client build for Ubuntu 18.04 and Windows 10
systems. The CMake build can also be targeted for other OSes and
platforms. We welcome any updates that expand the build functionality
and allow the clients to be built on additional platforms.

To build the libraries using CMake, first change directory to the root
of the repo and checkout the release version of the branch that you
want to build (or the master branch if you want to build the
under-development version)::

  $ git checkout r20.08

Ubuntu 18.04
............

For Ubuntu, the dependencies and how to install them can be found in
Dockerfile.client. Also note that the dependency name may be different
depending on the version of the system.

To build on Ubuntu, run the following to configure and build::

  $ mkdir builddir && cd builddir
  $ cmake -DCMAKE_BUILD_TYPE=Release ../build
  $ make -j8 client

If you want to build a version of the client libraries and examples
that does not include the CUDA shared memory support, use the
following cmake configuration::

  $ cmake -DTRITON_ENABLE_GPU=OFF -DTRITON_ENABLE_METRICS_GPU=OFF -DCMAKE_BUILD_TYPE=Release ../build

When the build completes the libraries can be found in
client/install/lib, the corresponding headers in
client/install/include, and the Python wheel files in
client/install/python. The client/install directory will also contain
the built client examples that you can learn more about in
:ref:`section-client-examples`.

Windows 10
..........

For Windows, the dependencies can be installed using pip
and `vcpkg <https://github.com/Microsoft/vcpkg>`_ which is a C++ library
management tool on Windows. The following shows how to install the dependencies
using them, and you can also install the dependencies in other ways that you
prefer::

  > .\vcpkg.exe install openssl:x64-windows zlib:x64-windows rapidjson:x64-windows
  > .\pip.exe install --upgrade setuptools grpcio-tools wheel

The vcpkg step above installs openssl, zlib and rapidjson,
":x64-windows" specifies the target and it is optional. The path to
the libraries should be added to environment variable "PATH", by
default it is \\path\\to\\vcpkg\\installed\\<target>\\bin. Update the
pip to get the proper wheel from PyPi. Users may need to invoke
pip.exe from a command line ran as an administrator.

To build the client for Windows, as there is no default
build system available, you will need to specify the generator for
CMake to match the build system you are using. For instance, if you
are using Microsoft Visual Studio, you should do the following::

  > cd build
  > cmake -G"Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Release
  > MSBuild.exe client.vcxproj -p:Configuration=Release

If you want to build a version of the client libraries and examples
that does not include the CUDA shared memory support, use the
following cmake configuration::

  > cmake -G"Visual Studio 16 2019" -DTRITON_ENABLE_GPU=OFF -DTRITON_ENABLE_METRICS_GPU=OFF -DCMAKE_BUILD_TYPE=Release

When the build completes the libraries can be found in
client\\install\\lib, the corresponding headers in
client\\install\\include, and the Python wheel files in
client\\install\\python. The client\\install directory will also
contain the built client Python examples that you can learn more about
in :ref:`section-client-examples`. At this time the Windows build does
not include the C++ examples.

The MSBuild.exe may need to be invoked twice for a successfull
build.

.. build-client-end-marker-do-not-remove

.. _section-client-libraries-download-from-github:

Download From GitHub
^^^^^^^^^^^^^^^^^^^^

An alternative to building the client library is to download the
pre-built client libraries from the `GitHub release page
<https://github.com/triton-inference-server/server/releases>`_
corresponding to the release you are interested in. The client
libraries are found in the "Assets" section of the release page in a
tar file named after the version of the release and the OS, for
example, v1.2.0_ubuntu1804.clients.tar.gz.

The pre-built libraries can be used on the corresponding host system
or you can install them into the Triton container to have both the
clients and server in the same container::

  $ mkdir clients
  $ cd clients
  $ wget https://github.com/triton-inference-server/server/releases/download/<tarfile_path>
  $ tar xzf <tarfile_name>

After installing the libraries can be found in lib/, the corresponding
headers in include/, and the Python wheel files in python/. The bin/
and python/ directories contain the built examples that you can learn
more about in :ref:`section-client-examples`.

To use the C++ libraries you must install some dependencies. For
Ubuntu 18.04::

  $ apt-get update
  $ apt-get install curl libcurl4-openssl-dev libb64-dev

.. _section-client-libraries-download-from-ngc:

Download Docker Image From NGC
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Docker image containing the client libraries and examples is
available from `NVIDIA GPU Cloud (NGC)
<https://ngc.nvidia.com>`_. Before attempting to pull the container
ensure you have access to NGC.  For step-by-step instructions, see the
`NGC Getting Started Guide
<http://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html>`_.

Use docker pull to get the client libraries and examples container
from NGC::

  $ docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3-clientsdk

Where <xx.yy> is the version that you want to pull.

Within the container the client libraries are in
/workspace/install/lib, the corresponding headers in
/workspace/install/include, and the Python wheel files in
/workspace/install/python. The image will also contain the built
client examples that you can learn more about in
:ref:`section-client-examples`.

.. _section-building-your-own-client:

Building Your Own Client
------------------------

No matter how you get the client libraries (Dockerfile, CMake or
download), using them to build your own client application is the
same. The *install* directory contains all the libraries and includes
needed for your client.

For Python you just need to install the wheel files from the python/
directory. The wheels contain everything you need to communicate with
Triton from you Python application, as shown in
:ref:`section-client-examples`.

For C++ the lib/ directory contains both shared and static libraries
and the include/ directory contains the corresponding headers.

.. _section-client-api:

Client Library API
------------------

The C++ client API exposes a class-based interface for querying server
and model status and for performing inference. The commented interface
is available in the `library headers
<https://github.com/triton-inference-server/server/tree/master/src/clients/c%2B%2B/library>`_
and in the API Reference.

The Python client API provides similar capabilities as the C++
API. The commented interface is available in `grpcclient.py and
httpclient.py
<https://github.com/triton-inference-server/server/tree/master/src/clients/python/library>`_
and in the API Reference.

Section :ref:`section-simple-examples` describes the example
applications that demonstrate different parts of the client library
API.
