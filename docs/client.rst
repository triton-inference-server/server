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

.. _section-client-libraries:

Client Libraries
================

The inference server *client libraries* make it easy to communicate
with the TensorRT Inference Server from your C++ or Python
application. Using these libraries you can send either HTTP or GRPC
requests to the server to check status or health and to make inference
requests. These libraries also support using shared memory for passing
inputs to and receiving outputs from the inference server.
:ref:`section-client-examples` describes examples that show the use of
both the C++ and Python libraries.

You can also communicate with the inference server by using the
`protoc compiler to generate the GRPC client stub
<https://grpc.io/docs/guides/>`_ in a large number of programming
languages. The *grpc\_image\_client* example in
:ref:`section-client-examples` illustrates how to use the GRPC client
stub.

This section shows how to get the client libraries by either building
or downloading, and also describes how to :ref:`build your own client
<section-building-your-own-client>` using these libraries.

.. _section-getting-the-client-libraries:

Getting the Client Libraries
----------------------------

The provided Dockerfile.client and CMake support can be used to build
the client libraries. As an alternative to building, it is also
possible to download the pre-build client libraries from GitHub.

.. build-client-begin-marker-do-not-remove

.. _section-client-libaries-build-using-dockerfile:

Build Using Dockerfile
^^^^^^^^^^^^^^^^^^^^^^

To build the libaries using Docker, first change directory to the root
of the repo and checkout the release version of the branch that you
want to build (or the master branch if you want to build the
under-development version). The branch you use for the client build
should match the version of the inference server you are using::

  $ git checkout r19.08

Then, issue the following command to build the C++ client library and
a Python wheel file for the Python client library::

  $ docker build -t tensorrtserver_client -f Dockerfile.client .

You can optionally add *-\\-build-arg "UBUNTU_VERSION=<ver>"* to set
the Ubuntu version that you want the client library built
for. Supported values for *<ver>* are 16.04 and 18.04, with 16.04
being the default.

The generated Python wheel file works with both Python2 and Python3,
but you can control which version of Python (and pip) are used to
generate the wheel file by editing PYVER in Dockerfile.client. The
default is Python3 and pip3.

After the build completes the tensorrtserver_client docker image will
contain the built client libraries in /workspace/install/lib, the
corresponding headers in /workspace/install/include, and the Python
wheel file in /workspace/install/python. The image will also contain
the built client examples that you can learn more about in
:ref:`section-client-examples`.

.. _section-client-libaries-build-using-cmake:

Build Using CMake
^^^^^^^^^^^^^^^^^

The client library build is performed using CMake. The build
dependencies and requirements are shown in Dockerfile.client. To build
without Docker you must first install those dependencies. This section
describes the client build for Ubuntu 16.04, Ubuntu 18.04, and Windows
10 systems. The CMake build can also be targeted for other OSes and
platforms. We welcome any updates that expand the build functionality
and allow the clients to be built on additional platforms.

To build the libaries using CMake, first change directory to the root
of the repo and checkout the release version of the branch that you
want to build (or the master branch if you want to build the
under-development version)::

  $ git checkout r19.08

Ubuntu 16.04 / Ubuntu 18.04
...........................

For Ubuntu, the dependencies and how to install them can be found in
Dockerfile.client. Also note that the dependency name may be different
depending on the version of the system.

To build on Ubuntu, change to the build/ directory and run the
following to configure and build::

  $ cd build
  $ cmake -DCMAKE_BUILD_TYPE=Release
  $ make -j8 trtis-clients

When the build completes the libraries can be found in
trtis-clients/install/lib, the corresponding headers in
trtis-clients/install/include, and the Python wheel file in
trtis-clients/install/python. The trtis-clients/install directory will
also contain the built client examples that you can learn more about
in :ref:`section-client-examples`.

Windows 10
..........

For Windows, the dependencies can be installed using pip
and `vcpkg <https://github.com/Microsoft/vcpkg>`_ which is a C++ library
management tool on Windows. The following shows how to install the dependencies
using them, and you can also install the dependencies in other ways that you
prefer::

  > .\vcpkg.exe install curl[openssl]:x64-windows
  > pip install grpcio-tools wheel

The vcpkg step above installs curl and openssl, ":x64-windows" specifies the
target and it is optional. The path to the libraries should be added to
environment variable "PATH", by default it is
\\path\\to\\vcpkg\\installed\\<target>\\bin.

To build the client for Windows, as there is no default
build system available, you will need to specify the generator for
CMake to match the build system you are using. For instance, if you
are using Microsoft Visual Studio, you should do the following::

  > cd build
  > cmake -G"Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Release
  > MSBuild.exe trtis-clients.vcxproj -p:Configuration=Release

When the build completes the libraries can be found in
trtis-clients\\install\\lib, the corresponding headers in
trtis-clients\\install\\include, and the Python wheel file in
trtis-clients\\install\\python. The trtis-clients\\install directory will
also contain the built client Python examples that you can learn more
about in :ref:`section-client-examples`. At this time the Windows
build does not include the C++ examples.

.. build-client-end-marker-do-not-remove

.. _section-client-libaries-download-from-github:

Download From GitHub
^^^^^^^^^^^^^^^^^^^^

An alternative to building the client library is to download the
pre-built client libraries from the `GitHub release page
<https://github.com/NVIDIA/tensorrt-inference-server/releases>`_
corresponding to the release you are interested in. The client
libraries are found in the "Assets" section of the release page in a
tar file named after the version of the release and the OS, for
example, v1.2.0_ubuntu1604.clients.tar.gz.

The pre-built libraries can be used on the corresponding host system
(for example Ubuntu-16.04 or Ubuntu-18.04) or you can install them
into the TensorRT Inference Server container to have both the clients
and server in the same container::

  $ mkdir clients
  $ cd clients
  $ wget https://github.com/NVIDIA/tensorrt-inference-server/releases/download/<tarfile_path>
  $ tar xzf <tarfile_name>

After installing the libraries can be found in lib/, the corresponding
headers in include/, and the Python wheel file in python/. The bin/
and python/ directories contain the built examples that you can learn
more about in :ref:`section-client-examples`.

.. _section-building-your-own-client:

Building Your Own Client
------------------------

No matter how you get the client libraries (Dockerfile, CMake or
download), using them to build your own client application is the
same. The *install* directory contains all the libraries and includes
needed for your client.

For Python you just need to install the wheel from from the python/
directory. The wheel contains everything you need to communicate with
the inference server from you Python application, as shown in
:ref:`section-client-examples`.

For C++ the lib/ directory contains both shared and static libraries
and then include/ directory contains the corresponding headers. The
src/ directory contains an example application and CMake file to show
how you can build your C++ application to use the libaries and
includes. To build the example you must first install dependencies
appropriate for your platform. For example, for Ubuntu 18.04::

  $ apt-get update
  $ apt-get install software-properties-common build-essential curl git zlib1g zlib1g-dev libssl-dev libcurl4-openssl-dev

Then you can build the example application::

  $ cd src/cmake
  $ cmake .
  $ make -j8 trtis-clients

The example CMake file that illustrates how to build is in
src/cmake/trtis-clients/CMakeLists.txt. The build produces both a
statically and dynamically linked version of the example application
into src/cmake/trtis-clients/install/bin.

.. _section-client-api:

Client API
----------

The C++ client API exposes a class-based interface for querying server
and model status and for performing inference. The commented interface
is available at `src/core/request.h
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/core/request.h>`_
and in the API Reference.

The Python client API provides similar capabilities as the C++
API. The commented interface is available at
`src/clients/python/\_\_init\_\_.py
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/python/__init__.py>`_
and in the API Reference.

A simple C++ example application at `src/clients/c++/simple\_client.cc
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/c%2B%2B/simple_client.cc>`_
and a Python version at `src/clients/python/simple\_client.py
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/python/simple_client.py>`_
demonstrate basic client API usage.

To run the the C++ version of the simple example, first build or
download it as described in :ref:`section-getting-the-client-examples`
and then::

  $ simple_client
  0 + 1 = 1
  0 - 1 = -1
  1 + 1 = 2
  1 - 1 = 0
  2 + 1 = 3
  2 - 1 = 1
  ...
  14 - 1 = 13
  15 + 1 = 16
  15 - 1 = 14

To run the the Python version of the simple example, first build or
download it as described in :ref:`section-getting-the-client-examples`
and install the tensorrtserver whl, then::

  $ python simple_client.py

Shared Memory
^^^^^^^^^^^^^

A simple C++ example application using shared memory at
`src/clients/c++/simple\_shm\_client.cc
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/c%2B%2B/simple_shm_client.cc>`_
and a Python version at `src/clients/python/simple\_shm\_client.py
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/python/simple_shm_client.py>`_
demonstrate the usage of shared memory with the client API.

To run the the C++ version of the simple shared memory example, first
build or download it as described in
:ref:`section-getting-the-client-examples` and then::

  $ simple_shm_client
  0 + 1 = 1
  0 - 1 = -1
  1 + 1 = 2
  1 - 1 = 0
  2 + 1 = 3
  2 - 1 = 1
  ...
  14 - 1 = 13
  15 + 1 = 16
  15 - 1 = 14

To run the the Python version of the simple shared memory example,
first build or download it as described in
:ref:`section-getting-the-client-examples` and install the
tensorrtserver whl, then::

  $ python simple_shm_client.py

String Datatype
^^^^^^^^^^^^^^^

Some frameworks support tensors where each element in the tensor is a
string (see :ref:`section-datatypes` for information on supported
datatypes). For the most part, the Client API is identical for string
and non-string tensors. One exception is that in the C++ API a string
input tensor must be initialized with SetFromString() instead of
SetRaw().

String tensors are demonstrated in the C++ example application at
`src/clients/c++/simple\_string\_client.cc
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/c%2B%2B/simple_string_client.cc>`_
and a Python version at `src/clients/python/simple\_string\_client.py
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/python/simple_string_client.py>`_.

.. _section-client-api-stateful-models:

Client API for Stateful Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When performing inference using a :ref:`stateful model
<section-stateful-models>`, a client must identify which inference
requests belong to the same sequence and also when a sequence starts
and ends.

Each sequence is identified with a correlation ID that is provided
when the inference context is created (in either the Python of C++
APIs). It is up to the clients to create a unique correlation ID. For
each sequence the first inference request should be marked as the
start of the sequence and the last inference requests should be marked
as the end of the sequence. Start and end are marked using the flags
provided with the RunOptions in the C++ API and the run() and
async_run() methods in the Python API.

The use of correlation ID and start and end flags are demonstrated in
the C++ example application at
`src/clients/c++/simple\_sequence\_client.cc
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/c%2B%2B/simple_sequence_client.cc>`_
and a Python version at
`src/clients/python/simple\_sequence\_client.py
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/clients/python/simple_sequence_client.py>`_.
