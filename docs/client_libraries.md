<!--
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
-->

# Client Libraries

The *client libraries* provide [APIs](#client-library-apis) that make
it easy to communicate with Triton from your C++ or Python
application. Using these libraries you can send either HTTP/REST or
GRPC requests to Triton to access all its capabilities: inferencing,
status and health, statistics and metrics, model repository
management, etc. These libraries also support using system and CUDA
shared memory for passing inputs to and receiving outputs from
Triton. [Examples](client_examples.md) show the use of both the C++ and Python
libraries.

## Getting the Client Libraries

The easiest way to get the Python client library is to [use pip to
install the tritonclient
module](#download-using-python-package-installer-pip). You can also
download both C++ and Python client libraries from [Triton GitHub
release](#download-from-github), or [download a pre-built Docker image
containing the client libraries](#download-docker-image-from-ngc) from
[NVIDIA GPU Cloud (NGC)](https://ngc.nvidia.com).

It is also possible to build build the client libraries with
[Docker](#build-using-docker) or with [cmake](#build-using-docker).

### Download Using Python Package Installer (pip)

The GRPC and HTTP client libraries are available as a Python package
that can be installed using a recent version of pip. **Currently pip
install is only available on Linux**.

```
$ pip install nvidia-pyindex
$ pip install tritonclient[all]
```

Using *all* installs both the HTTP/REST and GRPC client
libraries. There are two optional packages available, *grpc* and
*http* that can be used to install support specifically for the
protocol. For example, to install only the HTTP/REST client library
use,

```
$ pip install nvidia-pyindex
$ pip install tritonclient[http]
```

The components of the install packages are:

* http
* grpc [ `service_pb2`, `service_pb2_grpc`, `model_config_pb2` ]
* utils [ linux distribution will include `shared_memory` and `cuda_shared_memory`]

The Linux version of the package also includes the
[perf_analyzer](perf_analyzer.md) binary. The perf_analyzer binary is
built on Ubuntu 20.04 and may not run on other Linux distributions. To
run the perf_analyzer the following dependency must be installed:

```bash
sudo apt update
sudo apt install libb64-dev
```

### Download From GitHub

The client libraries and the perf_analyzer executable can be
downloaded from the [Triton GitHub release
page](https://github.com/triton-inference-server/server/releases)
corresponding to the release you are interested in. The client
libraries are found in the "Assets" section of the release page in a
tar file named after the version of the release and the OS, for
example, v2.3.0_ubuntu1804.clients.tar.gz.

The pre-built libraries can be used on the corresponding host system
or you can install them into the Triton container to have both the
clients and server in the same container.

```bash
$ mkdir clients
$ cd clients
$ wget https://github.com/triton-inference-server/server/releases/download/<tarfile_path>
$ tar xzf <tarfile_name>
```

After installing, the libraries can be found in lib/, the headers in
include/, and the Python wheel files in python/. The bin/ and python/
directories contain the built examples that you can learn more about
in [Examples](client_examples.md).

The perf_analyzer binary is built on Ubuntu 20.04 and may not run on
other Linux distributions. To use the C++ libraries or perf_analyzer
executable you must install some dependencies.

```bash
$ apt-get update
$ apt-get install curl libcurl4-openssl-dev libb64-dev
```

### Download Docker Image From NGC

A Docker image containing the client libraries and examples is
available from [NVIDIA GPU Cloud
(NGC)](https://ngc.nvidia.com). Before attempting to pull the
container ensure you have access to NGC.  For step-by-step
instructions, see the [NGC Getting Started
Guide](http://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html).

Use docker pull to get the client libraries and examples container
from NGC.

```bash
$ docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk
```

Where \<xx.yy\> is the version that you want to pull. Within the
container the client libraries are in /workspace/install/lib, the
corresponding headers in /workspace/install/include, and the Python
wheel files in /workspace/install/python. The image will also contain
the built client examples that you can learn more about in
[Examples](client_examples.md).

### Build Using Docker

To build the client libraries using Docker, first change directory to
the root of the repo and checkout the release version of the branch
that you want to build (or the master branch if you want to build the
under-development version). The branch you use for the client build
should match the version of Triton you are using.

```bash
$ git checkout r21.03
```

Then, issue the following command to build the C++ client library and
the Python wheel files for the Python client library.

```bash
$ docker build -t tritonserver_sdk -f Dockerfile.sdk .
```

You can optionally add *--build-arg "BASE_IMAGE=\<base_image\>"* to set
the base image that you want the client library built against. This
base image must be an Ubuntu CUDA image to be able to build CUDA
shared memory support. If CUDA shared memory support is not required,
you can use Ubuntu 20.04 as the base image.

After the build completes the tritonserver_sdk docker image will
contain the built client libraries in /workspace/install/lib, the
corresponding headers in /workspace/install/include, and the Python
wheel files in /workspace/install/python. The image will also contain
the built client examples that you can learn more about in
[Examples](client_examples.md).

### Build Using CMake

The client library build is performed using CMake. *IMPORTANT*
Note that version 3.18.4 of cmake is needed to compile the
client. The build dependencies and requirements are shown in
`Dockerfile.sdk`. To build without Docker you must first
install those dependencies along with required cmake version.
This section describes the client build for Ubuntu 20.04 and
Windows 10 systems.

To build the libraries using CMake, first change directory to the root
of the repo and checkout the release version of the branch that you
want to build (or the master branch if you want to build the
under-development version).

```bash
$ git checkout r21.03
```

#### Ubuntu 20.04

For Ubuntu, the dependencies and how to install them can be found in
`Dockerfile.sdk`. The appropriate CUDA library must be installed
if TRITON_ENABLE_GPU=OFF is not specified in the cmake. Follow the
dockerfile closely till the cmake invocation. Also note that
the dependency name may be different depending on the version of the
system.

To build on Ubuntu, run the following to configure and build:

```bash
$ mkdir builddir && cd builddir
$ cmake -DCMAKE_BUILD_TYPE=Release ../build
$ make -j8 client
```

If you want to build a version of the client libraries and examples
that does not include the CUDA shared memory support, use the
following cmake configuration.

```bash
$ cmake -DTRITON_ENABLE_GPU=OFF -DTRITON_ENABLE_METRICS_GPU=OFF -DCMAKE_BUILD_TYPE=Release ../build
```

When the build completes the libraries can be found in
client/install/lib, the corresponding headers in
client/install/include, and the Python wheel files in
client/install/python. The client/install directory will also contain
the built client examples that you can learn more about in
[Examples](client_examples.md).

#### Windows 10

For Windows, the dependencies can be installed using pip and
[vcpkg](https://github.com/Microsoft/vcpkg) which is a C++ library
management tool on Windows. The following shows how to install the
dependencies using them, and you can also install the dependencies in
other ways that you prefer.

```
> .\vcpkg.exe install openssl:x64-windows zlib:x64-windows rapidjson:x64-windows
> .\pip.exe install --upgrade setuptools grpcio-tools wheel
```

The vcpkg step above installs openssl, zlib and rapidjson,
":x64-windows" specifies the target and it is optional. The path to
the libraries should be added to environment variable "PATH", by
default it is \path\to\vcpkg\installed\\\<target>\bin. Update the
pip to get the proper wheel from PyPi. Users may need to invoke
pip.exe from a command line ran as an administrator.

To build the client for Windows, as there is no default
build system available, you will need to specify the generator for
CMake to match the build system you are using. For instance, if you
are using Microsoft Visual Studio, you should do the following.

```
> cd build
> cmake -G"Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Release
> MSBuild.exe client.vcxproj -p:Configuration=Release
```

If you want to build a version of the client libraries and examples
that does not include the CUDA shared memory support, use the
following cmake configuration.

```
> cmake -G"Visual Studio 16 2019" -DTRITON_ENABLE_GPU=OFF -DTRITON_ENABLE_METRICS_GPU=OFF -DCMAKE_BUILD_TYPE=Release -DTRITON_COMMON_REPO_TAG:STRING=<tag> -DTRITON_CORE_REPO_TAG:STRING=<tag>
```

Where \<tag\> is "main" if you are building the clients from the master
branch, or \<tag\> is "r\<x\>.\<y\>" if you are building on a release
branch.

When the build completes the libraries can be found in
client\install\lib, the corresponding headers in
client\install\include, and the Python wheel files in
client\install\python. The client\install directory will also contain
the built client Python examples that you can learn more about in
[Examples](client_examples.md). At this time the Windows build does
not include the C++ examples.

The MSBuild.exe may need to be invoked twice for a successful build.

## Client Library APIs

The C++ client API exposes a class-based interface. The commented
interface is available in
[grpc_client.h](../src/clients/c%2B%2B/library/grpc_client.h.in),
[http_client.h](../src/clients/c%2B%2B/library/http_client.h.in),
[common.h](../src/clients/c%2B%2B/library/common.h).

The Python client API provides similar capabilities as the C++
API. The commented interface is available in
[grpc](../src/clients/python/library/tritonclient/grpc/__init__.py)
and
[http](../src/clients/python/library/tritonclient/http/__init__.py).

[Examples](client_examples.md) describes the example applications that
demonstrate different parts of the client library APIs.
