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

# Building Triton

This section gives an overview of how to build the Triton server. For
information on building the Triton client libraries and examples see
[Client Libraries](client_libraries.md).

Triton server is built using the [build.py](../build.py) script. The
build.py script currently supports building for the following
platforms. See [Building on Unsupported
Platforms](#building-on-unsupported-platforms) if you are attempting
to build Triton on a platform that is not listed here.

* [Ubuntu 20.04, x86-64](#ubuntu)

* [Jetpack 4.x, NVIDIA Jetson (Xavier, Nano, TX2)](#jetpack)

* [Windows 10, x86-64](#windows)

## <a name="ubuntu"></a>Building for Ubuntu 20.04

For Ubuntu-20.04 build.py supports both a Docker build and a
non-Docker build.

* [Build using Docker](#ubuntu-docker) and the TensorFlow and PyTorch
  Docker images from [NVIDIA GPU Cloud
  (NGC)](https://ngc.nvidia.com>).

* [Build without Docker](#ubuntu-without-docker).

### <a name="ubuntu-docker"></a>Building with Docker

The easiest way to build Triton is to use Docker. The result of the
build will be a Docker image called *tritonserver* that will contain
the tritonserver executable in /opt/tritonserver/bin and the required
shared libraries in /opt/tritonserver/lib. The backends built for
Triton will be in /opt/tritonserver/backends (note that as of the
21.02 release the TensorRT backend is still included in the core of
Triton and so does not appear in /opt/tritonserver/backends).

The first step for any build is to checkout the
[triton-inference-server/server](https://github.com/triton-inference-server/server)
repo branch for the release you are interested in building (or the
master branch to build from the development branch). Then run build.py
as described below. The build.py script performs these steps when
building with Docker.

* Fetch the appropriate \<xx.yy\>-py3-min image from NGC. The *min*
  image is a minimal/base image that contains the CUDA, cuDNN,
  TensorRT and other dependencies that are required to build Triton.

* Create a *tritonserver_buildbase* Docker image that adds additional
  build dependencies to the *min* image.

* Run build.py within the *tritonserver_buildbase* image to actually
  build Triton. See [Build without Docker](#ubuntu-without-docker) for
  more details on that part of the build process. The result of this
  step is a *tritonserver_build* image that contains the built Triton
  artifacts.

* Create the final *tritonserver* Docker image by extracting the
  appropriate libraries, executables and other artifacts from
  *tritonserver_build*.

By default, build.py does not enable any of Triton's optional features
and so you must enable them explicitly. The following build.py
invocation builds all features, backends, and repository agents.

```
$ ./build.py --cmake-dir=<path/to/repo>/build --build-dir=/tmp/citritonbuild --enable-logging --enable-stats --enable-tracing --enable-metrics --enable-gpu-metrics --enable-gpu --filesystem=gcs --filesystem=azure_storage --filesystem=s3 --endpoint=http --endpoint=grpc --repo-tag=common:<container tag> --repo-tag=core:<container tag> --repo-tag=backend:<container tag> --repo-tag=thirdparty:<container tag> --backend=ensemble --backend=tensorrt --backend=identity:<container tag> --backend=repeat:<container tag> --backend=square:<container tag> --backend=onnxruntime:<container tag> --backend=pytorch:<container tag> --backend=tensorflow1:<container tag> --backend=tensorflow2:<container tag> --backend=openvino:<container tag> --backend=python:<container tag> --backend=dali:<container tag> --repoagent=checksum:<container tag>
```

If you are building on master/main branch then \<container tag\>
should be set to "main". If you are building on a release branch you
should set \<container tag\> to match the branch name. For example, if
you are building on the r21.03 branch you should set \<container tag\>
to be "r21.03". You can use a different \<container tag\> for a
component to instead use the corresponding branch/tag in the
build. For example, if you have a branch called "mybranch" in the
[identity_backend](https://github.com/triton-inference-server/identity_backend)
repo that you want to use in the build, you would specify
--backend=identity:mybranch.

### <a name="ubuntu-without-docker"></a>Building without Docker

To build Triton without using Docker you must install the build
dependencies that are handled automatically when building with Docker.
These dependencies include [CUDA and cuDNN](#cuda-cublas-cudnn),
[TensorRT](#tensorrt), and the dependencies listed in the
create_dockerfile_buildbase() function of [build.py](../build.py).

Once you have installed these dependencies on your build system you
can then use build.py with the --no-container-build flag to build
Triton. See the build.py invocation in [Build using
Docker](#ubuntu-docker) for an example of how to run build.py. You can
use that same invocation with the --no-container-build flag to build
without Docker.

The first step for any build is to checkout the
[triton-inference-server/server](https://github.com/triton-inference-server/server)
repo branch for the release you are interested in building (or the
master branch to build from the development branch). Then run build.py
as described below. The build.py script will perform the following
steps (note that if you are building with Docker that these same steps
will be performed during the Docker build within the
*tritonserver_build* container).

* Use the CMake files in [build](../build) to build Triton's core
  shared library and *tritonserver* executable.

* Fetch each requested backend and build it using the CMake file from
  the corresponding backend repo. For example, the ONNX Runtime
  backend is built using
  [triton-inference-server/onnxruntime_backend/CMakeLists.txt](https://github.com/triton-inference-server/onnxruntime_backend/blob/main/CMakeLists.txt). Some
  of the backends may use Docker as part of their build (for example
  [ONNX
  Runtime](https://github.com/triton-inference-server/onnxruntime_backend)
  and
  [OpenVINO](https://github.com/triton-inference-server/openvino_backend)). If
  you don't want to use Docker in those cases you must consult the
  build process for those backends.

* Fetch each repository agent and build it using the CMake file from
  the corresponding repo. For example, the
  [Checksum](https://github.com/triton-inference-server/checksum_repository_agent)
  repository agent is built using
  [triton-inference-server/checksum_repository_agent/CMakeLists.txt](https://github.com/triton-inference-server/checksum_repository_agent/blob/main/CMakeLists.txt).

By default build.py clones Triton repos from
https://github.com/triton-inference-server. Use the
--github-organization options to select a different URL.

The backends can also be built independently in each of the backend
repositories. See the [backend
repo](https://github.com/triton-inference-server/backend) for more
information.

#### CUDA, cuBLAS, cuDNN

For Triton to support NVIDIA GPUs you must install CUDA, cuBLAS and
cuDNN. These libraries must be installed on system include and library
paths so that they are available for the build. The version of the
libraries used for a given release can be found in the [Framework
Containers Support
Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

For a given version of Triton you can attempt to build with
non-supported versions of the libraries but you may have build or
execution issues since non-supported versions are not tested.

#### TensorRT

The TensorRT includes and libraries must be installed on system
include and library paths so that they are available for the
build. The version of TensorRT used in a given release can be found in
the [Framework Containers Support
Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

For a given version of Triton you can attempt to build with
non-supported versions of TensorRT but you may have build or execution
issues since non-supported versions are not tested.

## <a name="jetpack"></a>Building for JetPack 4.x

*Under Construction*

## <a name="windows"></a>Building for Windows 10

*Under Construction*

## Building on Unsupported Platforms

Building for an unsupported OS and/or hardware platform is
possible. All of the build scripting and CMake files are included in
the public repos. However, due to differences in compilers, libraries,
package management, etc. you may have to make changes in the build
scripts, CMake files and the source code.

You should familiarize yourself with the build process for supported
platforms by reading the above documentation and then follow the
process for the supported platform that most closely matches the
platform you are interested in (for example, if you are trying to
build for RHEL/x86-64 then follow the [Building for Ubuntu
20.04](building-for-ubuntu-2004) process. You will likely need to make
changes in the following areas.

* The build.py script installs dependencies for the build using
  platform-specific packaging tools, for example, apt-get for
  Ubuntu. You will need to change build.py to use the packaging tool
  appropriate for your platform.

* The package and libraries names for your platform may differ from
  those used by build.py when installing dependencies. You will need
  to find the corresponding packages on libraries on your platform.

* Your platform may use a different compiler or compiler version than
  the support platforms. As a result you may encounter build errors
  that need to be fixed by editing the source code or changing the
  compilation flags.

* Triton depends on a large number of open-source packages that it
  builds from source. If one of these packages does not support your
  platform them you may need to disable the Triton feature that
  depends on that package. For example, Triton supports the S3
  filesystem by building the aws-sdk-cpp package. If aws-sdk-cpp
  doesn't build for your platform then you can remove the need for
  that package by not specifying --filesystem=s3 when you run
  build.py. In general, you should start by running build.py with the
  minimal required feature set.

* The
  [TensorFlow](https://github.com/triton-inference-server/tensorflow_backend)
  backend extracts pre-built shared libraries from the TensorFlow NGC
  container as part of the build. This container is only available for
  Ubuntu-20.04 / x86-64 and so if you require the TensorFlow backend
  for your platform you will need download the TensorFlow container
  and modify its build to produce shared libraries for your
  platform. You must use the TensorFlow source and build scripts from
  within the NGC container because they contain Triton-specific
  patches that are required for the Triton TensorFlow backend.

* By default, the
  [PyTorch](https://github.com/triton-inference-server/pytorch_backend)
  backend build extracts pre-built shared libraries from The PyTorch
  NGC container. But the build can also use PyTorch shared libraries
  that you build separately for your platform. See the pytorch_backend
  build process for details.
