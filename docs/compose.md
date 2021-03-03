<!--
# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

# Customize Triton Container

Starting with the r20.10 release, two Docker images are available from
[NVIDIA GPU Cloud (NGC)](https://ngc.nvidia.com>) that make it
possible to easily construct customized versions of Triton. By
customizing Triton you can significantly reduce the size of the Triton
image by removing functionality that you don't require.

Currently the customization is limited as described below but future
releases will increase the amount of customization that is available.
It is also possible to [build Triton](build.md#building-triton)
yourself to get more exact customization.

The two Docker images used for customization are retrieved using the
following commands.

```
$ docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3-min
$ docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3
```

Where \<xx.yy\> is the version of Triton that you want to customize. The
\<xx.yy\>-py3-min image is a minimal, base image that contains the CUDA,
cuDNN, etc. dependencies that are required to run Triton. The
\<xx.yy\>-py3 image contains the complete Triton with all options and
backends.

### Minimum Triton

To create an image containing the minimal possible Triton use the
following multi-stage Dockerfile. As mentioned above the amount of
customization currently available is limited. As a result the minimum
Triton still contains both HTTP/REST and GRPC endpoints; S3, GCS and
Azure Storage filesystem support; and the TensorRT and legacy custom
backends.

```
FROM nvcr.io/nvidia/tritonserver:<xx.yy>-py3 as full
FROM nvcr.io/nvidia/tritonserver:<xx.yy>-py3-min
COPY --from=full /opt/tritonserver/bin /opt/tritonserver/bin
COPY --from=full /opt/tritonserver/lib /opt/tritonserver/lib
```

Then use Docker to create the image.

```
$ docker build -t tritonserver_min .
```

### Triton with Supported Backends

One or more of the supported
[PyTorch](https://github.com/triton-inference-server/pytorch_backend),
[TensorFlow1](https://github.com/triton-inference-server/tensorflow_backend),
[TensorFlow2](https://github.com/triton-inference-server/tensorflow_backend),
[ONNX
Runtime](https://github.com/triton-inference-server/onnxruntime_backend),
[Python](https://github.com/triton-inference-server/python_backend),
and [DALI](https://github.com/triton-inference-server/dali_backend)
backends can be added to the minimum Triton image. The backend can be
built from scratch or the appropriate backend directory can be copied
from from the full Triton image. For example, to create a Triton image
that creates a minimum Triton plus support for TensorFlow1 use the
following Dockerfile.

```
FROM nvcr.io/nvidia/tritonserver:<xx.yy>-py3 as full
FROM nvcr.io/nvidia/tritonserver:<xx.yy>-py3-min
COPY --from=full /opt/tritonserver/bin /opt/tritonserver/bin
COPY --from=full /opt/tritonserver/lib /opt/tritonserver/lib
COPY --from=full /opt/tritonserver/backends/tensorflow1 /opt/tritonserver/backends/tensorflow1
```

Depending on the backend it may also be necessary to include
additional dependencies in the image. For example, the Python backend
requires that Python3 be installed in the image.

Then use Docker to create the image.

```
$ docker build -t tritonserver_custom .
```

### Triton with Unsupported and Custom Backends

You can [create and build your own Triton
backend](https://github.com/triton-inference-server/backend).  The
result of that build should be a directory containing your backend
shared library and any additional files required by the
backend. Assuming your backend is called "mybackend" and that the
directory is "./mkbackend", the following Dockerfile will create a
Triton image that contains all the supported Triton backends plus your
custom backend.

```
FROM nvcr.io/nvidia/tritonserver:<xx.yy>-py3 as full
COPY ./mybackend /opt/tritonserver/backends/mybackend
```

You also need to install any additional dependencies required by your
backend as part of the Dockerfile. Then use Docker to create the
image.

```
$ docker build -t tritonserver_custom .
```

