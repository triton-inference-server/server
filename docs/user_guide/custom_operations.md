<!--
# Copyright 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Custom Operations

Modeling frameworks that allow custom operations are partially
supported by the Triton Inference Server. Custom operations can be
added to Triton at build time or at startup and are made available to
all loaded models.

## TensorRT

TensorRT allows a user to create [custom
layers](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#extending)
which can then be used in TensorRT models. For those models to run in
Triton the custom layers must be made available.

To make the custom layers available to Triton, the TensorRT custom
layer implementations must be compiled into one or more shared
libraries which must then be loaded into Triton using LD_PRELOAD. For
example, assuming your TensorRT custom layers are compiled into
libtrtcustom.so, starting Triton with the following command makes
those custom layers available to all TensorRT models.

```bash
$ LD_PRELOAD=libtrtcustom.so:${LD_PRELOAD} tritonserver --model-repository=/tmp/models ...
```

A limitation of this approach is that the custom layers must be
managed separately from the model repository itself. And more
seriously, if there are custom layer name conflicts across multiple
shared libraries there is currently no way to handle it.

When building the custom layer shared library it is important to use
the same version of TensorRT as is being used in Triton. You can find
the TensorRT version in the [Triton Release
Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html). A
simple way to ensure you are using the correct version of TensorRT is
to use the [NGC TensorRT
container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)
corresponding to the Triton container. For example, if you are using
the 23.12 version of Triton, use the 23.12 version of the TensorRT
container.

## TensorFlow

TensorFlow allows users to [add custom
operations](https://www.tensorflow.org/guide/create_op) which can then
be used in TensorFlow models. You can load custom TensorFlow operations
into Triton in two ways:
* At model load time, by listing them in the model configuration.
* At server launch time, by using LD_PRELOAD.

To register your custom operations library via the the model configuration,
you can include it as an additional field. See the below configuration as an example.

```bash
$ model_operations { op_library_filename: "path/to/libtfcustom.so" }
```

Note that even though the models are loaded at runtime, multiple models can use the custom
operators. There is currently no way to deallocate the custom operators, so they will stay
available until Triton is shut down.

You can also register your custom operations library via LD_PRELOAD. For example,
assuming your TensorFlow custom operations are compiled into libtfcustom.so,
starting Triton with the following command makes those operations
available to all TensorFlow models.

```bash
$ LD_PRELOAD=libtfcustom.so:${LD_PRELOAD} tritonserver --model-repository=/tmp/models ...
```

With this approach, all TensorFlow custom operations depend on a TensorFlow shared
library that must be available to the custom shared library when it is
loading. In practice, this means that you must make sure that
/opt/tritonserver/backends/tensorflow1 or
/opt/tritonserver/backends/tensorflow2 is on the library path before
issuing the above command. There are several ways to control the
library path and a common one is to use the LD_LIBRARY_PATH. You can
set LD_LIBRARY_PATH in the "docker run" command or inside the
container.

```bash
$ export LD_LIBRARY_PATH=/opt/tritonserver/backends/tensorflow1:$LD_LIBRARY_PATH
```

A limitation of this approach is that the custom operations must be
managed separately from the model repository itself. And more
seriously, if there are custom layer name conflicts across multiple
shared libraries there is currently no way to handle it.

When building the custom operations shared library it is important to
use the same version of TensorFlow as is being used in Triton. You can
find the TensorFlow version in the [Triton Release
Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html). A
simple way to ensure you are using the correct version of TensorFlow
is to use the [NGC TensorFlow
container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow)
corresponding to the Triton container. For example, if you are using
the 23.12 version of Triton, use the 23.12 version of the TensorFlow
container.

## PyTorch

Torchscript allows users to [add custom
operations](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)
which can then be used in Torchscript models. By using LD_PRELOAD you
can load your custom C++ operations into Triton. For example, if you
follow the instructions in the
[pytorch/extension-script](https://github.com/pytorch/extension-script)
repository and your Torchscript custom operations are compiled into
libpytcustom.so, starting Triton with the following command makes
those operations available to all PyTorch models. Since all Pytorch
custom operations depend on one or more PyTorch shared libraries
that must be available to the custom shared library when it is
loading. In practice this means that you must make sure that
/opt/tritonserver/backends/pytorch is on the library path while
launching the server. There are several ways to control the library path
and a common one is to use the LD_LIBRARY_PATH.

```bash
$ LD_LIBRARY_PATH=/opt/tritonserver/backends/pytorch:$LD_LIBRARY_PATH LD_PRELOAD=libpytcustom.so:${LD_PRELOAD} tritonserver --model-repository=/tmp/models ...
```

A limitation of this approach is that the custom operations must be
managed separately from the model repository itself. And more
seriously, if there are custom layer name conflicts across multiple
shared libraries or the handles used to register them in PyTorch there
is currently no way to handle it.

Starting with the 20.07 release of Triton the [TorchVision
operations](https://github.com/pytorch/vision) will be included with
the PyTorch backend and hence they do not have to be explicitly added
as custom operations.

When building the custom operations shared library it is important to
use the same version of PyTorch as is being used in Triton. You can
find the PyTorch version in the [Triton Release
Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html). A
simple way to ensure you are using the correct version of PyTorch is
to use the [NGC PyTorch
container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)
corresponding to the Triton container. For example, if you are using
the 23.12 version of Triton, use the 23.12 version of the PyTorch
container.

## ONNX

ONNX Runtime allows users to [add custom
operations](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html)
which can then be used in ONNX models. To register your custom
operations library you need to include it in the model configuration
as an additional field. For example, if you follow [this
example](https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/test/shared_lib/test_inference.cc)
from the
[microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
repository and your ONNXRuntime custom operations are compiled into
libonnxcustom.so, adding the following to the model configuration of
your model makes those operations available to that specific ONNX
model.

```bash
$ model_operations { op_library_filename: "/path/to/libonnxcustom.so" }
```

When building the custom operations shared library it is important to
use the same version of ONNXRuntime as is being used in Triton. You
can find the ONNXRuntime version in the [Triton Release
Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/index.html).
