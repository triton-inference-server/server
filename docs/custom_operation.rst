..
  # Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

Custom Operations
=================

Modeling frameworks that allow custom operations are partially
supported by the TensorRT Inference Server. Custom operations can be
added to the server at build time or at server startup and are made
available to all models loaded by the server.

TensorRT
--------

TensorRT allows a user to create `custom layers
<https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#extending>`_
which can then be used in TensorRT models. For those models to run in
the inference server the custom layers must be available to the
server.

To make the custom layers available to the server, the TensorRT custom
layer implementations must be compiled into one or more shared
libraries which are then loaded into the inference server using
LD_PRELOAD. For example, assuming your TensorRT custom layers are
compiled into trtcustom.so, starting the inference server with the
following command makes those custom layers available to all TensorRT
models loaded into the server::

  $ LD_PRELOAD=trtcustom.so trtserver --model-store=/tmp/models ...

A limitation of this approach is that the custom layers must be
managed separately from the model store itself. And more seriously, if
there are custom layer name conflicts across multiple shared libraries
there is now way to handle it.

TensorFlow
----------

Tensorflow allows users to `add custom operations
<https://www.tensorflow.org/guide/extend/op>`_ which can then be used
in TensorFlow models. Currently, the only way for the inference server
to support custom operations is to build them into the inference
server.

To build a TensorFlow custom operation into the inference server, the
source code for the custom operation must be placed in
`src/operations/tensorflow
<https://github.com/NVIDIA/tensorrt-inference-server/tree/master/src/operations/tensorflow>`_
and the `BUILD
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/operations/tensorflow/BUILD>`_
file updated.

An example operation called TRTISExampleAddSub is included in the
directory in files `trtis_example_addsub_op.cc
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/operations/tensorflow/trtis_example_addsub_op.cc>`_
and `trtis_example_addsub_op.cu.cc
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/operations/tensorflow/trtis_example_addsub_op.cu.cc>`_. The
build rule for the TRTISExampleAddSub operation is placed in `BUILD
<https://github.com/NVIDIA/tensorrt-inference-server/blob/master/src/operations/tensorflow/BUILD>`_::

  tf_kernel_library(
    name = "trtis_example_addsub_op",
    srcs = ["trtis_example_addsub_op.cc"],
    gpu_srcs = ["trtis_example_addsub_op.cu.cc"],
  )

The **all_custom_ops** entry in BUILD is updated to include
*trtis_example_addsub_op*. When adding a new custom operation similar
modifications must be made to BUILD.

After making these changes, :ref:`build the server
<section-building-the-server>` and the custom operations will be
available to every TensorFlow model loaded into the server.
