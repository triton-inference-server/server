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
compiled into libtrtcustom.so, starting the inference server with the
following command makes those custom layers available to all TensorRT
models loaded into the server::

  $ LD_PRELOAD=libtrtcustom.so trtserver --model-repository=/tmp/models ...

A limitation of this approach is that the custom layers must be
managed separately from the model repository itself. And more
seriously, if there are custom layer name conflicts across multiple
shared libraries there is currently no way to handle it.

TensorFlow
----------

Tensorflow allows users to `add custom operations
<https://www.tensorflow.org/guide/extend/op>`_ which can then be used
in TensorFlow models. By using LD_PRELOAD you can load your custom
TensorFlow operations into the inference server.  For example,
assuming your TensorFlow custom operations are compiled into
libtfcustom.so, starting the inference server with the following
command makes those operations available to all TensorFlow models
loaded into the server::

  $ LD_PRELOAD=libtfcustom.so trtserver --model-repository=/tmp/models ...

A limitation of this approach is that the custom operations must be
managed separately from the model repository itself. And more
seriously, if there are custom layer name conflicts across multiple
shared libraries there is currently no way to handle it.
