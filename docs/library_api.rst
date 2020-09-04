..
  # Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

.. _section-library-api:

Library API
===========

The Triton Inference Server provides a backwards-compatible C API that
allows Triton to be linked directly into a C/C++ application. The API
is documented in `tritonserver.h
<https://github.com/triton-inference-server/common/blob/main/include/triton/common/tritonserver.h>`_.

A simple example of the library API can be found at
`src/servers/simple.cc
<https://github.com/triton-inference-server/server/blob/master/src/servers/simple.cc>`_. A
more complicated example can be found in the files that make up the
inference server executable, *tritonserver*. The tritonserver
executable implements the HTTP/REST and GRPC endpoints and uses the
library API to communicate with the core of Triton. The primary files
composing *tritonserver* are `src/servers/main.cc
<https://github.com/triton-inference-server/server/blob/master/src/servers/main.cc>`_,
`src/servers/grpc_server.cc
<https://github.com/triton-inference-server/server/blob/master/src/servers/grpc_server.cc>`_,
and `src/servers/http_server.cc
<https://github.com/triton-inference-server/server/blob/master/src/servers/http_server.cc>`_.
