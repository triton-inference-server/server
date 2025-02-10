..
.. Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. Redistribution and use in source and binary forms, with or without
.. modification, are permitted provided that the following conditions
.. are met:
..  * Redistributions of source code must retain the above copyright
..    notice, this list of conditions and the following disclaimer.
..  * Redistributions in binary form must reproduce the above copyright
..    notice, this list of conditions and the following disclaimer in the
..    documentation and/or other materials provided with the distribution.
..  * Neither the name of NVIDIA CORPORATION nor the names of its
..    contributors may be used to endorse or promote products derived
..    from this software without specific prior written permission.
..
.. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
.. EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
.. IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
.. PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
.. CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
.. EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
.. PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
.. PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
.. OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
.. (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
.. OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

####
In-Process Triton Server API
####


The Triton Inference Server provides a backwards-compatible C API/ python-bindings/java-bindings that
allows Triton to be linked directly into a C/C++/java/python application. This API
is called the "Triton Server API" or just "Server API" for short. The
API is implemented in the Triton shared library which is built from
source contained in the `core
repository <https://github.com/triton-inference-server/core>`__. On Linux
this library is libtritonserver.so and on Windows it is
tritonserver.dll. In the Triton Docker image the shared library is
found in /opt/tritonserver/lib. The header file that defines and
documents the Server API is
`tritonserver.h <https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h>`__.
`Java bindings for In-Process Triton Server API <../customization_guide/inprocess_java_api.html#java-bindings-for-in-process-triton-server-api>`__
are built on top of `tritonserver.h` and can be used for Java applications that
need to use Tritonserver in-process.

All capabilities of Triton server are encapsulated in the shared
library and are exposed via the Server API. The `tritonserver`
executable implements HTTP/REST and GRPC endpoints and uses the Server
API to communicate with core Triton logic. The primary source files
for the endpoints are `grpc_server.cc <https://github.com/triton-inference-server/server/blob/main/src/grpc/grpc_server.cc>`__ and
`http_server.cc <https://github.com/triton-inference-server/server/blob/main/src/http_server.cc>`__. In these source files you can
see the Server API being used.

You can use the Server API in your own application as well. A simple
example using the Server API can be found in
`simple.cc <https://github.com/triton-inference-server/server/blob/main/src/simple.cc>`__.

.. toctree::
   :maxdepth: 1
   :hidden:

   C/C++ <../customization_guide/inprocess_c_api.md>
   python
   Java <../customization_guide/inprocess_java_api.md>