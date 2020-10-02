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

# Inference Protocols and APIs

Clients can communicate with Triton using either an HTTP/REST or GRPC
protocol, or by a C API.

## HTTP/REST and GRPC Protocols

Triton exposes both HTTP/REST and GRPC endpoints based on [standard
inference
protocols](https://github.com/kubeflow/kfserving/tree/master/docs/predict-api/v2)
that have been proposed by the [KFServing
project](https://github.com/kubeflow/kfserving). To fully enable all
capabilities Triton also implements a number [HTTP/REST and GRPC
extensions](https://github.com/triton-inference-server/server/tree/master/docs/protocol).
to the KFServing inference protocol.

The HTTP/REST and GRPC protcols provide endpoints to check server and
model health, metadata and statistics. Additional endpoints allow
model loading and unloading, and inferencing. See the KFServing and
extension documentation for details.

## C API

The Triton Inference Server provides a backwards-compatible C API that
allows Triton to be linked directly into a C/C++ application. The API
is documented in
[tritonserver.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h).

A simple example using the C API can be found in
[simple.cc](../src/servers/simple.cc).  A more complicated example can
be found in the source that implements the HTTP/REST and GRPC
endpoints for Triton. These endpoints use the C API to communicate
with the core of Triton. The primary source files for the endpoints
are [grpc_server.cc](../src/servers/grpc_server.cc) and
[http_server.cc](../src/servers/http_server.cc).
