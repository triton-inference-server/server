<!--
# Copyright 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

Clients can communicate with Triton using either an [HTTP/REST
protocol](#httprest-and-grpc-protocols), a [GRPC
protocol](#httprest-and-grpc-protocols), or by an [in-process C
API](#in-process-triton-server-api).

## HTTP/REST and GRPC Protocols

Triton exposes both HTTP/REST and GRPC endpoints based on [standard
inference
protocols](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2)
that have been proposed by the [KServe
project](https://github.com/kserve). To fully enable all capabilities
Triton also implements a number [HTTP/REST and GRPC
extensions](https://github.com/triton-inference-server/server/tree/main/docs/protocol)
to the KServe inference protocol.

The HTTP/REST and GRPC protcols provide endpoints to check server and
model health, metadata and statistics. Additional endpoints allow
model loading and unloading, and inferencing. See the KServe and
extension documentation for details.

### HTTP Options
Triton provides the following configuration options for server-client network transactions over HTTP protocol.

#### Compression

Triton allows the on-wire compression of request/response on HTTP through its clients. See [HTTP Compression](https://github.com/triton-inference-server/client/tree/main#compression) for more details.

### GRPC Options
Triton exposes various GRPC parameters for configuring the server-client network transactions. For usage of these options, refer to the output from `tritonserver --help`.

#### SSL/TLS

These options can be used to configure a secured channel for communication. The server-side options include:

* `--grpc-use-ssl`
* `--grpc-use-ssl-mutual`
* `--grpc-server-cert`
* `--grpc-server-key`
* `--grpc-root-cert`

For client-side documentation, see [Client-Side GRPC SSL/TLS](https://github.com/triton-inference-server/client/tree/main#ssltls)

For more details on overview of authentication in gRPC, refer [here](https://grpc.io/docs/guides/auth/).

#### Compression

Triton allows the on-wire compression of request/response messages by exposing following option on server-side:

* `--grpc-infer-response-compression-level`

For client-side documentation, see [Client-Side GRPC Compression](https://github.com/triton-inference-server/client/tree/main#compression-1)

Compression can be used to reduce the amount of bandwidth used in server-client communication. For more details, see [gRPC Compression](https://grpc.github.io/grpc/core/md_doc_compression.html).

#### GRPC KeepAlive

Triton exposes GRPC KeepAlive parameters with the default values for both
client and server described [here](https://github.com/grpc/grpc/blob/master/doc/keepalive.md).

These options can be used to configure the KeepAlive settings:

* `--grpc-keepalive-time`
* `--grpc-keepalive-timeout`
* `--grpc-keepalive-permit-without-calls`
* `--grpc-http2-max-pings-without-data`
* `--grpc-http2-min-recv-ping-interval-without-data`
* `--grpc-http2-max-ping-strikes`

For client-side documentation, see [Client-Side GRPC KeepAlive](https://github.com/triton-inference-server/client/blob/main/README.md#grpc-keepalive).

## In-Process Triton Server API

The Triton Inference Server provides a backwards-compatible C API that
allows Triton to be linked directly into a C/C++ application. This API
is called the "Triton Server API" or just "Server API" for short. The
API is implemented in the Triton shared library which is built from
source contained in the [core
repository](https://github.com/triton-inference-server/core). On Linux
this library is libtritonserver.so and on Windows it is
tritonserver.dll. In the Triton Docker image the shared library is
found in /opt/tritonserver/lib. The header file that defines and
documents the Server API is
[tritonserver.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h).

All capabilities of Triton server are encapsulated in the shared
library and are exposed via the Server API. The `tritonserver`
executable implements HTTP/REST and GRPC endpoints and uses the Server
API to communicate with core Triton logic. The primary source files
for the endpoints are [grpc_server.cc](../src/grpc_server.cc) and
[http_server.cc](../src/http_server.cc). In these source files you can
see the Server API being used.

You can use the Server API in your own application as well. A simple
example using the Server API can be found in
[simple.cc](../src/simple.cc).

### API Description

Triton server functionality is encapsulated in a shared library which
is built from source contained in the [core
repository](https://github.com/triton-inference-server/core). You can
include the full capabilities of Triton by linking the shared library
into your application and by using the C API defined in
[tritonserver.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h).

When you link the Triton shared library into your application you are
*not* spawning a separate Triton process, instead, you are including
the Triton core logic directly in your application. The Triton
HTTP/REST or GRPC protocols are not used to communicate with this
Triton core logic, instead all communication between your appliation
and the Triton core logic must take place via the [Server
API](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h).

The top-level abstraction used by Server API is `TRITONSERVER_Server`,
which represents the Triton core logic that is capable of implementing
all of the features and capabilities of Triton. A
`TRITONSERVER_Server` object is created by calling
`TRITONSERVER_ServerNew` with a set of options that indicate how the
object should be initialized.  Use of `TRITONSERVER_ServerNew` is
demonstrated in [simple.cc](../src/simple.cc). Once you have created a
`TRITONSERVER_Server` object, you can begin using the rest of the
Server API as described below.

#### Error Handling

Most Server API functions return an error object indicating success or
failure. Success is indicated by return `nullptr` (`NULL`). Failure is
indicated by returning a `TRITONSERVER_Error` object. The error code
and message can be retrieved from a `TRITONSERVER_Error` object with
`TRITONSERVER_ErrorCode` and `TRITONSERVER_ErrorMessage`.

The lifecycle and ownership of all Server API objects is documented in
[tritonserver.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h). For
`TRITONSERVER_Error`, ownership of the object passes to the caller of
the Server API function. As a result, your application is responsible
for managing the lifecycle of the returned `TRITONSERVER_Error`
object. You must delete the error object using
`TRITONSERVER_ErrorDelete` when you are done using it. Macros such as
`FAIL_IF_ERR` shown in [common.h](../src/common.h) are useful for
managing error object lifetimes.

#### Versioning and Backwards Compatibility

A typical pattern, demonstrated in [simple.cc](../src/simple.cc) and
shown below, shows how you can compare the Server API version provided
by the shared library against the Server API version that you compiled
your application against. The Server API is backwards compatible, so
as long as the major version provided by the shared library matches
the major version that you compiled against, and the minor version
provided by the shared library is greater-than-or-equal to the minor
version that you compiled against, then your application can use the
Server API.

```
#include "tritonserver.h"
// Error checking removed for clarity...
uint32_t api_version_major, api_version_minor;
TRITONSERVER_ApiVersion(&api_version_major, &api_version_minor);
if ((TRITONSERVER_API_VERSION_MAJOR != api_version_major) ||
    (TRITONSERVER_API_VERSION_MINOR > api_version_minor)) {
  // Error, the shared library implementing the Server API is older than
  // the version of the Server API that you compiled against.
}
```

#### Non-Inference APIs

The Server API contains functions for checking health and readiness,
getting model information, getting model statistics and metrics,
loading and unloading models, etc. The use of these functions is
straightforward and some of these functions are demonstrated in
[simple.cc](../src/simple.cc) and all are documented in
[tritonserver.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h).

#### Inference APIs

Performing an inference request requires the use of many Server API
functions and objects, as demonstrated in
[simple.cc](../src/simple.cc). The general usage requires the
following steps.

* Create a `TRITONSERVER_ResponseAllocator` using
  `TRITONSERVER_ResponseAllocatorNew`.  You can use the same response
  allocator for all of your inference requests, or you can create
  multiple response allocators.  When Triton produces an output
  tensor, it needs a memory buffer into which it can store the
  contents of that tensor. Triton defers the allocation of these
  output buffers by invoking callback functions in your
  application. You communicate these callback functions to Triton with
  the `TRITONSERVER_ResponseAllocator` object. You must implement two
  callback functions, one for buffer allocation and one for buffer
  free. The signatures for these functions are
  `TRITONSERVER_ResponseAllocatorAllocFn_t` and
  `TRITONSERVER_ResponseAllocatorReleaseFn_t` as defined in
  [tritonserver.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h). In
  [simple.cc](../src/simple.cc), these callback functions are
  implemented as `ResponseAlloc` and `ResponseRelease`.

* Create an inference request as a `TRITONSERVER_InferenceRequest`
  object. The inference request is where you specify what model you
  want to use, the input tensors and their values, the output tensors
  that you want returned, and other request parameters. You create an
  inference request using `TRITONSERVER_InferenceRequestNew`. You
  create each input tensor in the request using
  `TRITONSERVER_InferenceRequestAddInput` and set the data for the
  input tensor using `TRITONSERVER_InferenceRequestAppendInputData`
  (or one of the `TRITONSERVER_InferenceRequestAppendInputData*`
  variants defined in
  [tritonserver.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h)). By
  default, Triton will return all output tensors, but you can limit
  Triton to only return some outputs by using
  `TRITONSERVER_InferenceRequestAddRequestedOutput`.

  To correctly manage the lifecycle of the inference request, you must
  use `TRITONSERVER_InferenceRequestSetReleaseCallback` to set a
  callback into a function in your application. This callback will be
  invoke by Triton to return ownership of the
  `TRITONSERVER_InferenceRequest` object. Typically, in this callback
  you will just delete the `TRITONSERVER_InferenceRequest` object by
  using `TRITONSERVER_InferenceRequestDelete`. But you may also
  implement a different lifecycle management; for example, if you are
  reusing inference request objects you would want to make the object
  available for reuse.

  You can optionally use `TRITONSERVER_InferenceRequestSetId` to set a
  user-defined ID on the request. This ID is not used by Triton but
  will be returned in the response.

  You can reuse an existing `TRITONSERVER_InferenceRequest` object for
  a new inference request. A couple of examples of how this is done
  and why it is useful are shown in [simple.cc](../src/simple.cc).

* Ask Triton to execute the inference request using
  `TRITONSERVER_ServerInferAsync`. `TRITONSERVER_ServerInferAsync` is
  a asychronous call that returns immediately. The inference response
  is returned via a callback into your application. You register this
  callback using `TRITONSERVER_InferenceRequestSetResponseCallback`
  before you invoke `TRITONSERVER_ServerInferAsync`. In
  [simple.cc](../src/simple.cc) this callback is
  `InferResponseComplete`.

  When you invoke `TRITONSERVER_ServerInferAsync` and it returns
  without error, you are passing ownership of the
  `TRITONSERVER_InferenceRequest` object to Triton, and so you must
  not access that object in any way until Triton returns ownership to
  you via the callback you registered with
  `TRITONSERVER_InferenceRequestSetReleaseCallback`.

* Process the inference response. The inference response is returned
  to the callback function you registered with
  `TRITONSERVER_InferenceRequestSetResponseCallback`. Your callback
  receives the response as a `TRITONSERVER_InferenceResponse`
  object. Your callback takes ownership of the
  `TRITONSERVER_InferenceResponse` object and so must free it with
  `TRITONSERVER_InferenceResponseDelete` when it is no longer needed.

  The first step in processing a response is to use
  `TRITONSERVER_InferenceResponseError` to check if the response is
  returning an error or if it is returning valid results. If the
  response is valid you can use
  `TRITONSERVER_InferenceResponseOutputCount` to iterate over the
  output tensors, and `TRITONSERVER_InferenceResponseOutput` to get
  information about each output tensor.

  Note that the [simple.cc](../src/simple.cc) example uses a
  std::promise to simply wait for the response, but sychronizing
  response handling in this way is not required. You can have multiple
  inference requests in flight at the same time and can issue
  inference requests from the same thread or from multiple different
  threads.
