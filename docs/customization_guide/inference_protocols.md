<!--
# Copyright 2018-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
API](inprocess_c_api.md#in-process-triton-server-api) or its
[C++ wrapper](https://github.com/triton-inference-server/developer_tools/tree/main/server).

## HTTP/REST and GRPC Protocols

Triton exposes both HTTP/REST and GRPC endpoints based on [standard
inference
protocols](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2)
that have been proposed by the [KServe
project](https://github.com/kserve). To fully enable all capabilities
Triton also implements [HTTP/REST and GRPC
extensions](https://github.com/triton-inference-server/server/tree/main/docs/protocol)
to the KServe inference protocol. GRPC protocol also provides a
bi-directional streaming version of the inference RPC to allow a
sequence of inference requests/responses to be sent over a
GRPC stream. We typically recommend using the unary version for
inference requests. The streaming version should be used only if the
situation demands it. Some of such use cases can be:

* Assume a system with multiple Triton server instances running
  behind a Load Balancer. If a sequence of inference requests is
  needed to hit the same Triton server instance, a GRPC stream
  will hold a single connection throughout the lifetime and hence
  ensure the requests are delivered to the same Triton instance.
* If the order of requests/responses needs to be preserved over
  the network, a GRPC stream will ensure that the server receives
  the requests in the same order as they were sent from the
  client.

The HTTP/REST and GRPC protocols also provide endpoints to check
server and model health, metadata and statistics. Additional
endpoints allow model loading and unloading, and inferencing. See
the KServe and extension documentation for details.

### HTTP Options
Triton provides the following configuration options for server-client network transactions over HTTP protocol.

#### Compression

Triton allows the on-wire compression of request/response on HTTP through its clients. See [HTTP Compression](../client/README.md#compression) for more details.

#### Mapping Triton Server Error Codes to HTTP Status Codes

This table maps various Triton Server error codes to their corresponding HTTP status
codes. It can be used as a reference guide for understanding how Triton Server errors
are handled in HTTP responses.


| Triton Server Error Code                      | HTTP Status Code   | Description          |
| ----------------------------------------------| -------------------| ---------------------|
| `TRITONSERVER_ERROR_INTERNAL`                 | 500                | Internal Server Error|
| `TRITONSERVER_ERROR_NOT_FOUND`                | 404                | Not Found            |
| `TRITONSERVER_ERROR_UNAVAILABLE`              | 503                | Service Unavailable  |
| `TRITONSERVER_ERROR_UNSUPPORTED`              | 501                | Not Implemented      |
| `TRITONSERVER_ERROR_UNKNOWN`,<br>`TRITONSERVER_ERROR_INVALID_ARG`,<br>`TRITONSERVER_ERROR_ALREADY_EXISTS`,<br>`TRITONSERVER_ERROR_CANCELLED` | `400` | Bad Request (default for other errors)      |

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

#### GRPC Status Codes

Triton implements GRPC error handling for streaming requests when a specific flag is enabled through headers. Upon encountering an error, Triton returns the appropriate GRPC error code and subsequently closes the stream.

* `triton_grpc_error` : The header value needs to be set to true while starting the stream.

GRPC status codes can be used for better visibility and monitoring. For more details, see [gRPC Status Codes](https://grpc.io/docs/guides/status-codes/)

For client-side documentation, see [Client-Side GRPC Status Codes](https://github.com/triton-inference-server/client/tree/main#GRPC-Status-Codes)

#### GRPC Inference Handler Threads

In general, using 2 threads per completion queue seems to give the best performance, see [gRPC Performance Best Practices] (https://grpc.io/docs/guides/performance/#c). However, in cases where the performance bottleneck is at the request handling step (e.g. ensemble models), increasing the number of gRPC inference handler threads may lead to a higher throughput.

* `--grpc-infer-thread-count`: 2 by default.

Note: More threads don't always mean better performance.

### Limit Endpoint Access (BETA)

Triton users may want to restrict access to protocols or APIs that are
provided by the GRPC or HTTP endpoints of a server. For example, users
can provide one set of access credentials for inference APIs and
another for model control APIs such as model loading and unloading.

The following options can be specified to declare a restricted
protocol group (GRPC) or restricted API group (HTTP):

```
--grpc-restricted-protocol=<protocol_1>,<protocol_2>,...:<restricted-key>=<restricted-value>
--http-restricted-api=<API_1>,API_2>,...:<restricted-key>=<restricted-value>
```

The option can be specified multiple times to specifies multiple groups of
protocols or APIs with different restriction settings.

* `protocols / APIs` : A comma-separated list of protocols / APIs to be included in this
group. Note that currently a given protocol / API is not allowed to be included in
multiple groups. The following protocols / APIs are recognized:

  * `health` : Health endpoint defined for [HTTP/REST](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#health) and [GRPC](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#health-1). For GRPC endpoint, this value also exposes [GRPC health check protocol](https://github.com/triton-inference-server/common/blob/main/protobuf/health.proto).
  * `metadata` : Server / model metadata endpoints defined for [HTTP/REST](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#server-metadata) and [GRPC](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#server-metadata-1).
  * `inference` : Inference endpoints defined for [HTTP/REST](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference) and [GRPC](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference-1).
  * `shared-memory` : [Shared-memory endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_shared_memory.md).
  * `model-config` : [Model configuration endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_configuration.md).
  * `model-repository` : [Model repository endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md).
  * `statistics` : [statistics endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_statistics.md).
  * `trace` : [trace endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_trace.md).
  * `logging` : [logging endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_logging.md).

* `restricted-key` : The GRPC / HTTP request header
to be checked when a request is received. The
completed header for GRPC will be in the form of
`triton-grpc-protocol-<restricted-key>`. The completed header for HTTP
will be in the form of `<restricted-key>`.

* `restricted-value` : The header value required to access the specified protocols.

#### Example

To start the server with a set of protocols and APIs restricted for
`admin` usage and the rest of the protocols and APIs left unrestricted
use the following command line arguments:


```
tritonserver --grpc-restricted-protocol=shared-memory,model-config,model-repository,statistics,trace:<admin-key>=<admin-value> \
             --http-restricted-api=shared-memory,model-config,model-repository,statistics,trace:<admin-key>=<admin-value> ...
```

GRPC requests to `admin` protocols require that an additional header
`triton-grpc-protocol-<admin-key>` is provided with value
`<admin-value>`. HTTP requests to `admin` APIs required that an
additional header `<admin-key>` is provided with value `<admin-value>`.
