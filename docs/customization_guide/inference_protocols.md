<!--
# Copyright 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
API](#in-process-triton-server-api) or its
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
[Java bindings for In-Process Triton Server API](#java-bindings-for-in-process-triton-server-api)
are built on top of `tritonserver.h` and can be used for Java applications that
need to use Tritonserver in-process.

All capabilities of Triton server are encapsulated in the shared
library and are exposed via the Server API. The `tritonserver`
executable implements HTTP/REST and GRPC endpoints and uses the Server
API to communicate with core Triton logic. The primary source files
for the endpoints are [grpc_server.cc](https://github.com/triton-inference-server/server/blob/main/src/grpc/grpc_server.cc) and
[http_server.cc](https://github.com/triton-inference-server/server/blob/main/src/http_server.cc). In these source files you can
see the Server API being used.

You can use the Server API in your own application as well. A simple
example using the Server API can be found in
[simple.cc](https://github.com/triton-inference-server/server/blob/main/src/simple.cc).

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
Triton core logic, instead all communication between your application
and the Triton core logic must take place via the [Server
API](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h).

The top-level abstraction used by Server API is `TRITONSERVER_Server`,
which represents the Triton core logic that is capable of implementing
all of the features and capabilities of Triton. A
`TRITONSERVER_Server` object is created by calling
`TRITONSERVER_ServerNew` with a set of options that indicate how the
object should be initialized.  Use of `TRITONSERVER_ServerNew` is
demonstrated in [simple.cc](https://github.com/triton-inference-server/server/blob/main/src/simple.cc). Once you have created a
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
`FAIL_IF_ERR` shown in [common.h](https://github.com/triton-inference-server/server/blob/main/src/common.h) are useful for
managing error object lifetimes.

#### Versioning and Backwards Compatibility

A typical pattern, demonstrated in [simple.cc](https://github.com/triton-inference-server/server/blob/main/src/simple.cc) and
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
[simple.cc](https://github.com/triton-inference-server/server/blob/main/src/simple.cc) and all are documented in
[tritonserver.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h).

#### Inference APIs

Performing an inference request requires the use of many Server API
functions and objects, as demonstrated in
[simple.cc](https://github.com/triton-inference-server/server/blob/main/src/simple.cc). The general usage requires the
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
  [simple.cc](https://github.com/triton-inference-server/server/blob/main/src/simple.cc), these callback functions are
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
  and why it is useful are shown in [simple.cc](https://github.com/triton-inference-server/server/blob/main/src/simple.cc).

* Ask Triton to execute the inference request using
  `TRITONSERVER_ServerInferAsync`. `TRITONSERVER_ServerInferAsync` is
  a asynchronous call that returns immediately. The inference response
  is returned via a callback into your application. You register this
  callback using `TRITONSERVER_InferenceRequestSetResponseCallback`
  before you invoke `TRITONSERVER_ServerInferAsync`. In
  [simple.cc](https://github.com/triton-inference-server/server/blob/main/src/simple.cc) this callback is
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

  Note that the [simple.cc](https://github.com/triton-inference-server/server/blob/main/src/simple.cc) example uses a
  std::promise to simply wait for the response, but synchronizing
  response handling in this way is not required. You can have multiple
  inference requests in flight at the same time and can issue
  inference requests from the same thread or from multiple different
  threads.
allows Triton to be linked directly to a C/C++ application. The API
is documented in
[tritonserver.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h).

A simple example using the C API can be found in
[simple.cc](https://github.com/triton-inference-server/server/blob/main/src/simple.cc).  A more complicated example can be
found in the source that implements the HTTP/REST and GRPC endpoints
for Triton. These endpoints use the C API to communicate with the core
of Triton. The primary source files for the endpoints are
[grpc_server.cc](https://github.com/triton-inference-server/server/blob/main/src/grpc/grpc_server.cc) and
[http_server.cc](https://github.com/triton-inference-server/server/blob/main/src/http_server.cc).

## Java bindings for In-Process Triton Server API

The Triton Inference Server uses [Java CPP](https://github.com/bytedeco/javacpp)
to create bindings around Tritonserver to create Java API.

The API is documented in
[tritonserver.java](https://github.com/bytedeco/javacpp-presets/blob/master/tritonserver/src/gen/java/org/bytedeco/tritonserver/global/tritonserver.java).
Alternatively, the user can refer to the web version [API docs](http://bytedeco.org/javacpp-presets/tritonserver/apidocs/)
generated from `tritonserver.java`.
**Note:** Currently, `tritonserver.java` contains bindings for both the `In-process C-API`
and the bindings for `C-API Wrapper`. More information about the [developer_tools/server C-API wrapper](https://github.com/triton-inference-server/developer_tools/blob/main/server/README.md) can be found in the [developer_tools repository](https://github.com/triton-inference-server/developer_tools/).

A simple example using the Java API can be found in
[Samples folder](https://github.com/bytedeco/javacpp-presets/tree/master/tritonserver/samples)
which includes `Simple.java` which is similar to
[`simple.cc`](https://github.com/triton-inference-server/server/blob/main/src/simple.cc).
Please refer to
[sample usage documentation](https://github.com/bytedeco/javacpp-presets/tree/master/tritonserver#sample-usage)
to learn about how to build and run `Simple.java`.

In the [QA folder](https://github.com/triton-inference-server/server/blob/main/qa), folders starting with L0_java include Java API tests.
These can be useful references for getting started, such as the
[ResNet50 test](https://github.com/triton-inference-server/server/blob/main/qa/L0_java_resnet).

### Java API setup instructions

To use the Tritonserver Java API, you will need to have the Tritonserver library
and dependencies installed in your environment. There are two ways to do this:

1. Use a Tritonserver docker container with
   1. `.jar` Java bindings to C API (recommended)
   2. maven and build bindings yourself
2. Build Triton from your environment without Docker (not recommended)

#### Run Tritonserver container and install dependencies

To set up your environment with Triton Java API, please follow the following steps:
1. First run Docker container:
```
 $ docker run -it --gpus=all -v ${pwd}:/workspace nvcr.io/nvidia/tritonserver:<your container version>-py3 bash
```
2. Install `jdk`:
```bash
 $ apt update && apt install -y openjdk-11-jdk
```
3. Install `maven` (only if you want to build the bindings yourself):
```bash
$ cd /opt/tritonserver
 $ wget https://archive.apache.org/dist/maven/maven-3/3.8.4/binaries/apache-maven-3.8.4-bin.tar.gz
 $ tar zxvf apache-maven-3.8.4-bin.tar.gz
 $ export PATH=/opt/tritonserver/apache-maven-3.8.4/bin:$PATH
```

#### Run Java program with Java bindings Jar

After ensuring that Tritonserver and dependencies are installed, you can run your
Java program with the Java bindings with the following steps:

1. Place Java bindings into your environment. You can do this by either:

   a. Building Java API bindings with provided build script:
      ```bash
      # Clone Triton client repo. Recommended client repo tag is: main
      $ git clone --single-branch --depth=1 -b <client repo tag>
                     https://github.com/triton-inference-server/client.git clientrepo
      # Run build script
      ## For In-Process C-API Java Bindings
      $ source clientrepo/src/java-api-bindings/scripts/install_dependencies_and_build.sh
      ## For C-API Wrapper (Triton with C++ bindings) Java Bindings
      $ source clientrepo/src/java-api-bindings/scripts/install_dependencies_and_build.sh --enable-developer-tools-server
      ```
      This will install the Java bindings to `/workspace/install/java-api-bindings/tritonserver-java-bindings.jar`

   *or*

   b. Copying "Uber Jar" from Triton SDK container to your environment
      ```bash
      $ id=$(docker run -dit nvcr.io/nvidia/tritonserver:<triton container version>-py3-sdk bash)
      $ docker cp ${id}:/workspace/install/java-api-bindings/tritonserver-java-bindings.jar <Uber Jar directory>/tritonserver-java-bindings.jar
      $ docker stop ${id}
      ```
      **Note:** `tritonserver-java-bindings.jar` only includes the `In-Process Java Bindings`. To use the `C-API Wrapper Java Bindings`, please use the build script.
2. Use the built "Uber Jar" that contains the Java bindings
   ```bash
   $ java -cp <Uber Jar directory>/tritonserver-java-bindings.jar <your Java program>
   ```

#### Build Java bindings and run Java program with Maven

If you want to make changes to the Java bindings, then you can use Maven to
build yourself. You can refer to part 1.a of [Run Java program with Java
bindings Jar](#run-java-program-with-java-bindings-jar) to also build the jar
yourself without any modifications to the Tritonserver bindings in
JavaCPP-presets.
You can do this using the following steps:

1. Create the JNI binaries in your local repository (`/root/.m2/repository`)
   with [`javacpp-presets/tritonserver`](https://github.com/bytedeco/javacpp-presets/tree/master/tritonserver).
   For C-API Wrapper Java bindings (Triton with C++ bindings), you need to
   install some build specific dependencies including cmake and rapidjson.
   Refer to [java installation script](https://github.com/triton-inference-server/client/blob/main/src/java-api-bindings/scripts/install_dependencies_and_build.sh)
   for dependencies you need to install and modifications you need to make for your container.
After installing dependencies, you can build the tritonserver project on javacpp-presets:
```bash
 $ git clone https://github.com/bytedeco/javacpp-presets.git
 $ cd javacpp-presets
 $ mvn clean install --projects .,tritonserver
 $ mvn clean install -f platform --projects ../tritonserver/platform -Djavacpp.platform=linux-x86_64
```
2. Create your custom `*.pom` file for Maven. Please refer to
   [samples/simple/pom.xml](https://github.com/bytedeco/javacpp-presets/blob/master/tritonserver/samples/simple/pom.xml) as
   reference for how to create your pom file.
3. After creating your `pom.xml` file you can build your application with:
```bash
 $ mvn compile exec:java -Djavacpp.platform=linux-x86_64 -Dexec.args="<your input args>"
```
