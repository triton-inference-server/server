<!--
# Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

## C API

The Triton Inference Server provides a backwards-compatible C API that
allows Triton to be linked directly to a C/C++ application. The API
is documented in
[tritonserver.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h).

A simple example using the C API can be found in
[simple.cc](../src/servers/simple.cc).  A more complicated example can
be found in the source that implements the HTTP/REST and GRPC
endpoints for Triton. These endpoints use the C API to communicate
with the core of Triton. The primary source files for the endpoints
are [grpc_server.cc](../src/servers/grpc_server.cc) and
[http_server.cc](../src/servers/http_server.cc).

## Java API (pre-release)

The Triton Inference Server uses [Java CPP](https://github.com/bytedeco/javacpp)
to create bindings around Tritonserver to create Java API. It requires the user
to link a pre-built library to their application using Maven.   

The API is documented in 
[tritonserver.java](https://github.com/bytedeco/javacpp-presets/blob/master/tritonserver/src/gen/java/org/bytedeco/tritonserver/global/tritonserver.java).
Alternatively, the user can refer to the web version [API docs](http://bytedeco.org/javacpp-presets/tritonserver/apidocs/)
generated from `tritonserver.java`.
A simple example using the Java API can be found in
[Samples folder](https://github.com/bytedeco/javacpp-presets/tree/master/tritonserver/samples)
which includes `Simple.java` which is similar to 
[`simple.cc`](https://github.com/triton-inference-server/server/blob/main/src/servers/simple.cc). 
Please refer to
[sample usage documentation](https://github.com/bytedeco/javacpp-presets/tree/master/tritonserver#sample-usage)
to learn about how to build and run `Simple.java`.

### Java API setup instructions

The current snapshot is based on Triton container version `21.12`. To set up your
enviroment with Triton Java API, please follow the following steps:
1. First run Docker container:
```
 $ docker run -it --gpus=all -v ${pwd}:/workspace nvcr.io/nvidia/tritonserver:21.12-py3 bash
```
2. Then install `java sdk` and `maven`:
```
 $ cd /opt/tritonserver
 $ apt update && apt install -y openjdk-11-jdk
 $ wget https://archive.apache.org/dist/maven/maven-3/3.8.4/binaries/apache-maven-3.8.4-bin.tar.gz
 $ tar zxvf apache-maven-3.8.4-bin.tar.gz
 $ export PATH=/opt/tritonserver/apache-maven-3.8.4/bin:$PATH
```
3. Then you can create the JNI binaries in your local repository (`/root/.m2/repository`) 
   with [`javacpp-presets/tritonserver`](https://github.com/bytedeco/javacpp-presets/tree/master/tritonserver)
```
 $ git clone https://github.com/bytedeco/javacpp-presets.git
 $ cd javacpp-presets
 $ mvn clean install --projects .,tritonserver
 $ mvn clean install -f platform --projects ../tritonserver/platform -Djavacpp.platform.host
```
4. Maven requires a `pom.xml` file to compile. Please refer to 
   [samples/pom.xml](https://github.com/bytedeco/javacpp-presets/blob/master/tritonserver/samples/pom.xml)
   as reference for how to create your pom file.
5. After creating your `pom.xml` file you can build your application with:
```
 $ mvn compile exec:java -Djavacpp.platform=linux-x86_64 -Dexec.args="<your input args>"
```
