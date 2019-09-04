..
  # Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

.. _section-http-and-grpc-api:

HTTP and GRPC API
=================

The TensorRT Inference Server exposes both HTTP and GRPC
endpoints. The following endpoints are exposed for each protocol.

* :ref:`section-api-health`: The server health API for determining
  server liveness and readiness.

* :ref:`section-api-status`: The server status API for getting
  information about the server and about the models being served.

* :ref:`section-api-model-control`: The server model-control API for
  explicitly loading and unloading models.

* :ref:`section-api-inference`: The inference API that accepts model
  inputs, runs inference and returns the requested outputs.

The inference server also exposes an endpoint based on GRPC streams that is
only available when using the GRPC protocol:

* :ref:`section-api-stream-inference`: The stream inference API is the same
  as the Inference API, except that once the connection is established,
  the requests are sent in the same connection until it is closed.

The HTTP endpoints can be used directly as described in this section,
but for most use-cases the preferred way to access the inference
server is via the :ref:`C++ and Python Client libraries
<section-client-libraries>`.

The GRPC endpoints can also be used via the :ref:`C++ and Python
Client libraries <section-client-libraries>` or a GRPC-generated API
can be used directly as shown in the grpc_image_client.py example.

.. _section-api-health:

Health
------

Performing an HTTP GET to /api/health/live returns a 200 status if the
server is able to receive and process requests. Any other status code
indicates that the server is still initializing or has failed in some
way that prevents it from processing requests.

Once the liveness endpoint indicates that the server is active,
performing an HTTP GET to /api/health/ready returns a 200 status if
the server is able to respond to inference requests for some or all
models (based on the inference server's -\\-strict-readiness option
explained below). Any other status code indicates that the server is
not ready to respond to some or all inference requests.

For GRPC the :cpp:var:`GRPCService
<nvidia::inferenceserver::GRPCService>` uses the
:cpp:var:`HealthRequest <nvidia::inferenceserver::HealthRequest>` and
:cpp:var:`HealthResponse <nvidia::inferenceserver::HealthResponse>`
messages to implement the endpoint.

By default, the readiness endpoint will return success if the server
is responsive and all models loaded successfully. Thus, by default,
success indicates that an inference request for any model can be
handled by the server. For some use cases, you want the readiness
endpoint to return success even if all models are not available. In
this case, use the -\\-strict-readiness=false option to cause the
readiness endpoint to report success as long as the server is
responsive (even if one or more models are not available).

.. _section-api-status:

Status
------

Performing an HTTP GET to /api/status returns status information about
the server and all the models being served. Performing an HTTP GET to
/api/status/<model name> returns information about the server and the
single model specified by <model name>. The server status is returned
in the HTTP response body in either text format (the default) or in
binary format if query parameter format=binary is specified (for
example, /api/status?format=binary). The success or failure of the
status request is indicated in the HTTP response code and the
**NV-Status** response header. The **NV-Status** response header
returns a text protobuf formatted :cpp:var:`RequestStatus
<nvidia::inferenceserver::RequestStatus>` message.

For GRPC the :cpp:var:`GRPCService
<nvidia::inferenceserver::GRPCService>` uses the
:cpp:var:`StatusRequest <nvidia::inferenceserver::StatusRequest>` and
:cpp:var:`StatusResponse <nvidia::inferenceserver::StatusResponse>`
messages to implement the endpoint. The response includes a
:cpp:var:`RequestStatus <nvidia::inferenceserver::RequestStatus>`
message indicating success or failure.

For either protocol the status itself is returned as a
:cpp:var:`ServerStatus <nvidia::inferenceserver::ServerStatus>`
message.

.. _section-api-model-control:

Model Control
-------------

Performing an HTTP POST to /api/modelcontrol/<load|unload>/<model
name> loads or unloads a model from the inference server as described
in :ref:`section-model-management`.

The success or failure of the inference request is indicated in the
HTTP response code and the **NV-Status** response header. The
**NV-Status** response header returns a text protobuf formatted
:cpp:var:`RequestStatus <nvidia::inferenceserver::RequestStatus>`
message.

For GRPC the :cpp:var:`GRPCService
<nvidia::inferenceserver::GRPCService>` uses the
:cpp:var:`ModelControlRequest
<nvidia::inferenceserver::ModelControlRequest>` and
:cpp:var:`ModelControlResponse
<nvidia::inferenceserver::ModelControlResponse>` messages to implement
the endpoint.

.. _section-api-inference:

Inference
---------

Performing an HTTP POST to /api/infer/<model name> performs inference
using the latest version of the model that is being made available by
the model's :ref:`version policy <section-version-policy>`. The latest
version is the numerically greatest version number. Performing an HTTP
POST to /api/infer/<model name>/<model version> performs inference
using a specific version of the model.

The request uses the **NV-InferRequest** header to communicate an
:cpp:var:`InferRequestHeader
<nvidia::inferenceserver::InferRequestHeader>` message that describes
the input tensors and the requested output tensors. For example, for a
resnet50 model the following **NV-InferRequest** header indicates that
a batch-size 1 request is being made with a single input named
"input", and that the result of the tensor named "output" should be
returned as the top-3 classification values::

  NV-InferRequest: batch_size: 1 input { name: "input" } output { name: "output" cls { count: 3 } }

The input tensor values are communicated in the body of the HTTP POST
request as raw binary in the order as the inputs are listed in the
request header.

The HTTP response includes an **NV-InferResponse** header that
communicates an :cpp:var:`InferResponseHeader
<nvidia::inferenceserver::InferResponseHeader>` message that describes
the outputs. For example the above response could return the
following::

  NV-InferResponse: model_name: "mymodel" model_version: 1 batch_size: 1 output { name: "output" raw { dims: 4 dims: 4 batch_byte_size: 64 } }

This response shows that the output in a tensor with shape [ 4, 4 ]
and has a size of 64 bytes. The output tensor contents are returned in
the body of the HTTP response to the POST request. For outputs where
full result tensors were requested, the result values are communicated
in the body of the response in the order as the outputs are listed in
the **NV-InferResponse** header. After those, an
:cpp:var:`InferResponseHeader
<nvidia::inferenceserver::InferResponseHeader>` message is appended to
the response body. The :cpp:var:`InferResponseHeader
<nvidia::inferenceserver::InferResponseHeader>` message is returned in
either text format (the default) or in binary format if query
parameter format=binary is specified (for example,
/api/infer/foo?format=binary).

For example, assuming an inference request for a model that has 'n'
outputs, the outputs specified in the **NV-InferResponse** header in
order are “output[0]”, ..., “output[n-1]” the response body would
contain::

  <raw binary tensor values for output[0] >
  ...
  <raw binary tensor values for output[n-1] >
  <text or binary encoded InferResponseHeader proto>

The success or failure of the inference request is indicated in the
HTTP response code and the **NV-Status** response header. The
**NV-Status** response header returns a text protobuf formatted
:cpp:var:`RequestStatus <nvidia::inferenceserver::RequestStatus>`
message.

For GRPC the :cpp:var:`GRPCService
<nvidia::inferenceserver::GRPCService>` uses the
:cpp:var:`InferRequest <nvidia::inferenceserver::InferRequest>` and
:cpp:var:`InferResponse <nvidia::inferenceserver::InferResponse>`
messages to implement the endpoint. The response includes a
:cpp:var:`RequestStatus <nvidia::inferenceserver::RequestStatus>`
message indicating success or failure, :cpp:var:`InferResponseHeader
<nvidia::inferenceserver::InferResponseHeader>` message giving
response meta-data, and the raw output tensors.

.. _section-api-stream-inference:

Stream Inference
----------------

Some applications may request that multiple requests be sent using one
persistent connection rather than potentially establishing multiple
connections. For instance, in the case where multiple instances of
TensorRT Inference Server are created with the purpose of load
balancing, requests sent in different connections may be routed to
different server instances. This scenario will not fit the need if the
requests are correlated, where they are expected to be processed by
the same model instance, like inferencing with :ref:`stateful models
<section-stateful-models>`. By using stream inference, the requests
will be sent to the same server instance once the connection is
established.

For GRPC the :cpp:var:`GRPCService
<nvidia::inferenceserver::GRPCService>` uses the
:cpp:var:`InferRequest <nvidia::inferenceserver::InferRequest>` and
:cpp:var:`InferResponse <nvidia::inferenceserver::InferResponse>`
messages to implement the endpoint. The response includes a
:cpp:var:`RequestStatus <nvidia::inferenceserver::RequestStatus>`
message indicating success or failure, :cpp:var:`InferResponseHeader
<nvidia::inferenceserver::InferResponseHeader>` message giving
response meta-data, and the raw output tensors.
