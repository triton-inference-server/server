<!--
# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Decoupled Backends and Models

Triton can support [backends](https://github.com/triton-inference-server/backend)
and models that send multiple responses for a request or zero responses
for a request. A decoupled model/backend may also send responses out-of-order
relative to the order that the request batches are executed. This allows
backend to deliver response whenever it deems fit. This is specifically
useful in Automated Speech Recognition (ASR). The requests with large number
of responses, will not block the responses from other requests from being
delivered.

## Developing Decoupled Backend/Model

### C++ Backend

Read carefully about the [Triton Backend API](https://github.com/triton-inference-server/backend/blob/main/README.md#triton-backend-api),
[Inference Requests and Responses](https://github.com/triton-inference-server/backend/blob/main/README.md#inference-requests-and-responses)
and [Decoupled Responses](https://github.com/triton-inference-server/backend/blob/main/README.md#decoupled-responses).
The [repeat backend](https://github.com/triton-inference-server/repeat_backend)
and [square backend](https://github.com/triton-inference-server/square_backend)
demonstrate how the Triton Backend API can be used to implement a decoupled
backend. The example is designed to show the flexibility of the Triton API
and in no way should be used in production. This example may process multiple
batches of requests at the same time without having to increase the
[instance count](model_configuration.md#instance-groups). In real deployment,
the backend should not allow the caller thread to return from
TRITONBACKEND_ModelInstanceExecute until that instance is ready to
handle another set of requests. If not designed properly the backend
can be easily over-subscribed. This can also cause under-utilization
of features like [Dynamic Batching](model_configuration.md#dynamic-batcher)
as it leads to eager batching. 

### Python model using Python Backend

Read carefully about the [Python Backend](https://github.com/triton-inference-server/python_backend),
and specifically [`execute`](https://github.com/triton-inference-server/python_backend#execute).

The [decoupled examples](https://github.com/triton-inference-server/python_backend/tree/main/examples/decoupled)
demonstrates how decoupled API can be used to implement a decoupled
python model. As noted in the examples, these are designed to show
the flexibility of the decoupled API and in no way should be used
in production.


## Deploying Decoupled Models

The [decoupled model transaction policy](model_configuration.md#decoupled)
must be set in the provided [model configuration](model_configuration.md)
file for the model. Triton requires this information to enable special
handling required for decoupled models. Deploying decoupled models without
this configuration setting will throw errors at the runtime.

## Running Inference on Decoupled Models

[Inference Protocols and APIs](../customization_guide/inference_protocols.md) describes various ways
a client can communicate and run inference on the server. For decoupled models,
Triton's HTTP endpoint cannot be used for running inference as it supports
exactly one response per request. Even standard ModelInfer RPC in the GRPC endpoint
does not support decoupled responses. In order to run inference on a decoupled
model, the client must use the bi-directional streaming RPC. See
[here](https://github.com/triton-inference-server/common/blob/main/protobuf/grpc_service.proto)
for more details. The [decoupled_test.py](https://github.com/triton-inference-server/server/blob/main/qa/L0_decoupled/decoupled_test.py) demonstrates
how the gRPC streaming can be used to infer decoupled models.

If using [Triton's in-process C API](../customization_guide/inference_protocols.md#in-process-triton-server-api),
your application should be cognizant that the callback function you registered with 
`TRITONSERVER_InferenceRequestSetResponseCallback` can be invoked any number of times,
each time with a new response. You can take a look at [grpc_server.cc](https://github.com/triton-inference-server/server/blob/main/src/grpc_server.cc)

### Knowing When a Decoupled Inference Request is Complete

A request is considered complete when a response containing the
`TRITONSERVER_RESPONSE_COMPLETE_FINAL` flag is received. For decoupled models,
there are two ways this can happen. The model/backend calls one of the
following [TRITONBACKEND APIs](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonbackend.h):
1. `TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, ...)`
2. `TRITONBACKEND_ResponseFactorySendFlags(factory, TRITONSERVER_RESPONSE_COMPLETE_FINAL)`

As described in the
[backend repo](https://github.com/triton-inference-server/backend/blob/main/README.md#special-cases)
for decoupled models:

> If the backend should not send any more responses for the request,
> `TRITONBACKEND_ResponseFactorySendFlags` can be used to send 
> `TRITONSERVER_RESPONSE_COMPLETE_FINAL` using the ResponseFactory.

In the `TRITONBACKEND_ResponseFactorySendFlags` case, only the `flags` are
communicated back to the frontend to update some internal state, and there 
is no actual Inference Response sent back along with the flags. The default
behavior in this case is not to send anything back to the client, as there
is no response to send.

In some cases, this default behavior proved to be significantly more performant.
For example, take a decoupled model with an `N` request -> `1` response structure.
For each of the first `N-1` requests, the model will send "zero" responses back
by using `TRITONBACKEND_ResponseFactorySendFlags` as described above, and likely
update some internal states in the model. Finally, on the `N`th request, the
model is ready to send a response. 

If the client is written in such a way that it is aware of the model's
expected behavior, it can save resources and avoid network contention by
not needing to communicate the `N-1` "empty" responses back and forth with
the client, and instead the client will just wait until it receives the single
non-empty response expected at the end on request `N`.

However, there are cases where a user may want to write a client that can
generically handle any model, without knowing implementation details about it.
Similarly, there are cases where the number of responses a model will send
is unknown beforehand, so the client may need a programatic way to know when
the final response for a given request has been received. A common case for
this may be where for a language model that has a `1` request -> `N` response
structure. 

To handle this case, Triton exposes a boolean `"triton_final_response"` response 
parameter that communicates to the client that this response is the final response
for the associated request/response ID when communicating with decoupled models. 

> **NOTE**
> This response parameter is only provided for `decoupled` models at this time.
> Since every response will be the final response for non-decoupled models,
> this would be redundant to communicate.


#### TRITONBACKEND_ResponseSend 

When a final response is sent via
`TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, ...)`,
an actual response is sent to the frontend by the backend/model, so this response
parameter will be included in the response back to the client by default.

#### TRITONBACKEND_ResponseFactorySendFlags 

When a final response is sent via
`TRITONBACKEND_ResponseFactorySendFlags(factory, TRITONSERVER_RESPONSE_COMPLETE_FINAL)`,
no response is sent to the frontend by the backend/model, so nothing will be sent
back to the client by default. For the client to receive a response containing
this "final" signal via the `"triton_final_response"` parameter, the client
will have to opt-in through the client library.

To opt-in through the Python client library, the `enable_empty_final_response` arg
should be set when calling `async_stream_infer(..., enable_empty_final_response=True)`.

> **NOTE**
> The `enable_empty_final_response` response parameter is only exposed in
> the `async_stream_infer` method as this time, since this feature is only
> needed for `decoupled` models.

The [decoupled_test.py](https://github.com/triton-inference-server/server/blob/main/qa/L0_decoupled/decoupled_test.py)
demonstrates an example of using this opt-in arg and programatically identifying
when a final response is received through the `"triton_final_response"`
response parameter.

If using 
[Triton's in-process C API](../customization_guide/inference_protocols.md#in-process-triton-server-api)
instead of the GRPC frontend,
then your application should be handling the logic to identify when the final
response associated with a request has been received, such as by checking for
the `TRITONSERVER_RESPONSE_COMPLETE_FINAL` response flag mentioned above.
