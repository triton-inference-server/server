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

[Inference Protocols and APIs](inference_protocols.md) describes various ways
a client can communicate and run inference on the server. For decoupled models,
Triton's HTTP endpoint cannot be used for running inference as it supports
exactly one response per request. Even standard ModelInfer RPC in the GRPC endpoint
does not support decoupled responses. In order to run inference on a decoupled
model, the client must use the bi-directional streaming RPC. See
[here](https://github.com/triton-inference-server/common/blob/main/protobuf/grpc_service.proto)
for more details. The [decoupled_test.py](../qa/L0_decoupled/decoupled_test.py) demonstrates
how the gRPC streaming can be used to infer decoupled models.

If using [Triton's in-process C API](inference_protocols.md#in-process-triton-server-api),
your application should be cognizant that the callback function you registered with 
`TRITONSERVER_InferenceRequestSetResponseCallback` can be invoked any number of times,
each time with a new response. You can take a look at [grpc_server.cc](../src/grpc_server.cc)
