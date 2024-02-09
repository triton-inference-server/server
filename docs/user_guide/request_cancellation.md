<!--
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Request Cancellation

Starting from r23.10, Triton supports handling request cancellation received
from the gRPC client or a C API user. Long running inference requests such
as for auto generative large language models may run for an indeterminate
amount of time or indeterminate number of steps. Additionally clients may
enqueue a large number of requests as part of a sequence or request stream
and later determine the results are no longer needed. Continuing to process
requests whose results are no longer required can significantly impact server
resources.

## Issuing Request Cancellation

### In-Process C API

[In-Process Triton Server C API](../customization_guide/inference_protocols.md#in-process-triton-server-api) has been enhanced with `TRITONSERVER_InferenceRequestCancel`
and `TRITONSERVER_InferenceRequestIsCancelled` to issue cancellation and query
whether cancellation has been issued on an inflight request respectively. Read more
about the APIs in [tritonserver.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h).


### gRPC Endpoint

In addition, [gRPC endpoint](../customization_guide/inference_protocols.md#httprest-and-grpc-protocols) can
now detect cancellation from the client and attempt to terminate request.
At present, only gRPC python client supports issuing request cancellation
to the server endpoint. See [request-cancellation](https://github.com/triton-inference-server/client#request-cancellation)
for more details on how to issue requests from the client-side.
See gRPC guide on RPC [cancellation](https://grpc.io/docs/guides/cancellation/) for
finer details.

## Handling in Triton Core

Triton core checks for requests that have been cancelled at some critical points
when using [dynamic](./model_configuration.md#dynamic-batcher) or
[sequence](./model_configuration.md#sequence-batcher) batching. The checking is
also performed between each
[ensemble](./model_configuration.md#ensemble-scheduler) steps and terminates
further processing if the request is cancelled.

On detecting a cancelled request, Triton core responds with CANCELLED status. If a request
is cancelled when using [sequence_batching](./model_configuration.md#sequence-batcher),
then all the pending requests in the same sequence will also be cancelled. The sequence
is represented by the requests that has identical sequence id.

**Note**: Currently, Triton core does not detect cancellation status of a request once
it is forwarded to [rate limiter](./rate_limiter.md). Improving the request cancellation
detection and handling within Triton core is work in progress.

## Handling in Backend

Upon receiving request cancellation, Triton does its best to terminate request
at various points. However, once a request has been given to the backend
for execution, it is up to the individual backends to detect and handle
request termination.
Currently, the following backends support early termination:
- [TensorRT-LLM backend](https://github.com/triton-inference-server/tensorrtllm_backend)
- [vLLM backend](https://github.com/triton-inference-server/vllm_backend)
- [python backend](https://github.com/triton-inference-server/python_backend)

Python backend is a special case where we expose the APIs to detect cancellation
status of the request but it is up to the `model.py` developer to detect whether
the request is cancelled and terminate further execution.

**For the backend developer**: The backend APIs have also been enhanced to let the
backend detect whether the request received from Triton core has been cancelled.
See `TRITONBACKEND_RequestIsCancelled` and `TRITONBACKEND_ResponseFactoryIsCancelled`
in [tritonbackend.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonbackend.h)
for more details. The backend upon detecting request cancellation can stop processing
it any further.
The Python models running behind Python backend can also query the cancellation status
of request and response_sender. See [this](https://github.com/triton-inference-server/python_backend#request-cancellation-handling)
section in python backend documentation for more details.

