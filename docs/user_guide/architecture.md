<!--
# Copyright 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Triton Architecture

The following figure shows the Triton Inference Server high-level
architecture. The [model repository](model_repository.md) is a
file-system based repository of the models that Triton will make
available for inferencing. Inference requests arrive at the server via
either [HTTP/REST or GRPC](../customization_guide/inference_protocols.md) or by the [C
API](../customization_guide/inference_protocols.md) and are then routed to the appropriate per-model
scheduler. Triton implements [multiple scheduling and batching
algorithms](#models-and-schedulers) that can be configured on a
model-by-model basis. Each model's scheduler optionally performs
batching of inference requests and then passes the requests to the
[backend](https://github.com/triton-inference-server/backend/blob/main/README.md)
corresponding to the model type. The backend performs inferencing
using the inputs provided in the batched requests to produce the
requested outputs. The outputs are then returned.

Triton supports a [backend C
API](https://github.com/triton-inference-server/backend/blob/main/README.md#triton-backend-api)
that allows Triton to be extended with new functionality such as
custom pre- and post-processing operations or even a new deep-learning
framework.

The models being served by Triton can be queried and controlled by a
dedicated [model management API](model_management.md) that is
available by HTTP/REST or GRPC protocol, or by the C API.

Readiness and liveness health endpoints and utilization, throughput
and latency metrics ease the integration of Triton into deployment
framework such as Kubernetes.

![Triton Architecture Diagram](images/arch.jpg)