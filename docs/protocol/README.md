<!--
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

# HTTP/REST and GRPC Protocol

This directory contains documents related to the HTTP/REST and GRPC
protocols used by Triton. Triton uses the [KServe community standard
inference
protocols](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2)
plus several extensions that are defined in the following documents:

- [Binary tensor data extension](./extension_binary_data.md)
- [Classification extension](./extension_classification.md)
- [Model configuration extension](./extension_model_configuration.md)
- [Model repository extension](./extension_model_repository.md)
- [Schedule policy extension](./extension_schedule_policy.md)
- [Sequence extension](./extension_sequence.md)
- [Shared-memory extension](./extension_shared_memory.md)
- [Statistics extension](./extension_statistics.md)
- [Trace extension](./extension_trace.md)

For the GRPC protocol the [protobuf
specification](https://github.com/triton-inference-server/common/blob/main/protobuf/grpc_service.proto)
is also available.
