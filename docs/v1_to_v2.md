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

# Version 1 to Version 2 Migration

Version 2 of Triton does not generally maintain backwards
compatibility with version 1.  Specifically, you should take the
following items into account when transitioning from version 1 to
version 2.

* The Triton executables and libraries are in /opt/tritonserver. The
  Triton executable is /opt/tritonserver/bin/tritonserver.

* Some *tritonserver* command-line arguments are removed, changed or
  have different default behavior in version 2.

  * --api-version, --http-health-port, --grpc-infer-thread-count,
    --grpc-stream-infer-thread-count,--allow-poll-model-repository, --allow-model-control
    and --tf-add-vgpu are removed.

  * The default for --model-control-mode is changed to *none*.

  * --tf-allow-soft-placement and --tf-gpu-memory-fraction are renamed
     to --backend-config="tensorflow,allow-soft-placement=\<true,false\>"
     and --backend-config="tensorflow,gpu-memory-fraction=\<float\>".

* The HTTP/REST and GRPC protocols, while conceptually similar to
  version 1, are completely changed in version 2. See [inference
  protocols](inference_protocols.md) for more information.

* Python and C++ client libraries are re-implemented to match the new
  HTTP/REST and GRPC protocols. The Python client no longer depends on
  a C++ shared library and so should be usable on any platform that
  supports Python. See [client
  libraries](https://github.com/triton-inference-server/client) for
  more information.

* Building Triton has changed significantly in version 2. See
  [build](build.md) for more information.

* In the Docker containers the environment variables indicating the
  Triton version have changed to have a TRITON prefix, for example,
  TRITON_SERVER_VERSION.
