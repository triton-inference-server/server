# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Runtime image for L0_simple_go_client.
#
# Adds only what the pytest conversion needs on top of the image the
# pipeline already built. The test files themselves are deliberately NOT
# copied in: CI mounts the qa-tests volume over /opt/tritonserver/qa, so a
# COPY to that path would be masked at run time and would only mislead.
#
# BASE_IMAGE must be the same devel image the job would otherwise have run
# against, so the tritonserver binary, Go toolchain and protoc all come
# from the pipeline build rather than being pinned here.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# pytest is not part of the devel image contract. Install it explicitly and
# pin it so a broken upstream release cannot change what this job runs.
RUN python3 -m pip install --no-cache-dir "pytest==8.3.5"

WORKDIR /opt/tritonserver/qa/L0_simple_go_client

CMD ["python3", "run_tests.py"]
