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
# In CI the pipeline overrides BASE_IMAGE with the devel image it just
# built, so the tritonserver binary, Go toolchain and protoc all come from
# the pipeline rather than being pinned here. The default below is only for
# running this test standalone on a workstation.

ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:26.08-py3

FROM ${BASE_IMAGE}

# These must be re-declared after FROM: an ARG given before FROM is in scope
# only for the FROM line itself, so referencing it in ENV below would bake in
# an empty string -- which is worse than leaving the variable unset, because
# conftest.py would then have to treat "" as a value.
ARG TRITON_SERVER_BRANCH_NAME=main
ARG TRITON_COMMON_REPO_TAG=main
ARG TRITON_CLIENT_REPO_TAG=main

# Baked-in defaults for standalone runs. CI passes the same names with -e,
# which takes precedence over image ENV.
ENV TRITON_SERVER_BRANCH_NAME=${TRITON_SERVER_BRANCH_NAME}
ENV TRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG}
ENV TRITON_CLIENT_REPO_TAG=${TRITON_CLIENT_REPO_TAG}

# pytest is not part of the devel image contract. Pin it so a broken
# upstream release cannot change what this job runs.
RUN python3 -m pip install --no-cache-dir "pytest==8.3.5"

WORKDIR /opt/tritonserver/qa/L0_simple_go_client

# For standalone use only. CI mounts the qa-tests volume over
# /opt/tritonserver/qa, so these files are shadowed there and the mounted
# copy is what runs. Note the model repository is not in the build context;
# a standalone run still needs ./models supplied.
COPY . .

CMD ["python3", "run_tests.py"]
