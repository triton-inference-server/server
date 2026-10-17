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

# Self-contained runtime image for L0_simple_go_client.
#
# This image installs the build tooling the test needs on top of a plain
# Triton runtime image: the Go toolchain, protoc and both protoc-gen-go
# plugins. pytest is the one thing it does not install -- the test runs
# against whatever version the base image ships. It deliberately does NOT
# depend on Dockerfile.QA having provided any of this, so the test runs the
# same way in CI and on a workstation. conftest.py likewise provisions its
# own model repository rather than relying on the models/ directory
# Dockerfile.QA stages.
#
# In CI the pipeline overrides BASE_IMAGE with the image it just built. The
# default below is only for running this test standalone.

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

ARG GO_VERSION=1.25.14

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl git protobuf-compiler \
 && rm -rf /var/lib/apt/lists/*

# dpkg's architecture names (amd64, arm64) match the Go release tarballs.
RUN GOARCH="$(dpkg --print-architecture)" \
 && curl -sSL "https://go.dev/dl/go${GO_VERSION}.linux-${GOARCH}.tar.gz" \
    | tar -C /usr/local -xz

ENV PATH=/usr/local/go/bin:/root/go/bin:${PATH}
# The client module pins a patch-level go directive; allow the toolchain to
# upgrade itself rather than failing the build when the pin moves ahead.
ENV GOTOOLCHAIN=auto

# gen_go_stubs.sh invokes protoc with --go_out and --go-grpc_out, so both
# plugins are required. Left unpinned so the generated code stays compatible
# with whatever protobuf runtime the client module resolves at test time.
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@latest \
 && go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# pytest is taken from the base image rather than installed or pinned here, so
# the test runs against the same version the image ships. Fail the build, not
# the test run, if the base image does not provide it.
RUN python3 -m pytest --version

WORKDIR /opt/tritonserver/qa/L0_simple_go_client

# For standalone use. CI mounts the qa-tests volume over /opt/tritonserver/qa,
# so these files are shadowed there and the mounted copy is what runs.
COPY . .

CMD ["python3", "run_tests.py"]
