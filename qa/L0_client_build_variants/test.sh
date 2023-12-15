#!/bin/bash
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

# Install required dependencies for client build
apt-get update && \
apt-get install -y --no-install-recommends \
        rapidjson-dev

# Client build requires recent version of CMake (FetchContent required)
# Using CMAKE installation instruction from:: https://apt.kitware.com/
apt update -q=2 \
    && apt install -y gpg wget \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - |  tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && . /etc/os-release \
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $UBUNTU_CODENAME main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
    && apt-get update -q=2 \
    && apt-get install -y --no-install-recommends cmake=3.27.7* cmake-data=3.27.7*
cmake --version


set +e

mkdir -p /workspace/build

#
# Build without GPU support
#
(cd /workspace/build && \
        rm -fr cc-clients java-clients python-clients && \
        cmake -DCMAKE_INSTALL_PREFIX=/workspace/install \
              -DTRITON_ENABLE_CC_HTTP=ON \
              -DTRITON_ENABLE_CC_GRPC=ON \
              -DTRITON_ENABLE_PYTHON_HTTP=ON \
              -DTRITON_ENABLE_PYTHON_GRPC=ON \
              -DTRITON_ENABLE_JAVA_HTTP=ON \
              -DTRITON_ENABLE_PERF_ANALYZER=ON \
              -DTRITON_ENABLE_PERF_ANALYZER_C_API=ON \
              -DTRITON_ENABLE_PERF_ANALYZER_TFS=OFF \
              -DTRITON_ENABLE_PERF_ANALYZER_TS=OFF \
              -DTRITON_ENABLE_EXAMPLES=ON \
              -DTRITON_ENABLE_TESTS=ON \
              -DTRITON_ENABLE_GPU=OFF \
              -DTRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG} \
              -DTRITON_CORE_REPO_TAG=${TRITON_CORE_REPO_TAG} \
              -DTRITON_THIRD_PARTY_REPO_TAG=${TRITON_THIRD_PARTY_REPO_TAG} \
              /workspace/client && \
        make -j16 cc-clients java-clients python-clients)
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** No-GPU Passed\n***"
else
    echo -e "\n***\n*** No-GPU FAILED\n***"
    exit 1
fi

#
# Build without HTTP
# Skip this test for java-clients because we can only build
# java-clients with http protocol
#
(cd /workspace/build && \
        rm -fr cc-clients python-clients && \
        cmake -DCMAKE_INSTALL_PREFIX=/workspace/install \
              -DTRITON_ENABLE_CC_HTTP=OFF \
              -DTRITON_ENABLE_CC_GRPC=ON \
              -DTRITON_ENABLE_PYTHON_HTTP=OFF \
              -DTRITON_ENABLE_PYTHON_GRPC=ON \
              -DTRITON_ENABLE_PERF_ANALYZER=ON \
              -DTRITON_ENABLE_PERF_ANALYZER_C_API=ON \
              -DTRITON_ENABLE_PERF_ANALYZER_TFS=OFF \
              -DTRITON_ENABLE_PERF_ANALYZER_TS=OFF \
              -DTRITON_ENABLE_EXAMPLES=ON \
              -DTRITON_ENABLE_TESTS=ON \
              -DTRITON_ENABLE_GPU=ON \
              -DTRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG} \
              -DTRITON_CORE_REPO_TAG=${TRITON_CORE_REPO_TAG} \
              -DTRITON_THIRD_PARTY_REPO_TAG=${TRITON_THIRD_PARTY_REPO_TAG} \
              /workspace/client && \
        make -j16 cc-clients python-clients)
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** No-HTTP Passed\n***"
else
    echo -e "\n***\n*** No-HTTP FAILED\n***"
    exit 1
fi

#
# Build without GRPC
# Skip this test for java-clients because grpc protocol is not supported
#
(cd /workspace/build && \
        rm -fr cc-clients python-clients && \
        cmake -DCMAKE_INSTALL_PREFIX=/workspace/install \
              -DTRITON_ENABLE_CC_HTTP=ON \
              -DTRITON_ENABLE_CC_GRPC=OFF \
              -DTRITON_ENABLE_PYTHON_HTTP=ON \
              -DTRITON_ENABLE_PYTHON_GRPC=OFF \
              -DTRITON_ENABLE_PERF_ANALYZER=ON \
              -DTRITON_ENABLE_PERF_ANALYZER_C_API=ON \
              -DTRITON_ENABLE_PERF_ANALYZER_TFS=OFF \
              -DTRITON_ENABLE_PERF_ANALYZER_TS=OFF \
              -DTRITON_ENABLE_EXAMPLES=ON \
              -DTRITON_ENABLE_TESTS=ON \
              -DTRITON_ENABLE_GPU=ON \
              -DTRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG} \
              -DTRITON_CORE_REPO_TAG=${TRITON_CORE_REPO_TAG} \
              -DTRITON_THIRD_PARTY_REPO_TAG=${TRITON_THIRD_PARTY_REPO_TAG} \
              /workspace/client && \
        make -j16 cc-clients python-clients)
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** No-GRPC Passed\n***"
else
    echo -e "\n***\n*** No-GRPC FAILED\n***"
    exit 1
fi

#
# Build without Perf Analyzer
#
(cd /workspace/build && \
        rm -fr cc-clients python-clients && \
        cmake -DCMAKE_INSTALL_PREFIX=/workspace/install \
              -DTRITON_ENABLE_CC_HTTP=ON \
              -DTRITON_ENABLE_CC_GRPC=ON \
              -DTRITON_ENABLE_PYTHON_HTTP=ON \
              -DTRITON_ENABLE_PYTHON_GRPC=ON \
              -DTRITON_ENABLE_PERF_ANALYZER=OFF \
              -DTRITON_ENABLE_PERF_ANALYZER_C_API=OFF \
              -DTRITON_ENABLE_PERF_ANALYZER_TFS=OFF \
              -DTRITON_ENABLE_PERF_ANALYZER_TS=OFF \
              -DTRITON_ENABLE_EXAMPLES=ON \
              -DTRITON_ENABLE_TESTS=ON \
              -DTRITON_ENABLE_GPU=ON \
              -DTRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG} \
              -DTRITON_CORE_REPO_TAG=${TRITON_CORE_REPO_TAG} \
              -DTRITON_THIRD_PARTY_REPO_TAG=${TRITON_THIRD_PARTY_REPO_TAG} \
              /workspace/client && \
        make -j16 cc-clients python-clients)
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** No-Perf-Analyzer Passed\n***"
else
    echo -e "\n***\n*** No-Perf-Analyzer FAILED\n***"
    exit 1
fi

#
# Build without C API in Perf Analyzer
#
(cd /workspace/build && \
        rm -fr cc-clients python-clients && \
        cmake -DCMAKE_INSTALL_PREFIX=/workspace/install \
              -DTRITON_ENABLE_CC_HTTP=ON \
              -DTRITON_ENABLE_CC_GRPC=ON \
              -DTRITON_ENABLE_PYTHON_HTTP=ON \
              -DTRITON_ENABLE_PYTHON_GRPC=ON \
              -DTRITON_ENABLE_PERF_ANALYZER=ON \
              -DTRITON_ENABLE_PERF_ANALYZER_C_API=OFF \
              -DTRITON_ENABLE_PERF_ANALYZER_TFS=ON \
              -DTRITON_ENABLE_PERF_ANALYZER_TS=ON \
              -DTRITON_ENABLE_EXAMPLES=ON \
              -DTRITON_ENABLE_TESTS=ON \
              -DTRITON_ENABLE_GPU=ON \
              -DTRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG} \
              -DTRITON_CORE_REPO_TAG=${TRITON_CORE_REPO_TAG} \
              -DTRITON_THIRD_PARTY_REPO_TAG=${TRITON_THIRD_PARTY_REPO_TAG} \
              /workspace/client && \
        make -j16 cc-clients python-clients)
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** No-CAPI Passed\n***"
else
    echo -e "\n***\n*** No-CAPI FAILED\n***"
    exit 1
fi

#
# Build without TensorFlow Serving in Perf Analyzer
#
(cd /workspace/build && \
        rm -fr cc-clients python-clients && \
        cmake -DCMAKE_INSTALL_PREFIX=/workspace/install \
              -DTRITON_ENABLE_CC_HTTP=ON \
              -DTRITON_ENABLE_CC_GRPC=ON \
              -DTRITON_ENABLE_PYTHON_HTTP=ON \
              -DTRITON_ENABLE_PYTHON_GRPC=ON \
              -DTRITON_ENABLE_PERF_ANALYZER=ON \
              -DTRITON_ENABLE_PERF_ANALYZER_C_API=ON \
              -DTRITON_ENABLE_PERF_ANALYZER_TFS=OFF \
              -DTRITON_ENABLE_PERF_ANALYZER_TS=ON \
              -DTRITON_ENABLE_EXAMPLES=ON \
              -DTRITON_ENABLE_TESTS=ON \
              -DTRITON_ENABLE_GPU=ON \
              -DTRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG} \
              -DTRITON_CORE_REPO_TAG=${TRITON_CORE_REPO_TAG} \
              -DTRITON_THIRD_PARTY_REPO_TAG=${TRITON_THIRD_PARTY_REPO_TAG} \
              /workspace/client && \
        make -j16 cc-clients python-clients)
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** No-TF-Serving Passed\n***"
else
    echo -e "\n***\n*** No-TF-Serving FAILED\n***"
    exit 1
fi

#
# Build without TorchServe in Perf Analyzer
#
(cd /workspace/build && \
        rm -fr cc-clients python-clients && \
        cmake -DCMAKE_INSTALL_PREFIX=/workspace/install \
              -DTRITON_ENABLE_CC_HTTP=ON \
              -DTRITON_ENABLE_CC_GRPC=ON \
              -DTRITON_ENABLE_PYTHON_HTTP=ON \
              -DTRITON_ENABLE_PYTHON_GRPC=ON \
              -DTRITON_ENABLE_PERF_ANALYZER=ON \
              -DTRITON_ENABLE_PERF_ANALYZER_C_API=ON \
              -DTRITON_ENABLE_PERF_ANALYZER_TFS=ON \
              -DTRITON_ENABLE_PERF_ANALYZER_TS=OFF \
              -DTRITON_ENABLE_EXAMPLES=ON \
              -DTRITON_ENABLE_TESTS=ON \
              -DTRITON_ENABLE_GPU=ON \
              -DTRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG} \
              -DTRITON_CORE_REPO_TAG=${TRITON_CORE_REPO_TAG} \
              -DTRITON_THIRD_PARTY_REPO_TAG=${TRITON_THIRD_PARTY_REPO_TAG} \
              /workspace/client && \
        make -j16 cc-clients python-clients)
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** No-TorchServe Passed\n***"
else
    echo -e "\n***\n*** No-TorchServe FAILED\n***"
    exit 1
fi

set -e

echo -e "\n***\n*** Test Passed\n***"
