#!/bin/bash
# Copyright 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
RET=0
set -e
if [ ${PYTHON_ENV_VERSION} = "12" ]; then
    echo No need to set up anything for default python3.${PYTHON_ENV_VERSION}
    exit $RET
fi

source common.sh
source ../common/util.sh

TRITON_REPO_ORGANIZATION=${TRITON_REPO_ORGANIZATION:="http://github.com/triton-inference-server"}
BASE_SERVER_ARGS="--model-repository=${MODELDIR}/models --log-verbose=1 --disable-auto-complete-config"
PYTHON_BACKEND_BRANCH=$PYTHON_BACKEND_REPO_TAG
SERVER_ARGS=$BASE_SERVER_ARGS
SERVER_LOG="./inference_server.log"
export PYTHON_ENV_VERSION=${PYTHON_ENV_VERSION:="12"}
RET=0

rm -fr ./models
rm -rf *.tar.gz
install_build_deps

echo "python environment 3.${PYTHON_ENV_VERSION}"
# copy the stub out to /opt/tritonserver/backends/python/triton_python_backend_stub
cp python_backend/builddir/triton_python_backend_stub /opt/tritonserver/backends/python/triton_python_backend_stub
# Set up environment and stub for each test
apt-get update -qq && apt-get install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update && apt-get -y install \
                            "python3.${PYTHON_ENV_VERSION}-dev" \
                            "python3.${PYTHON_ENV_VERSION}-distutils" \
                            libboost-dev
rm -f /usr/bin/python3 && \
ln -s "/usr/bin/python3.${PYTHON_ENV_VERSION}" /usr/bin/python3
pip3 install --upgrade requests numpy virtualenv protobuf
find /opt/tritonserver/qa/pkgs/ -maxdepth 1 -type f -name \
    "tritonclient-*-py3-none-any.whl" | xargs printf -- '%s[all]' | \
    xargs pip3 install --upgrade

# Build triton-shm-monitor for the test
cd python_backend && rm -rf install build && mkdir build && cd build && \
    export CMAKE_POLICY_VERSION_MINIMUM=3.5 && \
    cmake -DCMAKE_INSTALL_PREFIX:PATH=$PWD/install \
        -DTRITON_REPO_ORGANIZATION:STRING=${TRITON_REPO_ORGANIZATION} \
        -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG} \
        -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG} \
        -DTRITON_BACKEND_REPO_TAG:STRING=${TRITON_BACKEND_REPO_TAG} .. && \
    make -j16 triton-shm-monitor install
cp $PWD/install/backends/python/triton_shm_monitor.cpython-* /opt/tritonserver/qa/common/.
set +e
exit $RET
