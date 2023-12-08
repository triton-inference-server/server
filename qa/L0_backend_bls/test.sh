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

export CUDA_VISIBLE_DEVICES=0

TRITON_BACKEND_REPO_TAG=${TRITON_BACKEND_REPO_TAG:="main"}
TRITON_CORE_REPO_TAG=${TRITON_CORE_REPO_TAG:="main"}
TRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG:="main"}

SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

RET=0

# Backend build requires recent version of CMake (FetchContent required)
# Using CMAKE installation instruction from:: https://apt.kitware.com/
apt update -q=2 \
    && apt install -y gpg wget \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - |  tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && . /etc/os-release \
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $UBUNTU_CODENAME main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
    && apt-get update -q=2 \
    && apt-get install -y --no-install-recommends cmake=3.27.7* cmake-data=3.27.7* \
            rapidjson-dev
cmake --version

rm -fr *.log ./backend

git clone --single-branch --depth=1 -b $TRITON_BACKEND_REPO_TAG \
    https://github.com/triton-inference-server/backend.git

(cd backend/examples/backends/bls &&
 mkdir build &&
 cd build &&
 cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
       -DTRITON_BACKEND_REPO_TAG=${TRITON_BACKEND_REPO_TAG} \
       -DTRITON_CORE_REPO_TAG=${TRITON_CORE_REPO_TAG} \
       -DTRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG} \
       .. &&
 make -j4 install)

rm -fr /opt/tritonserver/backends/bls
cp -r backend/examples/backends/bls/build/install/backends/bls /opt/tritonserver/backends/.

SERVER_ARGS="--model-repository=`pwd`/backend/examples/model_repos/bls_models --log-verbose=1"
SERVER_LOG="./inference_server.log"
CLIENT_LOG="./client.log"

mkdir `pwd`/backend/examples/model_repos/bls_models/bls_fp32/1/

# Run the server with all the required models.
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
backend/examples/clients/bls_client >> $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo "Failed: Client test had a non-zero return code."
    RET=1
fi

grep "PASS" $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** bls_test.py FAILED. \n***"
    cat $CLIENT_LOG
    cat $SERVER_LOG
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
