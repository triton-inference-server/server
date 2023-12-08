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

MINIMAL_LOG="./minimal.log"
RECOMMENDED_LOG="./recommended.log"

SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

RET=0

# Client build requires recent version of CMake (FetchContent required)
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

#
# Minimal backend
#
(cd backend/examples/backends/minimal &&
 mkdir build &&
 cd build &&
 cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
       -DTRITON_BACKEND_REPO_TAG=${TRITON_BACKEND_REPO_TAG} \
       -DTRITON_CORE_REPO_TAG=${TRITON_CORE_REPO_TAG} \
       -DTRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG} \
       .. &&
 make -j4 install)

rm -fr /opt/tritonserver/backends/minimal
cp -r backend/examples/backends/minimal/build/install/backends/minimal /opt/tritonserver/backends/.

SERVER_LOG="./minimal_server.log"
SERVER_ARGS="--model-repository=`pwd`/backend/examples/model_repos/minimal_models"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

backend/examples/clients/minimal_client >> ${MINIMAL_LOG} 2>&1
if [ $? -ne 0 ]; then
    cat $MINIMAL_LOG
    RET=1
fi

grep "OUT0 = \[1 2 3 4\]" $MINIMAL_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify minimal nonbatching example. \n***"
    cat $MINIMAL_LOG
    RET=1
fi

grep "OUT0 = \[\[10 11 12 13\]\]" $MINIMAL_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify minimal batching example. \n***"
    cat $MINIMAL_LOG
    RET=1
fi

grep "OUT0 = \[\[20 21 22 23\]\]" $MINIMAL_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify minimal batching example. \n***"
    cat $MINIMAL_LOG
    RET=1
fi

grep "model batching: requests in batch 2" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify minimal server log. \n***"
    cat $SERVER_LOG
    cat $MINIMAL_LOG
    RET=1
fi

grep "batched IN0 value: \[ 10, 11, 12, 13, 20, 21, 22, 23 \]" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify minimal server log. \n***"
    cat $SERVER_LOG
    cat $MINIMAL_LOG
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

rm -fr /opt/tritonserver/backends/minimal

#
# Recommended backend
#
(cd backend/examples/backends/recommended &&
 mkdir build &&
 cd build &&
 cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
       -DTRITON_BACKEND_REPO_TAG=${TRITON_BACKEND_REPO_TAG} \
       -DTRITON_CORE_REPO_TAG=${TRITON_CORE_REPO_TAG} \
       -DTRITON_COMMON_REPO_TAG=${TRITON_COMMON_REPO_TAG} \
       .. &&
 make -j4 install)

rm -fr /opt/tritonserver/backends/recommended
cp -r backend/examples/backends/recommended/build/install/backends/recommended /opt/tritonserver/backends/.

SERVER_LOG="./recommended_server.log"
SERVER_ARGS="--model-repository=`pwd`/backend/examples/model_repos/recommended_models"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

backend/examples/clients/recommended_client >> ${RECOMMENDED_LOG} 2>&1
if [ $? -ne 0 ]; then
    cat $RECOMMENDED_LOG
    RET=1
fi

grep -z "OUTPUT = \[\[\[1.  1.1 1.2 1.3\].*\[2.  2.1 2.2 2.3\].*\[3.  3.1 3.2 3.3\].*\[4.  4.1 4.2 4.3\]\]\]" $RECOMMENDED_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify recommended example. \n***"
    cat $RECOMMENDED_LOG
    RET=1
fi

grep -z "OUTPUT = \[\[\[10.  10.1 10.2 10.3\].*\[20.  20.1 20.2 20.3\].*\[30.  30.1 30.2 30.3\].*\[40.  40.1 40.2 40.3\]\]\]" $RECOMMENDED_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify recommended example. \n***"
    cat $RECOMMENDED_LOG
    RET=1
fi

grep "model batching: requests in batch 2" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify recommended server log. \n***"
    cat $SERVER_LOG
    cat $RECOMMENDED_LOG
    RET=1
fi

FOUND_MATCH=0
grep "batched INPUT value: \[ 1.000000, 1.100000, 1.200000, 1.300000, 2.000000, 2.100000, 2.200000, 2.300000, 3.000000, 3.100000, 3.200000, 3.300000, 4.000000, 4.100000, 4.200000, 4.300000, 10.000000, 10.100000, 10.200000, 10.300000, 20.000000, 20.100000, 20.200001, 20.299999, 30.000000, 30.100000, 30.200001, 30.299999, 40.000000, 40.099998, 40.200001, 40.299999 \]" $SERVER_LOG
if [ $? -ne 0 ]; then
    FOUND_MATCH=1
fi
grep "batched INPUT value: \[ 10.000000, 10.100000, 10.200000, 10.300000, 20.000000, 20.100000, 20.200001, 20.299999, 30.000000, 30.100000, 30.200001, 30.299999, 40.000000, 40.099998, 40.200001, 40.299999, 1.000000, 1.100000, 1.200000, 1.300000, 2.000000, 2.100000, 2.200000, 2.300000, 3.000000, 3.100000, 3.200000, 3.300000, 4.000000, 4.100000, 4.200000, 4.300000 \]" $SERVER_LOG
if [ $? -ne 0 ]; then
    FOUND_MATCH=1
fi
if [ $FOUND_MATCH -eq 0 ]; then
    echo -e "\n***\n*** Failed to verify recommended server log. \n***"
    cat $SERVER_LOG
    cat $RECOMMENDED_LOG
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

rm -fr /opt/tritonserver/backends/recommended

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
