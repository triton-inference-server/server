#!/bin/bash
# Copyright (c) 2019-2024, NVIDIA CORPORATION. All rights reserved.
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

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi
if [ ! -z "$TEST_REPO_ARCH" ]; then
    REPO_VERSION=${REPO_VERSION}_${TEST_REPO_ARCH}
fi

TEST_RESULT_FILE='test_results.txt'
export CUDA_VISIBLE_DEVICES=0

CLIENT_LOG="./client.log"
PLUGIN_TEST=trt_plugin_test.py

# On windows the paths invoked by the script (running in WSL) must use
# /mnt/c when needed but the paths on the tritonserver command-line
# must be C:/ style.
if [[ -v WSL_DISTRO_NAME ]] || [[ -v MSYSTEM ]]; then
    DATADIR=${DATADIR:="/mnt/c/data/inferenceserver/${REPO_VERSION}"}
    MODELDIR=${MODELDIR:=C:/models}
    CUSTOMPLUGIN=${CUSTOMPLUGIN:=$MODELDIR/HardmaxPlugin.dll}
    BACKEND_DIR=${BACKEND_DIR:=C:/tritonserver/backends}
    SERVER=${SERVER:=/mnt/c/tritonserver/bin/tritonserver.exe}
    TEST_WINDOWS=1
else
    DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
    MODELDIR=${MODELDIR:=`pwd`/models}
    CUSTOMPLUGIN=${CUSTOMPLUGIN:=$MODELDIR/libcustomHardmaxPlugin.so}
    TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
    BACKEND_DIR=${TRITON_DIR}/backends
    SERVER=${TRITON_DIR}/bin/tritonserver
fi

source ../common/util.sh

RET=0
rm -f ./*.log

SERVER_ARGS_BASE="--model-repository=${MODELDIR} --backend-directory=${BACKEND_DIR} --log-verbose=1"
SERVER_TIMEOUT=20

LOG_IDX=0

## Custom Plugin Tests

## Create model folder with custom plugin models
rm -fr models && mkdir -p models
find $DATADIR/qa_trt_plugin_model_repository/ -maxdepth 1 -iname '*Hardmax*' -exec cp -r {} models \;

LOG_IDX=$((LOG_IDX+1))

## Baseline Failure Test
## Plugin library not loaded
SERVER_ARGS=$SERVER_ARGS_BASE
SERVER_LOG="./inference_server_$LOG_IDX.log"

run_server
if [ "$SERVER_PID" != "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Test Failed\n"
    echo -e "Unexpected successful server start $SERVER\n***"
    kill_server
    exit 1
fi

LOG_IDX=$((LOG_IDX+1))

## Backend Config, Plugin Test
SERVER_ARGS="${SERVER_ARGS_BASE} --backend-config=tensorrt,plugins=${CUSTOMPLUGIN}"
SERVER_LOG="./inference_server_$LOG_IDX.log"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f $CLIENT_LOG
set +e
python3 $PLUGIN_TEST PluginModelTest.test_raw_hard_max >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE 1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

kill_server

LOG_IDX=$((LOG_IDX+1))

## LD_PRELOAD, Plugin Test
## LD_PRELOAD is only on Linux

SERVER_LD_PRELOAD=$CUSTOMPLUGIN
SERVER_ARGS=$SERVER_ARGS_BASE
SERVER_LOG="./inference_server_$LOG_IDX.log"

# Skip test for Windows
if  [ $TEST_WINDOWS -eq 0 ]; then
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    rm -f $CLIENT_LOG
    set +e
    python3 $PLUGIN_TEST PluginModelTest.test_raw_hard_max >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        check_test_results $TEST_RESULT_FILE 1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi
    set -e

    kill_server
fi

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
