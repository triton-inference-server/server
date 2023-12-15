#!/bin/bash
# Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

REPORTER=../common/reporter.py
CLIENT_LOG="./simple_perf_client.log"
SIMPLE_PERF_CLIENT=simple_perf_client.py

TF_VERSION=${TF_VERSION:=2}

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=`pwd`/custom_models --backend-config=tensorflow,version=${TF_VERSION}"
source ../common/util.sh

# Select the single GPU that will be available to the inference
# server.
export CUDA_VISIBLE_DEVICES=0
PROTOCOLS="grpc http"

rm -f *.log *.csv *.tjson *.json

RET=0

MODEL_NAME="custom_zero_1_int32"

for PROTOCOL in $PROTOCOLS; do
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi


    NAME=${MODEL_NAME}_${PROTOCOL}
    EXTRA_ARGS="" && [[ "${PROTOCOL}" == "grpc" ]] && EXTRA_ARGS="-i grpc -u localhost:8001"
    python $SIMPLE_PERF_CLIENT -m $MODEL_NAME --shape 100000 --csv ${NAME}.csv ${EXTRA_ARGS}>> ${NAME}.log 2>&1
    if (( $? != 0 )); then
        RET=1
    fi

    echo -e "[{\"s_benchmark_kind\":\"benchmark_perf\"," >> ${NAME}.tjson
    echo -e "\"s_benchmark_name\":\"python_client\"," >> ${NAME}.tjson
    echo -e "\"s_server\":\"triton\"," >> ${NAME}.tjson
    echo -e "\"s_protocol\":\"${PROTOCOL}\"," >> ${NAME}.tjson
    echo -e "\"s_framework\":\"custom\"," >> ${NAME}.tjson
    echo -e "\"s_model\":\"${MODEL_NAME}\"," >> ${NAME}.tjson
    echo -e "\"l_concurrency\":1," >> ${NAME}.tjson
    echo -e "\"l_batch_size\":1," >> ${NAME}.tjson
    echo -e "\"l_instance_count\":1}]" >> ${NAME}.tjson


    if [ -f $REPORTER ]; then
        set +e

        URL_FLAG=
        if [ ! -z ${BENCHMARK_REPORTER_URL} ]; then
            URL_FLAG="-u ${BENCHMARK_REPORTER_URL}"
        fi

        python $REPORTER -v -o ${NAME}.json --csv ${NAME}.csv ${URL_FLAG} ${NAME}.tjson
        if (( $? != 0 )); then
            RET=1
        fi

        set -e
    fi

    kill $SERVER_PID
    wait $SERVER_PID
done

if (( $RET == 0 )); then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
