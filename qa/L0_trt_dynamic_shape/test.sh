#!/bin/bash
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

export CUDA_VISIBLE_DEVICES=0

CLIENT_LOG="./client.log"
PERF_CLIENT=../clients/perf_client
TRT_OP_TEST=trt_dynamic_shape_test.py

DATADIR="./models"

rm -rf ${DATADIR}
mkdir -p ${DATADIR}
cp -r /data/inferenceserver/${REPO_VERSION}/qa_variable_model_repository/plan_float32_float32_float32-4-32 ${DATADIR}/

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=$DATADIR"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

rm -f *.log*

RET=0

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# Shape beyond the limits of optimization profile
set +e
$PERF_CLIENT -v -i grpc -u localhost:8001 -m plan_float32_float32_float32-4-32 --shape INPUT0:33 --shape INPUT1:33 -t 1 -p2000 -b 1 > ${CLIENT_LOG}_max 2>&1
if [ $? -eq 0 ]; then
    cat ${CLIENT_LOG}_max
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

EXPECTED_MESSAGE="model expected the shape of dimension 1 to be between 4 and 32 but received"
if [ $(cat ${CLIENT_LOG}_max | grep "${EXPECTED_MESSAGE} 33" | wc -l) -eq 0 ]; then
    cat ${CLIENT_LOG}_max
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

$PERF_CLIENT -v -i grpc -u localhost:8001 -m plan_float32_float32_float32-4-32 --shape INPUT0:3 --shape INPUT1:3 -t 1 -p2000 -b 1 > ${CLIENT_LOG}_min 2>&1
if [ $? -eq 0 ]; then
    cat ${CLIENT_LOG}_min
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ $(cat ${CLIENT_LOG}_min | grep "${EXPECTED_MESSAGE} 3" | wc -l) -eq 0 ]; then
    cat ${CLIENT_LOG}_min
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

# Tests with multiple optimization profiles

# plan_float32_float32_float32 models with dynamic shapes has 6 profiles
# min, opt, max, idx
# [1, 1], [1, 16], [8, 33], 0 (*)
# [1, 1], [2, 16], [7, 32], 1
# [1, 1], [3, 16], [6, 32], 2
# [1, 1], [4, 16], [5, 32], 3
# [5, 1], [6, 16], [8, 32], 4 (*)
# [6, 1], [6, 16], [8, 32], 5 (*)
# [1, 1], [1, 16], [8, 32], 6
rm -rf ${DATADIR} && rm -f config.pbtxt && mkdir -p ${DATADIR}
cp -r /data/inferenceserver/${REPO_VERSION}/qa_variable_model_repository/plan_float32_float32_float32 ${DATADIR}/

# Keep a copy of original model config for different modifications
cp -r /data/inferenceserver/${REPO_VERSION}/qa_variable_model_repository/plan_float32_float32_float32/config.pbtxt .

# TrtDynamicShapeTest.test_load_specific_optimization_profile
CLIENT_LOG="./test_load_specific_optimization_profile.client.log"
SERVER_LOG="./test_load_specific_optimization_profile.inference_server.log"
cp config.pbtxt ${DATADIR}/plan_float32_float32_float32/config.pbtxt && \
sed -i "s/profile:.*/profile: [\"5\"]/" ${DATADIR}/plan_float32_float32_float32/config.pbtxt

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $TRT_OP_TEST TrtDynamicShapeTest.test_load_specific_optimization_profile >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    cat $CLIENT_LOG
    RET=1
else
    check_test_results $CLIENT_LOG 1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# TrtDynamicShapeTest.test_load_default_optimization_profile
CLIENT_LOG="./test_load_default_optimization_profile.client.log"
SERVER_LOG="./test_load_default_optimization_profile.inference_server.log"
cp config.pbtxt ${DATADIR}/plan_float32_float32_float32/config.pbtxt && \
sed -i "s/profile:.*//" ${DATADIR}/plan_float32_float32_float32/config.pbtxt

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $TRT_OP_TEST TrtDynamicShapeTest.test_load_default_optimization_profile >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    cat $CLIENT_LOG
    RET=1
else
    check_test_results $CLIENT_LOG 1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# TrtDynamicShapeTest.test_select_optimization_profile
# Note that this test needs to check server log for which OP is used
#
# finding OP that best fit the input shape:
#     load OP 0, 1, 2, 3, send [4 16] and 3 should be used
SERVER_ARGS="--model-repository=$DATADIR --log-verbose=1"
CLIENT_LOG="./test_select_optimization_profile.client.best.log"
SERVER_LOG="./test_select_optimization_profile.inference_server.best.log"
(cp config.pbtxt ${DATADIR}/plan_float32_float32_float32/config.pbtxt && \
        sed -i "s/max_batch_size:.*/max_batch_size: 5/" ${DATADIR}/plan_float32_float32_float32/config.pbtxt && \
        sed -i "s/profile:.*/profile: [\"0\", \"1\", \"2\", \"3\"]/" ${DATADIR}/plan_float32_float32_float32/config.pbtxt)

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $TRT_OP_TEST TrtDynamicShapeTest.test_select_optimization_profile >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $CLIENT_LOG 1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
fi
set -e

set +e
grep "Context with profile 3 \[3\] is being executed for " test_select_optimization_profile.inference_server.best.log
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected profile 3 is used\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# finding OP that best fit the input shape while the input shape is allowed:
#     load OP 0, 5, send [4 16] and 0 should be used
#     (OP 5 is the best in terms of OPT dims, but it requires min dims [6, 1])
CLIENT_LOG="./test_select_optimization_profile.client.allow.log"
SERVER_LOG="./test_select_optimization_profile.inference_server.allow.log"
cp config.pbtxt ${DATADIR}/plan_float32_float32_float32/config.pbtxt && \
sed -i "s/profile:.*/profile: [\"0\", \"5\"]/" ${DATADIR}/plan_float32_float32_float32/config.pbtxt

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $TRT_OP_TEST TrtDynamicShapeTest.test_select_optimization_profile >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    cat $CLIENT_LOG
    RET=1
else
    check_test_results $CLIENT_LOG 1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
fi
set -e

set +e
grep "Context with profile 0 \[0\] is being executed for " test_select_optimization_profile.inference_server.allow.log
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected profile 0 is used\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# TrtDynamicShapeTest.test_load_wrong_optimization_profile
SERVER_ARGS="--model-repository=$DATADIR --exit-on-error=false --strict-readiness=false"
CLIENT_LOG="./test_load_wrong_optimization_profile.client.log"
SERVER_LOG="./test_load_wrong_optimization_profile.inference_server.log"
cp config.pbtxt ${DATADIR}/plan_float32_float32_float32/config.pbtxt && \
sed -i "s/profile:.*/profile: [\"7\"]/" ${DATADIR}/plan_float32_float32_float32/config.pbtxt

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $TRT_OP_TEST TrtDynamicShapeTest.test_load_wrong_optimization_profile >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    cat $CLIENT_LOG
    RET=1
else
    check_test_results $CLIENT_LOG 1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID


if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test Failed\n***"
fi

exit $RET
