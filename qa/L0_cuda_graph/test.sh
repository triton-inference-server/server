#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

export CUDA_VISIBLE_DEVICES=0

CLIENT_LOG="./client.log"
TRT_CUDA_GRAPH_TEST=trt_cuda_graph_test.py
TEST_RESULT_FILE='test_results.txt'
DATADIR="./models"

rm -rf ${DATADIR}
mkdir -p ${DATADIR}

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--log-verbose=1 --model-repository=$DATADIR --strict-model-config=true"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

rm -f *.log*

RET=0

# TrtCudaGraphTest.test_fixed_shape
rm -rf ${DATADIR} && mkdir -p ${DATADIR}
cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/plan_float32_float32_float32 ${DATADIR}/
# Make sure only one version is present
rm -rf ${DATADIR}/plan_float32_float32_float32/3

CLIENT_LOG="./fixed_shape.client.log"
SERVER_LOG="./fixed_shape.inference_server.log"
echo "optimization { cuda { graphs: true } }" >> ${DATADIR}/plan_float32_float32_float32/config.pbtxt

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $TRT_CUDA_GRAPH_TEST TrtCudaGraphTest.test_fixed_shape>>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    cat $CLIENT_LOG
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

set +e
if [ `grep -c "Context with profile default \[0\] is being executed for " $SERVER_LOG` != "1" ]; then
    echo -e "\n***\n*** Failed. Expected only one execution without CUDA graph\n***"
    RET=1
fi

if [ `grep -c "captured CUDA graph for" $SERVER_LOG` != "6" ]; then
    echo -e "\n***\n*** Failed. Expected 6 CUDA graphs are captured\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# TrtCudaGraphTest.test_dynamic_shape
# plan_float32_float32_float32 models with dynamic shapes has 6 profiles
# min, opt, max, idx
# [1, 1], [1, 16], [8, 33], 0 (*)
# [1, 1], [2, 16], [7, 32], 1
# [1, 1], [3, 16], [6, 32], 2
# [1, 1], [4, 16], [5, 32], 3
# [5, 1], [6, 16], [8, 32], 4 (*)
# [6, 1], [6, 16], [8, 32], 5 (*)
# [1, 1], [1, 16], [8, 32], 6
rm -rf ${DATADIR} && mkdir -p ${DATADIR}
cp -r /data/inferenceserver/${REPO_VERSION}/qa_variable_model_repository/plan_float32_float32_float32 ${DATADIR}/

SERVER_ARGS="--log-verbose=1 --model-repository=$DATADIR --strict-model-config=true"
CLIENT_LOG="./dynamic_shape.client.log"
SERVER_LOG="./dynamic_shape.inference_server.log"
sed -i "s/profile:.*/profile: [\"0\"]/" ${DATADIR}/plan_float32_float32_float32/config.pbtxt
echo "optimization { cuda { graphs: true } }" >> ${DATADIR}/plan_float32_float32_float32/config.pbtxt

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $TRT_CUDA_GRAPH_TEST TrtCudaGraphTest.test_dynamic_shape>>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    cat $CLIENT_LOG
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

set +e
if [ `grep -c "Context with profile 0 \[0\] is being executed for " $SERVER_LOG` != "2" ]; then
    echo -e "\n***\n*** Failed. Expected 2 execution without CUDA graph\n***"
    RET=1
fi

if [ `grep -c "captured CUDA graph for" $SERVER_LOG` != "6" ]; then
    echo -e "\n***\n*** Failed. Expected 6 CUDA graphs are captured\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# TrtCudaGraphTest.test_range_fixed_shape
rm -rf ${DATADIR} && mkdir -p ${DATADIR}
cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/plan_float32_float32_float32 ${DATADIR}/
# Make sure only one version is present
rm -rf ${DATADIR}/plan_float32_float32_float32/3

SERVER_ARGS="--log-verbose=1 --model-repository=$DATADIR"
CLIENT_LOG="./range_fixed_shape.client.log"
SERVER_LOG="./range_fixed_shape.inference_server.log"
echo "optimization { \
    cuda { \
        graphs: true \
        graph_spec [ { \
            batch_size: 4 \
            graph_lower_bound { \
                batch_size: 2 \
            } \
} ] } }" >> ${DATADIR}/plan_float32_float32_float32/config.pbtxt

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $TRT_CUDA_GRAPH_TEST TrtCudaGraphTest.test_range_fixed_shape>>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    cat $CLIENT_LOG
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

set +e
if [ `grep -c "Context with profile default \[0\] is being executed for " $SERVER_LOG` != "2" ]; then
    echo -e "\n***\n*** Failed. Expected only 2 execution without CUDA graph\n***"
    RET=1
fi

if [ `grep -c "captured CUDA graph for" $SERVER_LOG` != "1" ]; then
    echo -e "\n***\n*** Failed. Expected 1 CUDA graphs are captured\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# TrtCudaGraphTest.test_range_dynamic_shape
# plan_float32_float32_float32 models with dynamic shapes has 6 profiles
# min, opt, max, idx
# [1, 1], [1, 16], [8, 33], 0 (*)
# [1, 1], [2, 16], [7, 32], 1
# [1, 1], [3, 16], [6, 32], 2
# [1, 1], [4, 16], [5, 32], 3
# [5, 1], [6, 16], [8, 32], 4 (*)
# [6, 1], [6, 16], [8, 32], 5 (*)
# [1, 1], [1, 16], [8, 32], 6
rm -rf ${DATADIR} && mkdir -p ${DATADIR}
cp -r /data/inferenceserver/${REPO_VERSION}/qa_variable_model_repository/plan_float32_float32_float32 ${DATADIR}/

CLIENT_LOG="./range_dynamic_shape.client.log"
SERVER_LOG="./range_dynamic_shape.inference_server.log"
sed -i "s/profile:.*/profile: [\"0\"]/" ${DATADIR}/plan_float32_float32_float32/config.pbtxt
echo "optimization { \
    cuda { \
        graphs: true \
        graph_spec [ { \
            batch_size: 4 \
            input { key: \"INPUT0\" value: {dim : [16]} } \
            input { key: \"INPUT1\" value: {dim : [16]} } \
            graph_lower_bound { \
                batch_size: 2 \
                input { key: \"INPUT0\" value: {dim : [8]} } \
                input { key: \"INPUT1\" value: {dim : [8]} } \
            } \
} ] } }" >> ${DATADIR}/plan_float32_float32_float32/config.pbtxt

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $TRT_CUDA_GRAPH_TEST TrtCudaGraphTest.test_range_dynamic_shape>>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    cat $CLIENT_LOG
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

set +e
if [ `grep -c "Context with profile 0 \[0\] is being executed for " $SERVER_LOG` != "4" ]; then
    echo -e "\n***\n*** Failed. Expected 4 execution without CUDA graph\n***"
    RET=1
fi

if [ `grep -c "captured CUDA graph for" $SERVER_LOG` != "1" ]; then
    echo -e "\n***\n*** Failed. Expected 1 CUDA graphs are captured\n***"
    RET=1
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
