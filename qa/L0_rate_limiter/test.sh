#!/bin/bash
# Copyright 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
RATE_LIMITER_TEST=rate_limiter_test.py
TEST_RESULT_FILE='test_results.txt'

MODELDIR=${MODELDIR:=`pwd`}
DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends


SERVER_ARGS_EXTRA="--backend-directory=${BACKEND_DIR}"
source ../common/util.sh

RET=0

rm -f *.log
rm -fr ./custom_models && mkdir ./custom_models && \
cp -r ../custom_models/custom_zero_1_float32 ./custom_models/. && \
cp -r ../custom_models/custom_sequence_int32 ./custom_models/. && \
mkdir -p ./custom_models/custom_zero_1_float32/1 && \
cp -r ./custom_models/custom_zero_1_float32 ./custom_models/custom_zero_1_float32_v2


(cd custom_models/custom_zero_1_float32 && \
        sed -i "s/dims:.*\[.*\]/dims: \[ -1 \]/g" config.pbtxt && \
        sed -i "s/max_batch_size:.*/max_batch_size: 4/g" config.pbtxt && \
        echo "instance_group [{"  >> config.pbtxt && \
        echo "kind: KIND_GPU count: 1"  >> config.pbtxt && \
        echo "rate_limiter { resources [{name: \"resource1\" count: 4 }]}"  >> config.pbtxt && \
        echo "}]" >> config.pbtxt && \
        echo "parameters [" >> config.pbtxt && \
        echo "{ key: \"execute_delay_ms\"; value: { string_value: \"100\" }}" >> config.pbtxt && \
        echo "]" >> config.pbtxt)


(cd custom_models/custom_zero_1_float32_v2 && \
        sed -i "s/custom_zero_1_float32/custom_zero_1_float32_v2/g" config.pbtxt && \
        sed -i "s/dims:.*\[.*\]/dims: \[ -1 \]/g" config.pbtxt && \
        sed -i "s/max_batch_size:.*/max_batch_size: 4/g" config.pbtxt && \
        echo "instance_group [{"  >> config.pbtxt && \
        echo "kind: KIND_GPU count: 1"  >> config.pbtxt && \
        echo "rate_limiter { resources [{name: \"resource1\" count: 2 }, {name: \"resource2\" global: True count: 2 }] priority: 2}"  >> config.pbtxt && \
        echo "}]" >> config.pbtxt && \
        echo "parameters [" >> config.pbtxt && \
        echo "{ key: \"execute_delay_ms\"; value: { string_value: \"100\" }}" >> config.pbtxt && \
        echo "]" >> config.pbtxt)

##
## Test cases that fails to load models
##
# Case1: Both resource lesser than required
SERVER_ARGS="--rate-limit=execution_count --rate-limit-resource=resource1:1 --model-repository=$MODELDIR/custom_models"
SERVER_LOG="./inference_server_r1.log"
run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** Unexpected success with resource count 1\n***"
    RET=1

    kill $SERVER_PID
    wait $SERVER_PID
fi
grep "Resource count for \"resource1\" is limited to 1 which will prevent scheduling of one or more model instances, the minimum required count is 4" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected error message while loading the model \"custom_zero_1_float32\"\n***"
    RET=1
fi

# Case2: resources sufficient only for one model
SERVER_ARGS="--rate-limit=execution_count --rate-limit-resource=resource1:3 --rate-limit-resource=resource2:2 --model-repository=$MODELDIR/custom_models"
SERVER_LOG="./inference_server_r3.log"
run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** Unexpected success with resource count 1\n***"
    RET=1

    kill $SERVER_PID
    wait $SERVER_PID
fi
grep "Resource count for \"resource1\" is limited to 3 which will prevent scheduling of one or more model instances, the minimum required count is 4" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected error message while loading the model \"custom_zero_1_float32\"\n***"
    RET=1
fi

# Case3: Resource specified only for specific device id 10 and not for the GPU that loads the model instance.
SERVER_ARGS="--rate-limit=execution_count --rate-limit-resource=resource1:10:10 --rate-limit-resource=resource2:2 --model-repository=$MODELDIR/custom_models"
SERVER_LOG="./inference_server_rdevice.log"
run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** Unexpected success with resource count 1\n***"
    RET=1

    kill $SERVER_PID
    wait $SERVER_PID
fi
grep "Resource count for \"resource1\" is limited to 0 which will prevent scheduling of one or more model instances, the minimum required count is 4" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected error message while loading the model \"custom_zero_1_float32\"\n***"
    RET=1
fi

# Case4: Conflicting resource types in the config
cp -r ./custom_models/custom_zero_1_float32_v2 ./custom_models/custom_zero_1_float32_v3
(cd custom_models/custom_zero_1_float32_v3 && \
        sed -i "s/custom_zero_1_float32_v2/custom_zero_1_float32_v3/g" config.pbtxt && \
        sed -i "s/global: True/global: False/g " config.pbtxt)

SERVER_ARGS="--rate-limit=execution_count --model-repository=$MODELDIR/custom_models"
SERVER_LOG="./inference_server_conflict.log"
run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** Unexpected success with resource count 1\n***"
    RET=1

    kill $SERVER_PID
    wait $SERVER_PID
fi
grep "Resource \"resource2\" is present as both global and device-specific resource in the model configuration." $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected error message for conflicting resource types\n***"
    RET=1
fi
rm -rf ./custom_models/custom_zero_1_float32_v3

##
## Tests with cross-model prioritization with various cases:
##
# CASE1: Explicit limited resource: only allows one model to run at a time
SERVER_ARGS="--rate-limit=execution_count --rate-limit-resource=resource1:4 --rate-limit-resource=resource2:2 --model-repository=$MODELDIR/custom_models"
SERVER_LOG="./inference_server.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python3 $RATE_LIMITER_TEST RateLimiterTest.test_cross_model_prioritization_limited_resource >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
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

kill $SERVER_PID
wait $SERVER_PID

# CASE2: Implicit Limited resource: By default, server will select max resources of one of the
# model as available resource. This means only one model will run at a time.
SERVER_ARGS="--rate-limit=execution_count --model-repository=$MODELDIR/custom_models"
SERVER_LOG="./inference_server.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python3 $RATE_LIMITER_TEST RateLimiterTest.test_cross_model_prioritization_limited_resource >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
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

kill $SERVER_PID
wait $SERVER_PID

# CASE3: Explicit plenty resource: Allows multiple models to run simultaneously
SERVER_ARGS="--rate-limit=execution_count --rate-limit-resource=resource1:6 --rate-limit-resource=resource2:2 --model-repository=$MODELDIR/custom_models"
SERVER_LOG="./inference_server.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

python3 $RATE_LIMITER_TEST RateLimiterTest.test_cross_model_prioritization_plenty_resource >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
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

kill $SERVER_PID
wait $SERVER_PID

##
## Tests with mulitple instances of the same model
##
# Replace the second model with a second instance with same resource requirements and priority.
# TODO: Currently there is no way to check which instance got to run inferences hence we only
# check the resource constraint. Add more extensive tests for multiple instances once required
# information is made available.
rm -rf custom_models/custom_zero_1_float32_v2
(cd custom_models/custom_zero_1_float32 && \
        echo "instance_group [{"  >> config.pbtxt && \
        echo "kind: KIND_GPU count: 1"  >> config.pbtxt && \
        echo "rate_limiter { resources [{name: \"resource1\" count: 2 }, {name: \"resource2\" global: True count: 2 }] priority: 2}"  >> config.pbtxt && \
        echo "}]" >> config.pbtxt)

# CASE1: limited resource: only allows one model instance to run at a time.
SERVER_ARGS="--rate-limit=execution_count --model-repository=$MODELDIR/custom_models"
SERVER_LOG="./inference_server.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
SECONDS=0
python3 $RATE_LIMITER_TEST RateLimiterTest.test_single_model >>$CLIENT_LOG 2>&1
LIMITED_RESOURCE_TEST_DURATION=$SECONDS
echo -e "Limited resource time: ${LIMITED_RESOURCE_TEST_DURATION}s"
if [ $? -ne 0 ]; then
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

kill $SERVER_PID
wait $SERVER_PID

# CASE 2: plenty resource: allows both the instances to run simultaneously
SERVER_ARGS="--rate-limit=execution_count  --rate-limit-resource=resource1:6 --rate-limit-resource=resource2:2  --model-repository=$MODELDIR/custom_models"
SERVER_LOG="./inference_server.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
SECONDS=0
python3 $RATE_LIMITER_TEST RateLimiterTest.test_single_model >>$CLIENT_LOG 2>&1
PLENTY_RESOURCE_TEST_DURATION=$SECONDS
echo -e "Plenty resource time: ${LIMITED_RESOURCE_TEST_DURATION}s"
if [ $? -ne 0 ]; then
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

if [ $PLENTY_RESOURCE_TEST_DURATION -gt $LIMITED_RESOURCE_TEST_DURATION ]; then
   echo -e "Error: Test with limited resources should take more time"
   echo -e "\n***\n*** Test Failed\n***"
   RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# Case 3: resources sufficient only for one model instance. Hence, should fail to load
SERVER_ARGS="--rate-limit=execution_count --rate-limit-resource=resource1:3 --rate-limit-resource=resource2:2 --model-repository=$MODELDIR/custom_models"
SERVER_LOG="./inference_server_r3i.log"
run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** Unexpected success with resource count 1\n***"
    RET=1

    kill $SERVER_PID
    wait $SERVER_PID
fi
grep "Resource count for \"resource1\" is limited to 3 which will prevent scheduling of one or more model instances, the minimum required count is 4" $SERVER_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed. Expected error message while loading the model \"custom_zero_1_float32\"\n***"
    RET=1
fi

##
## Tests with dynamic batching
##
# Despite all the possible bs being preferred triton should always form full batches as
# the second instance would be blocked because of the resource constraints.
(cd custom_models/custom_zero_1_float32 && \
        sed -i "s/.*execute_delay_ms.*/{ key: \"execute_delay_ms\"; value: { string_value: \"1000\" }}/g" config.pbtxt && \
        echo "dynamic_batching { preferred_batch_size: [ 1, 2, 3, 4 ]" >> config.pbtxt && \
        echo " max_queue_delay_microseconds: 5000000 }"  >> config.pbtxt)
export TRITONSERVER_DELAY_SCHEDULER=8
SERVER_ARGS="--rate-limit=execution_count --model-repository=$MODELDIR/custom_models"
SERVER_LOG="./inference_server.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python3 $RATE_LIMITER_TEST RateLimiterTest.test_single_model_dynamic_batching >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
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

kill $SERVER_PID
wait $SERVER_PID

unset TRITONSERVER_DELAY_SCHEDULER

##
## Tests with sequence batching
##
# Send one sequence and check for correct accumulator result. The result should be returned immediately.
# This test checks whether all the requests are directed to the same instance despite there being other
# instances with higher priority.
FIRST_INSTANCE_RESOURCE="rate_limiter { resources [{name: \"resource1\" count: 4 }]}"
(cd custom_models/custom_sequence_int32/ && \
        sed -i "s/max_sequence_idle_microseconds:.*/max_sequence_idle_microseconds: 1000000/" config.pbtxt && \
        sed -i "s/^max_batch_size:.*/max_batch_size: 1/" config.pbtxt && \
        sed -i "s/kind: KIND_GPU/kind: KIND_CPU\\ncount: 1 \n${FIRST_INSTANCE_RESOURCE}/" config.pbtxt && \
        sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 1 \n${FIRST_INSTANCE_RESOURCE}/" config.pbtxt &&\
        echo "instance_group [{"  >> config.pbtxt && \
        echo "kind: KIND_CPU count: 1"  >> config.pbtxt && \
        echo "rate_limiter { resources [{name: \"resource1\" count: 2 }, {name: \"resource2\" global: True count: 2 }] priority: 2}"  >> config.pbtxt && \
        echo "}]" >> config.pbtxt && \
        echo "instance_group [{"  >> config.pbtxt && \
        echo "kind: KIND_CPU count: 2"  >> config.pbtxt && \
        echo "rate_limiter { resources [{name: \"resource1\" count: 2 }, {name: \"resource2\" global: True count: 2 }] priority: 3}"  >> config.pbtxt && \
        echo "}]" >> config.pbtxt)
SERVER_ARGS="--rate-limit=execution_count --model-repository=$MODELDIR/custom_models"
SERVER_LOG="./inference_server.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python3 $RATE_LIMITER_TEST RateLimiterTest.test_single_model_sequence_batching >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
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

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
