#!/bin/bash
# Copyright 2019-2026, NVIDIA CORPORATION. All rights reserved.
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

SIMPLE_TEST_PY=./ensemble_test.py

CLIENT_LOG="./client.log"

TEST_MODEL_DIR="`pwd`/models"
BACKPRESSURE_TEST_MODEL_DIR="`pwd`/backpressure_test_models"
TEST_RESULT_FILE='test_results.txt'
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=${TEST_MODEL_DIR}"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

# ensure ensemble models have version sub-directory
mkdir -p ${TEST_MODEL_DIR}/ensemble_add_sub_int32_int32_int32/1
mkdir -p ${TEST_MODEL_DIR}/ensemble_partial_add_sub/1

rm -f $CLIENT_LOG $SERVER_LOG

# Run ensemble model with all outputs requested
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

RET=0

set +e
python $SIMPLE_TEST_PY EnsembleTest.test_ensemble_add_sub >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
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

# Run ensemble model with sequence flags and verify response sequence
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $SIMPLE_TEST_PY EnsembleTest.test_ensemble_sequence_flags >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
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

# Run ensemble model with only one output requested
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $SIMPLE_TEST_PY EnsembleTest.test_ensemble_add_sub_one_output >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
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

# Run partial ensemble model with all outputs requested
SERVER_ARGS="$SERVER_ARGS --log-verbose=1"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
SERVER_LOG=$SERVER_LOG python $SIMPLE_TEST_PY EnsembleTest.test_ensemble_partial_add_sub >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
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

######## Test max_queue_size dynamic batching parameter in ensemble steps ########
## Ensemble model: step1-decoupled_producer -> step2-slow_consumer
MAX_QUEUE_SIZE_TEST_MODEL_DIR="`pwd`/max_queue_size_test_models"
rm -rf ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}

# Enable max_queue_size in the first step (decoupled_producer)
mkdir -p ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/ensemble_step1_enabled_max_queue_size/1 \
    ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/decoupled_producer_enabled_max_queue_size/1 \
    ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/slow_consumer/1
cp ${BACKPRESSURE_TEST_MODEL_DIR}/ensemble_disabled_max_inflight_requests/config.pbtxt \
    ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/ensemble_step1_enabled_max_queue_size/
sed -i 's/"decoupled_producer"/"decoupled_producer_enabled_max_queue_size"/g' \
    ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/ensemble_step1_enabled_max_queue_size/config.pbtxt

cp ../python_models/ground_truth/model.py ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/slow_consumer/1
cp ../python_models/ground_truth/config.pbtxt ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/slow_consumer/
sed -i 's/name: "ground_truth"/name: "slow_consumer"/g' ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/slow_consumer/config.pbtxt
sed -i 's/max_batch_size: 64/max_batch_size: 1/g' ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/slow_consumer/config.pbtxt

cp ${BACKPRESSURE_TEST_MODEL_DIR}/decoupled_producer/1/model.py ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/decoupled_producer_enabled_max_queue_size/1
cp ${BACKPRESSURE_TEST_MODEL_DIR}/decoupled_producer/config.pbtxt ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/decoupled_producer_enabled_max_queue_size/
sed -i 's/name: "decoupled_producer"/name: "decoupled_producer_enabled_max_queue_size"/g' ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/decoupled_producer_enabled_max_queue_size/config.pbtxt
# Add dynamic_batching with max_queue_size to decoupled_producer
cat >> ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/decoupled_producer_enabled_max_queue_size/config.pbtxt << 'EOF'

dynamic_batching {
  preferred_batch_size: [ 1 ]
  default_queue_policy {
    max_queue_size: 4
  }
}
EOF

# Enable max_queue_size in the second step (slow_consumer)
mkdir -p ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/ensemble_step2_enabled_max_queue_size/1 \
    ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/decoupled_producer/1 ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/slow_consumer_enabled_max_queue_size/1
cp ${BACKPRESSURE_TEST_MODEL_DIR}/ensemble_disabled_max_inflight_requests/config.pbtxt \
    ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/ensemble_step2_enabled_max_queue_size/
sed -i 's/"slow_consumer"/"slow_consumer_enabled_max_queue_size"/g' \
    ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/ensemble_step2_enabled_max_queue_size/config.pbtxt

cp ${BACKPRESSURE_TEST_MODEL_DIR}/decoupled_producer/1/model.py ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/decoupled_producer/1
cp ${BACKPRESSURE_TEST_MODEL_DIR}/decoupled_producer/config.pbtxt ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/decoupled_producer/

cp ../python_models/ground_truth/model.py ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/slow_consumer_enabled_max_queue_size/1
cp ../python_models/ground_truth/config.pbtxt ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/slow_consumer_enabled_max_queue_size/
sed -i 's/name: "ground_truth"/name: "slow_consumer_enabled_max_queue_size"/g' \
    ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/slow_consumer_enabled_max_queue_size/config.pbtxt
sed -i 's/max_batch_size: 64/max_batch_size: 1/g' ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/slow_consumer_enabled_max_queue_size/config.pbtxt
# Add dynamic_batching with max_queue_size to slow_consumer
cat >> ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/slow_consumer_enabled_max_queue_size/config.pbtxt << 'EOF'

dynamic_batching {
  preferred_batch_size: [ 1 ]
  default_queue_policy {
    max_queue_size: 4
  }
}
EOF

BACKPRESSURE_TEST_PY=./ensemble_backpressure_test.py
TEST_NAME="EnsembleStepMaxQueueSizeTest"
SERVER_LOG="./ensemble_step_max_queue_size_test_server.log"
CLIENT_LOG="./ensemble_step_max_queue_size_test_client.log"
rm -f $SERVER_LOG $CLIENT_LOG

SERVER_ARGS="--model-repository=${MAX_QUEUE_SIZE_TEST_MODEL_DIR}"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $BACKPRESSURE_TEST_PY $TEST_NAME -v >> $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    RET=1
    cat $CLIENT_LOG
else
    check_test_results $TEST_RESULT_FILE 2
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID


######## Test parallel-step failed enqueue path in ensemble scheduler ########
PARALLEL_FAILED_ENQUEUE_MODEL_DIR="`pwd`/parallel_failed_enqueue_test_models"
rm -rf ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}

mkdir -p ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/ensemble_parallel_step_failed_enqueue/1
mkdir -p ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/decoupled_producer_parallel_queue/1
mkdir -p ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/slow_consumer_queue_limited/1
mkdir -p ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/fast_consumer/1
mkdir -p ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/join_add_sub/1

# Producer emits repeated responses with a larger payload value so the
# queue-limited branch fills first.
cp ${BACKPRESSURE_TEST_MODEL_DIR}/decoupled_producer/1/model.py \
  ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/decoupled_producer_parallel_queue/1
cp ${BACKPRESSURE_TEST_MODEL_DIR}/decoupled_producer/config.pbtxt \
  ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/decoupled_producer_parallel_queue/
sed -i 's/name: "decoupled_producer"/name: "decoupled_producer_parallel_queue"/g' \
  ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/decoupled_producer_parallel_queue/config.pbtxt
sed -i 's/0.5/2.0/g' \
  ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/decoupled_producer_parallel_queue/1/model.py

# Queue-limited branch used to trigger a failed enqueue.
cp ../python_models/ground_truth/model.py \
  ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/slow_consumer_queue_limited/1
cp ../python_models/ground_truth/config.pbtxt \
  ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/slow_consumer_queue_limited/
sed -i 's/name: "ground_truth"/name: "slow_consumer_queue_limited"/g' \
  ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/slow_consumer_queue_limited/config.pbtxt
sed -i 's/max_batch_size: 64/max_batch_size: 1/g' \
  ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/slow_consumer_queue_limited/config.pbtxt
cat >> ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/slow_consumer_queue_limited/config.pbtxt << 'EOF'

dynamic_batching {
  preferred_batch_size: [ 1 ]
  default_queue_policy {
    max_queue_size: 1
  }
}
EOF

# Parallel branch with the same interface and no added delay.
cp ../python_models/ground_truth/model.py ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/fast_consumer/1
cp ../python_models/ground_truth/config.pbtxt ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/fast_consumer/
sed -i 's/name: "ground_truth"/name: "fast_consumer"/g' \
  ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/fast_consumer/config.pbtxt
sed -i 's/max_batch_size: 64/max_batch_size: 1/g' \
  ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/fast_consumer/config.pbtxt
sed -i 's/time.sleep(delay)/time.sleep(0)/g' \
  ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/fast_consumer/1/model.py

# Join both parallel branches into the ensemble output.
cp ../python_models/join_add_sub/model.py ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/join_add_sub/1
cp ../python_models/join_add_sub/config.pbtxt ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/join_add_sub/

cat > ${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}/ensemble_parallel_step_failed_enqueue/config.pbtxt << 'EOF'
name: "ensemble_parallel_step_failed_enqueue"
platform: "ensemble"
max_batch_size: 0

input [
  {
    name: "IN"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }
]

output [
  {
    name: "OUT"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "decoupled_producer_parallel_queue"
      model_version: -1
      input_map {
        key: "IN"
        value: "IN"
      }
      output_map {
        key: "OUT"
        value: "intermediate"
      }
    },
    {
      model_name: "slow_consumer_queue_limited"
      model_version: -1
      input_map {
        key: "INPUT0"
        value: "intermediate"
      }
      output_map {
        key: "OUTPUT0"
        value: "slow_out"
      }
    },
    {
      model_name: "fast_consumer"
      model_version: -1
      input_map {
        key: "INPUT0"
        value: "intermediate"
      }
      output_map {
        key: "OUTPUT0"
        value: "fast_out"
      }
    },
    {
      model_name: "join_add_sub"
      model_version: -1
      input_map {
        key: "INPUT0"
        value: "slow_out"
      }
      input_map {
        key: "INPUT1"
        value: "fast_out"
      }
      output_map {
        key: "OUTPUT0"
        value: "OUT"
      }
    }
  ]
}
EOF

BACKPRESSURE_TEST_PY=./ensemble_backpressure_test.py
TEST_NAME="EnsembleParallelFailedEnqueueTest.test_parallel_step_failed_enqueue"
SERVER_LOG="./ensemble_parallel_failed_enqueue_test_server.log"
CLIENT_LOG="./ensemble_parallel_failed_enqueue_test_client.log"
rm -f $SERVER_LOG $CLIENT_LOG

SERVER_ARGS="--model-repository=${PARALLEL_FAILED_ENQUEUE_MODEL_DIR}"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
PARALLEL_FAILED_ENQUEUE_LOOPS=${PARALLEL_FAILED_ENQUEUE_LOOPS:-1} \
python $BACKPRESSURE_TEST_PY $TEST_NAME -v >> $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    RET=1
    cat $CLIENT_LOG
else
    check_test_results $TEST_RESULT_FILE 1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi

if ! kill -0 $SERVER_PID > /dev/null 2>&1; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Server exited during parallel failed enqueue test\n***"
    RET=1
else
    wait_for_server_live $SERVER_PID 5
    if [ "$WAIT_RET" != "0" ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** Server did not remain live after parallel failed enqueue test\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID > /dev/null 2>&1 || true
wait $SERVER_PID > /dev/null 2>&1 || true


######## Test backpressure feature - 'max_inflight_requests' config option ########
ENSEMBLE_BACKPRESSURE_TEST_MODEL_DIR="`pwd`/ensemble_backpressure_test_models"
rm -rf ${ENSEMBLE_BACKPRESSURE_TEST_MODEL_DIR}

TEST_NAME="EnsembleBackpressureTest"
SERVER_LOG="./ensemble_backpressure_test_server.log"
CLIENT_LOG="./ensemble_backpressure_test_client.log"
SERVER_ARGS="--model-repository=${ENSEMBLE_BACKPRESSURE_TEST_MODEL_DIR}"
rm -f $SERVER_LOG $CLIENT_LOG

# Step 1 - decoupled_producer (batch size 2)
mkdir -p ${ENSEMBLE_BACKPRESSURE_TEST_MODEL_DIR}/decoupled_producer/1
cp ${BACKPRESSURE_TEST_MODEL_DIR}/decoupled_producer/1/model.py ${ENSEMBLE_BACKPRESSURE_TEST_MODEL_DIR}/decoupled_producer/1/
cp ${BACKPRESSURE_TEST_MODEL_DIR}/decoupled_producer/config.pbtxt ${ENSEMBLE_BACKPRESSURE_TEST_MODEL_DIR}/decoupled_producer/
sed -i 's/max_batch_size: 1/max_batch_size: 2/g' ${ENSEMBLE_BACKPRESSURE_TEST_MODEL_DIR}/decoupled_producer/config.pbtxt

generate_consumer_model() {
    local name=$1
    local delay=$2

    mkdir -p ${ENSEMBLE_BACKPRESSURE_TEST_MODEL_DIR}/${name}/1
    cat > ${ENSEMBLE_BACKPRESSURE_TEST_MODEL_DIR}/${name}/1/model.py << EOF
import time
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            out_tensor = pb_utils.Tensor("OUTPUT0", in_tensor.as_numpy())
            responses.append(pb_utils.InferenceResponse([out_tensor]))
            time.sleep(${delay})
        return responses
EOF
    cat > ${ENSEMBLE_BACKPRESSURE_TEST_MODEL_DIR}/${name}/config.pbtxt << EOF
name: "${name}"
backend: "python"
max_batch_size: 2
input [ { name: "INPUT0", data_type: TYPE_FP32, dims: [ 1 ] } ]
output [ { name: "OUTPUT0", data_type: TYPE_FP32, dims: [ 1 ] } ]
instance_group [ { count: 1, kind: KIND_CPU } ]
dynamic_batching { preferred_batch_size: [ 2 ] }
EOF
}

generate_ensemble_model() {
    local name=$1
    local limit=$2
    local batch_size=2

    local limit_str=""
    if [ "$limit" != "disabled" ]; then
        limit_str="max_inflight_requests: $limit"
    fi

    mkdir -p ${ENSEMBLE_BACKPRESSURE_TEST_MODEL_DIR}/${name}/1
    cat > ${ENSEMBLE_BACKPRESSURE_TEST_MODEL_DIR}/${name}/config.pbtxt << EOF
name: "${name}"
platform: "ensemble"
max_batch_size: ${batch_size}
input [ { name: "IN", data_type: TYPE_INT32, dims: [ 1 ] } ]
output [ { name: "OUT", data_type: TYPE_FP32, dims: [ 1 ] } ]
ensemble_scheduling {
  ${limit_str}
  step [
    {
      model_name: "decoupled_producer"
      model_version: -1
      input_map { key: "IN", value: "IN" }
      output_map { key: "OUT", value: "intermediate_1" }
    },
    {
      model_name: "consumer_high_delay"
      model_version: -1
      input_map { key: "INPUT0", value: "intermediate_1" }
      output_map { key: "OUTPUT0", value: "intermediate_2" }
    },
    {
      model_name: "consumer_low_delay"
      model_version: -1
      input_map { key: "INPUT0", value: "intermediate_2" }
      output_map { key: "OUTPUT0", value: "OUT" }
    }
  ]
}
EOF
}

# Steps 2 and 3 - consumer_high_delay and consumer_low_delay (batch size 2)
generate_consumer_model "consumer_high_delay" "0.5"
generate_consumer_model "consumer_low_delay" "0.1"

# Ensemble models with different max_inflight_requests limits (including disabled)
generate_ensemble_model "ensemble_disabled" "disabled"
generate_ensemble_model "ensemble_limit_1" 1
generate_ensemble_model "ensemble_limit_4" 4

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

# Verify valid config was loaded successfully
if ! grep -q "Ensemble model 'ensemble_limit_1' configured with max_inflight_requests: 1" $SERVER_LOG; then
    echo -e "\n***\n*** FAILED: ensemble_limit_1 did not load\n***"
    RET=1
fi
if ! grep -q "Ensemble model 'ensemble_limit_4' configured with max_inflight_requests: 4" $SERVER_LOG; then
    echo -e "\n***\n*** FAILED: ensemble_limit_4 did not load\n***"
    RET=1
fi

python $BACKPRESSURE_TEST_PY $TEST_NAME -v >> $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    RET=1
    cat $CLIENT_LOG
else
    check_test_results $TEST_RESULT_FILE 4
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID


######## Test invalid values for 'max_inflight_requests' config option ########
INVALID_PARAM_MODEL_DIR="`pwd`/invalid_param_test_models"
SERVER_ARGS="--model-repository=${INVALID_PARAM_MODEL_DIR}"
SERVER_LOG="./invalid_max_inflight_requests_server.log"
rm -rf $SERVER_LOG ${INVALID_PARAM_MODEL_DIR}

mkdir -p ${INVALID_PARAM_MODEL_DIR}/ensemble_invalid_negative_limit/1
mkdir -p ${INVALID_PARAM_MODEL_DIR}/ensemble_invalid_string_limit/1
mkdir -p ${INVALID_PARAM_MODEL_DIR}/ensemble_invalid_large_value_limit/1
# Reuse the decoupled_producer and slow_consumer models built in the previous test section.
cp -r ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/decoupled_producer ${MAX_QUEUE_SIZE_TEST_MODEL_DIR}/slow_consumer ${INVALID_PARAM_MODEL_DIR}/

# max_inflight_requests = -5
cp ${BACKPRESSURE_TEST_MODEL_DIR}/ensemble_disabled_max_inflight_requests/config.pbtxt ${INVALID_PARAM_MODEL_DIR}/ensemble_invalid_negative_limit/
sed -i 's/ensemble_scheduling {/ensemble_scheduling {\n  max_inflight_requests: -5/g' \
  ${INVALID_PARAM_MODEL_DIR}/ensemble_invalid_negative_limit/config.pbtxt

# max_inflight_requests = "invalid_value"
cp ${BACKPRESSURE_TEST_MODEL_DIR}/ensemble_disabled_max_inflight_requests/config.pbtxt ${INVALID_PARAM_MODEL_DIR}/ensemble_invalid_string_limit/
sed -i 's/ensemble_scheduling {/ensemble_scheduling {\n  max_inflight_requests: "invalid_value"/g' \
  ${INVALID_PARAM_MODEL_DIR}/ensemble_invalid_string_limit/config.pbtxt

# max_inflight_requests = 12345678901
cp ${BACKPRESSURE_TEST_MODEL_DIR}/ensemble_disabled_max_inflight_requests/config.pbtxt ${INVALID_PARAM_MODEL_DIR}/ensemble_invalid_large_value_limit/
sed -i 's/ensemble_scheduling {/ensemble_scheduling {\n  max_inflight_requests: 12345678901/g' \
  ${INVALID_PARAM_MODEL_DIR}/ensemble_invalid_large_value_limit/config.pbtxt


run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** FAILED: unexpected success starting $SERVER\n***"
    kill $SERVER_PID
    wait $SERVER_PID
    cat $SERVER_LOG
    RET=1
fi

set +e
# Verify negative value caused model load failure
if ! grep -q "Expected integer, got: -" $SERVER_LOG; then
    echo -e "\n***\n*** FAILED: Negative value should fail model load\n***"
    RET=1
fi

# Verify invalid string caused model load failure
if ! grep -q 'Expected integer, got: "invalid_value"' $SERVER_LOG; then
    echo -e "\n***\n*** FAILED: Invalid string should fail model load\n***"
    RET=1
fi

# Verify very large value caused model load failure
if ! grep -q "Integer out of range (12345678901)" $SERVER_LOG; then
    echo -e "\n***\n*** FAILED: Large value should fail model load\n***"
    RET=1
fi
set -e


if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
