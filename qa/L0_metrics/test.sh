#!/bin/bash
# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

MODELDIR=`pwd`/models
DATADIR=/data/inferenceserver/${REPO_VERSION}/qa_model_repository
TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BASE_SERVER_ARGS="--model-repository=${MODELDIR}"
SERVER_ARGS="${BASE_SERVER_ARGS}"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

CLIENT_LOG="client.log"
TEST_RESULT_FILE="test_results.txt"
function check_unit_test() {
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        EXPECTED_NUM_TESTS="${1:-1}"
        check_test_results ${TEST_RESULT_FILE} ${EXPECTED_NUM_TESTS}
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi
}

function run_and_check_server() {
    run_server
    if [ "$SERVER_PID" == "0" ]; then
      echo -e "\n***\n*** Failed to start $SERVER\n***"
      cat $SERVER_LOG
      exit 1
    fi
}

rm -f $SERVER_LOG
RET=0

if [ `ps | grep -c "tritonserver"` != "0" ]; then
    echo -e "Tritonserver already running"
    echo -e `ps | grep tritonserver`
    exit 1
fi

### UNIT TESTS

TEST_LOG="./metrics_api_test.log"
UNIT_TEST=./metrics_api_test

rm -fr *.log

set +e
export CUDA_VISIBLE_DEVICES=0
LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH $UNIT_TEST >>$TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $TEST_LOG
    echo -e "\n***\n*** Metrics API Unit Test Failed\n***"
    RET=1
fi
set -e

### GPU Metrics

# Prepare a libtorch float32 model with basic config
rm -rf $MODELDIR
model=libtorch_float32_float32_float32
mkdir -p $MODELDIR/${model}/1 && \
  cp -r $DATADIR/${model}/1/* $MODELDIR/${model}/1/. && \
  cp $DATADIR/${model}/config.pbtxt $MODELDIR/${model}/. && \
  (cd $MODELDIR/${model} && \
  sed -i "s/label_filename:.*//" config.pbtxt && \
  echo "instance_group [{ kind: KIND_GPU }]" >> config.pbtxt)

set +e
export CUDA_VISIBLE_DEVICES=0,1,2
run_and_check_server

num_gpus=`curl -s localhost:8002/metrics | grep "nv_gpu_utilization{" | wc -l`
if [ $num_gpus -ne 3 ]; then
  echo "Found $num_gpus GPU(s) instead of 3 GPUs being monitored."
  echo -e "\n***\n*** GPU metric test failed. \n***"
  RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

export CUDA_VISIBLE_DEVICES=0
run_and_check_server

num_gpus=`curl -s localhost:8002/metrics | grep "nv_gpu_utilization{" | wc -l`
if [ $num_gpus -ne 1 ]; then
  echo "Found $num_gpus GPU(s) instead of 1 GPU being monitored."
  echo -e "\n***\n*** GPU metric test failed. \n***"
  RET=1
fi
kill $SERVER_PID
wait $SERVER_PID


# Test metrics interval by querying host and checking energy
METRICS_INTERVAL_MS=500
# Below time interval is larger than actual metrics interval in case
# the update is not ready for unexpected reason
WAIT_INTERVAL_SECS=0.6

SERVER_ARGS="$BASE_SERVER_ARGS --metrics-interval-ms=${METRICS_INTERVAL_MS}"
run_and_check_server

num_iterations=10

# Add "warm up" iteration because in some cases the GPU metrics collection
# doesn't start immediately
prev_energy=`curl -s localhost:8002/metrics | awk '/nv_energy_consumption{/ {print $2}'`
for (( i = 0; i < $num_iterations; ++i )); do
  sleep $WAIT_INTERVAL_SECS
  current_energy=`curl -s localhost:8002/metrics | awk '/nv_energy_consumption{/ {print $2}'`
  if [ $current_energy != $prev_energy ]; then
    echo -e "\n***\n*** Detected changing metrics, warmup completed.\n***"
    break
  fi
  prev_energy=$current_energy
done

prev_energy=`curl -s localhost:8002/metrics | awk '/nv_energy_consumption{/ {print $2}'`
for (( i = 0; i < $num_iterations; ++i )); do
  sleep $WAIT_INTERVAL_SECS
  current_energy=`curl -s localhost:8002/metrics | awk '/nv_energy_consumption{/ {print $2}'`
  if [ $current_energy == $prev_energy ]; then
    cat $SERVER_LOG
    echo "Metrics were not updated in interval of ${METRICS_INTERVAL_MS} milliseconds"
    echo -e "\n***\n*** Metric Interval test failed. \n***"
    RET=1
    break
  fi
  prev_energy=$current_energy
done

### CPU / RAM Metrics

# The underlying values for these metrics do not always update frequently,
# so give ample WAIT time to make sure they change and are being updated.
CPU_METRICS="nv_cpu_utilization nv_cpu_memory_used_bytes"
WAIT_INTERVAL_SECS=2.0
for metric in ${CPU_METRICS}; do
    echo -e "\n=== Checking Metric: ${metric} ===\n"
    prev_value=`curl -s localhost:8002/metrics | grep ${metric} | grep -v "HELP\|TYPE" | awk '{print $2}'`

    num_not_updated=0
    num_not_updated_threshold=3
    for (( i = 0; i < $num_iterations; ++i )); do
      sleep $WAIT_INTERVAL_SECS
      current_value=`curl -s localhost:8002/metrics | grep ${metric} | grep -v "HELP\|TYPE" | awk '{print $2}'`
      if [ $current_value == $prev_value ]; then
        num_not_updated=$((num_not_updated+1))
      fi
      prev_value=$current_value
    done

    # Give CPU metrics some tolerance to not update, up to a threshold
    # DLIS-4304: An alternative may be to run some busy work on CPU in the
    #            background rather than allowing a tolerance threshold
    if [[ ${num_not_updated} -gt ${num_not_updated_threshold} ]]; then
        cat $SERVER_LOG
        echo "Metrics were not updated ${num_not_updated}/${num_iterations} times for interval of ${METRICS_INTERVAL_MS} milliseconds for metric: ${metric}"
        echo -e "\n***\n*** Metric Interval test failed. \n***"
        RET=1
        break
    fi
done

# Verify reported total memory is non-zero
total_memory=`curl -s localhost:8002/metrics | grep "nv_cpu_memory_total_bytes" | grep -v "HELP\|TYPE" | awk '{print $2}'`
test -z "${total_memory}" && total_memory=0
if [ ${total_memory} -eq 0 ]; then
  echo "Found nv_cpu_memory_total_bytes had a value of zero, this should not happen."
  echo -e "\n***\n*** CPU total memory test failed. \n***"
  RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

### Metric Config CLI and different Metric Types ###
MODELDIR="${PWD}/unit_test_models"
mkdir -p "${MODELDIR}/identity_cache_on/1"
mkdir -p "${MODELDIR}/identity_cache_off/1"
BASE_SERVER_ARGS="--model-repository=${MODELDIR} --model-control-mode=explicit"
PYTHON_TEST="metrics_config_test.py"

# Check default settings: Counters should be enabled, summaries should be disabled
SERVER_ARGS="${BASE_SERVER_ARGS} --load-model=identity_cache_off"
run_and_check_server
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_counters_exist 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_summaries_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_cache_counters_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_cache_summaries_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
kill $SERVER_PID
wait $SERVER_PID

# Enable summaries, counters still enabled by default
SERVER_ARGS="${BASE_SERVER_ARGS} --load-model=identity_cache_off --metrics-config summary_latencies=true"
run_and_check_server
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_counters_exist 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_summaries_exist 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_cache_counters_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_cache_summaries_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
kill $SERVER_PID
wait $SERVER_PID

# Enable summaries, disable counters
SERVER_ARGS="${BASE_SERVER_ARGS} --load-model=identity_cache_off --metrics-config summary_latencies=true --metrics-config counter_latencies=false"
run_and_check_server
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_counters_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_summaries_exist 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_cache_counters_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_cache_summaries_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
kill $SERVER_PID
wait $SERVER_PID

# Enable summaries and counters, check cache metrics
CACHE_ARGS="--cache-config local,size=1048576"
SERVER_ARGS="${BASE_SERVER_ARGS} ${CACHE_ARGS} --load-model=identity_cache_on --metrics-config summary_latencies=true --metrics-config counter_latencies=true"
run_and_check_server
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_counters_exist 2>&1 | tee ${CLIENT_LOG}
check_unit_test
# DLIS-4762: Asserts that request summary is not published when cache is
# enabled for a model, until this if fixed.
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_summaries_exist_with_cache 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_cache_counters_exist 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_cache_summaries_exist 2>&1 | tee ${CLIENT_LOG}
check_unit_test
kill $SERVER_PID
wait $SERVER_PID

# Check setting custom summary quantiles
export SUMMARY_QUANTILES="0.1:0.0.1,0.7:0.01,0.75:0.01"
SERVER_ARGS="${BASE_SERVER_ARGS} --load-model=identity_cache_off --metrics-config summary_latencies=true --metrics-config summary_quantiles=${SUMMARY_QUANTILES}"
run_and_check_server
python3 ${PYTHON_TEST} MetricsConfigTest.test_summaries_custom_quantiles 2>&1 | tee ${CLIENT_LOG}
check_unit_test
kill $SERVER_PID
wait $SERVER_PID

### Pending Request Count (Queue Size) Metric Behavioral Tests ###
MODELDIR="${PWD}/queue_size_models"
SERVER_ARGS="--model-repository=${MODELDIR} --log-verbose=1"
PYTHON_TEST="metrics_queue_size_test.py"
rm -rf "${MODELDIR}"
mkdir -p "${MODELDIR}"

# Re-use an identity model that sleeps during execution for N seconds for the
# batch of requests. Then we can confirm queue size behaviors for various
# scheduling/batching strategies.
BASE_MODEL="identity_delay"
# Don't use special debug env var for this, just set sufficient parameters for
# each scheduler to let them fill batches when possible.
unset TRITONSERVER_DELAY_SCHEDULER
export MAX_BATCH_SIZE=4
# Delay up to 100ms to form batches up to MAX_BATCH_SIZE
export MAX_QUEUE_DELAY_US=100000

# Create a model per scheduler type
DEFAULT_MODEL="${MODELDIR}/default"
cp -r "${BASE_MODEL}" "${DEFAULT_MODEL}"
mkdir -p "${DEFAULT_MODEL}/1"
sed -i "s/^max_batch_size.*/max_batch_size: ${MAX_BATCH_SIZE}/" "${DEFAULT_MODEL}/config.pbtxt"

DYNAMIC_MODEL="${MODELDIR}/dynamic"
cp -r "${DEFAULT_MODEL}" "${DYNAMIC_MODEL}"
echo -e "\ndynamic_batching { max_queue_delay_microseconds: ${MAX_QUEUE_DELAY_US} }\n" >> "${DYNAMIC_MODEL}/config.pbtxt"

MAX_QUEUE_SIZE_MODEL="${MODELDIR}/max_queue_size"
cp -r "${DEFAULT_MODEL}" "${MAX_QUEUE_SIZE_MODEL}"
echo -e "\ndynamic_batching { max_queue_delay_microseconds: ${MAX_QUEUE_DELAY_US} default_queue_policy { max_queue_size: 4 } }\n" >> "${MAX_QUEUE_SIZE_MODEL}/config.pbtxt"

SEQUENCE_DIRECT_MODEL="${MODELDIR}/sequence_direct"
cp -r "${DEFAULT_MODEL}" "${SEQUENCE_DIRECT_MODEL}"
echo -e "\nsequence_batching { direct { max_queue_delay_microseconds: ${MAX_QUEUE_DELAY_US}, minimum_slot_utilization: 1.0 } }\n" >> "${SEQUENCE_DIRECT_MODEL}/config.pbtxt"

SEQUENCE_OLDEST_MODEL="${MODELDIR}/sequence_oldest"
cp -r "${DEFAULT_MODEL}" "${SEQUENCE_OLDEST_MODEL}"
echo -e "\nsequence_batching { oldest { max_queue_delay_microseconds: ${MAX_QUEUE_DELAY_US}, max_candidate_sequences: ${MAX_BATCH_SIZE} } }\n" >> "${SEQUENCE_OLDEST_MODEL}/config.pbtxt"

BASE_ENSEMBLE="ensemble_delay"
ENSEMBLE_MODEL="${MODELDIR}/ensemble"
cp -r "${BASE_ENSEMBLE}" "${ENSEMBLE_MODEL}"
mkdir -p "${ENSEMBLE_MODEL}/1"
# Use uniquely named composing models to avoid clashing
# metric values with individual and ensemble tests.
cp -r "${DEFAULT_MODEL}" "${MODELDIR}/default_composing"
cp -r "${DYNAMIC_MODEL}" "${MODELDIR}/dynamic_composing"


run_and_check_server
python3 ${PYTHON_TEST} 2>&1 | tee ${CLIENT_LOG}
kill $SERVER_PID
wait $SERVER_PID
expected_tests=6
check_unit_test "${expected_tests}"

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
