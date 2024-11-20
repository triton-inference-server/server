#!/bin/bash
# Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
PYTHON_TEST="metrics_config_test.py"
HISTOGRAM_PYTEST="histogram_metrics_test.py"
source ../common/util.sh

CLIENT_LOG="client.log"
TEST_RESULT_FILE="test_results.txt"
function check_unit_test() {
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
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
UNIT_TEST="./metrics_api_test --gtest_output=xml:metrics_api.report.xml"

rm -fr *.log *.xml

set +e
export CUDA_VISIBLE_DEVICES=0
LD_LIBRARY_PATH=/opt/tritonserver/lib:$LD_LIBRARY_PATH $UNIT_TEST >>$TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $TEST_LOG
    echo -e "\n***\n*** Metrics API Unit Test Failed\n***"
    RET=1
fi
set -e

# Prepare a libtorch float32 model with basic config
rm -rf $MODELDIR
model=libtorch_float32_float32_float32
mkdir -p $MODELDIR/${model}/1 && \
  cp -r $DATADIR/${model}/1/* $MODELDIR/${model}/1/. && \
  cp $DATADIR/${model}/config.pbtxt $MODELDIR/${model}/. && \
  (cd $MODELDIR/${model} && \
  sed -i "s/label_filename:.*//" config.pbtxt && \
  echo "instance_group [{ kind: KIND_GPU }]" >> config.pbtxt)

### CPU / RAM metrics tests
set +e
SERVER_LOG="cpu_metrics_test_server.log"
# NOTE: CPU utilization is computed based on the metrics interval, so having
# too small of an interval can skew the results.
SERVER_ARGS="$BASE_SERVER_ARGS --metrics-interval-ms=1000 --log-verbose=1"
run_and_check_server

CLIENT_PY="./cpu_metrics_test.py"
CLIENT_LOG="cpu_metrics_test_client.log"
python3 -m pytest --junitxml="cpu_metrics.report.xml" ${CLIENT_PY} >> ${CLIENT_LOG} 2>&1
if [ $? -ne 0 ]; then
    cat ${SERVER_LOG}
    cat ${CLIENT_LOG}
    echo -e "\n***\n*** ${CLIENT_PY} FAILED. \n***"
    RET=1
fi

kill_server
set -e

### Pinned memory metrics tests
set +e
CLIENT_PY="./pinned_memory_metrics_test.py"
CLIENT_LOG="pinned_memory_metrics_test_client.log"
SERVER_LOG="pinned_memory_metrics_test_server.log"
SERVER_ARGS="$BASE_SERVER_ARGS --metrics-interval-ms=1 --model-control-mode=explicit --log-verbose=1"
run_and_check_server
python3 ${PYTHON_TEST} MetricsConfigTest.test_pinned_memory_metrics_exist -v 2>&1 | tee ${CLIENT_LOG}
check_unit_test

python3 -m pytest --junitxml="pinned_memory_metrics.report.xml" ${CLIENT_PY} >> ${CLIENT_LOG} 2>&1
if [ $? -ne 0 ]; then
    cat ${SERVER_LOG}
    cat ${CLIENT_LOG}
    echo -e "\n***\n*** ${CLIENT_PY} FAILED. \n***"
    RET=1
fi

kill_server

# Custom Pinned memory pool size
export CUSTOM_PINNED_MEMORY_POOL_SIZE=1024 # bytes
SERVER_LOG="custom_pinned_memory_test_server.log"
CLIENT_LOG="custom_pinned_memory_test_client.log"
SERVER_ARGS="$BASE_SERVER_ARGS --metrics-interval-ms=1 --model-control-mode=explicit --log-verbose=1 --pinned-memory-pool-byte-size=$CUSTOM_PINNED_MEMORY_POOL_SIZE"
run_and_check_server
python3 -m pytest --junitxml="custom_pinned_memory_metrics.report.xml" ${CLIENT_PY} >> ${CLIENT_LOG} 2>&1
if [ $? -ne 0 ]; then
    cat ${SERVER_LOG}
    cat ${CLIENT_LOG}
    echo -e "\n***\n*** Custom ${CLIENT_PY} FAILED. \n***"
    RET=1
fi

kill_server
set -e

# Peer access GPU memory utilization Test
# Custom Pinned memory pool size
export CUSTOM_PINNED_MEMORY_POOL_SIZE=0 # bytes
export CUDA_VISIBLE_DEVICES=0
SERVER_LOG="gpu_peer_memory_test_server.log"
CLIENT_LOG="gpu_peer_memory_test_client.log"

SERVER_ARGS="$BASE_SERVER_ARGS --model-control-mode=explicit --log-verbose=1 --pinned-memory-pool-byte-size=$CUSTOM_PINNED_MEMORY_POOL_SIZE --enable-peer-access=FALSE --cuda-memory-pool-byte-size 0:0 --log-verbose=1"
run_and_check_server
#grep usage stats for triton server from nvidia-smi
memory_size_without_peering=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | grep $(pgrep tritonserver) | awk '{print $3}')

#nvidia-smi only lists process which use gpu memory with --enable-peer-access=FALSE nvidia-smi may not list tritonserver
if [ -z $memory_size_without_peering ]; then
  memory_size_without_peering=0
fi

kill_server

# Check if memory usage HAS reduced to 0 after using the --enable-peer-access flag
if [ $memory_size_without_peering -ne 0 ]; then
   # Print the memory usage for each GPU
  echo "Disabling PEERING does not reduce GPU memory usage to ZERO"
  echo -e "\n***\n*** GPU Peer enable failed. \n***"
  RET=1
fi

### GPU Metrics
set +e
export CUDA_VISIBLE_DEVICES=0,1
SERVER_LOG="./inference_server.log"
CLIENT_LOG="client.log"
run_and_check_server

num_gpus=`curl -s ${TRITONSERVER_IPADDR}:8002/metrics | grep "nv_gpu_utilization{" | wc -l`
if [ $num_gpus -ne 2 ]; then
  echo "Found $num_gpus GPU(s) instead of 2 GPUs being monitored."
  echo -e "\n***\n*** GPU metric test failed. \n***"
  RET=1
fi

kill_server

export CUDA_VISIBLE_DEVICES=0
run_and_check_server

num_gpus=`curl -s ${TRITONSERVER_IPADDR}:8002/metrics | grep "nv_gpu_utilization{" | wc -l`
if [ $num_gpus -ne 1 ]; then
  echo "Found $num_gpus GPU(s) instead of 1 GPU being monitored."
  echo -e "\n***\n*** GPU metric test failed. \n***"
  RET=1
fi
kill_server


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
prev_energy=`curl -s ${TRITONSERVER_IPADDR}:8002/metrics | awk '/nv_energy_consumption{/ {print $2}'`
for (( i = 0; i < $num_iterations; ++i )); do
  sleep $WAIT_INTERVAL_SECS
  current_energy=`curl -s ${TRITONSERVER_IPADDR}:8002/metrics | awk '/nv_energy_consumption{/ {print $2}'`
  if [ $current_energy != $prev_energy ]; then
    echo -e "\n***\n*** Detected changing metrics, warmup completed.\n***"
    break
  fi
  prev_energy=$current_energy
done

prev_energy=`curl -s ${TRITONSERVER_IPADDR}:8002/metrics | awk '/nv_energy_consumption{/ {print $2}'`
for (( i = 0; i < $num_iterations; ++i )); do
  sleep $WAIT_INTERVAL_SECS
  current_energy=`curl -s ${TRITONSERVER_IPADDR}:8002/metrics | awk '/nv_energy_consumption{/ {print $2}'`
  if [ $current_energy == $prev_energy ]; then
    cat $SERVER_LOG
    echo "Metrics were not updated in interval of ${METRICS_INTERVAL_MS} milliseconds"
    echo -e "\n***\n*** Metric Interval test failed. \n***"
    RET=1
    break
  fi
  prev_energy=$current_energy
done

kill_server

### Metric Config CLI and different Metric Types ###
MODELDIR="${PWD}/unit_test_models"
mkdir -p "${MODELDIR}/identity_cache_on/1"
mkdir -p "${MODELDIR}/identity_cache_off/1"
BASE_SERVER_ARGS="--model-repository=${MODELDIR} --model-control-mode=explicit"

# Check default settings: Counters should be enabled, histograms and summaries should be disabled
SERVER_ARGS="${BASE_SERVER_ARGS} --load-model=identity_cache_off"
run_and_check_server
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_counters_exist 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_histograms_decoupled_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_summaries_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_cache_counters_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_cache_summaries_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
kill_server

# Check default settings: Histograms should be always disabled in non-decoupled model.
SERVER_ARGS="${BASE_SERVER_ARGS} --load-model=identity_cache_off --metrics-config histogram_latencies=true"
run_and_check_server
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_counters_exist 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_histograms_decoupled_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_summaries_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_cache_counters_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_cache_summaries_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
kill_server

# Check default settings: Histograms should be disabled in decoupled model
decoupled_model="async_execute_decouple"
mkdir -p "${MODELDIR}/${decoupled_model}/1/"
cp ../python_models/${decoupled_model}/model.py ${MODELDIR}/${decoupled_model}/1/
cp ../python_models/${decoupled_model}/config.pbtxt ${MODELDIR}/${decoupled_model}/

SERVER_ARGS="${BASE_SERVER_ARGS} --load-model=${decoupled_model}"
run_and_check_server
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_counters_exist 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_histograms_decoupled_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_summaries_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_cache_counters_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_cache_summaries_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
kill_server

# Enable histograms in decoupled model
SERVER_ARGS="${BASE_SERVER_ARGS} --load-model=${decoupled_model} --metrics-config histogram_latencies=true"
run_and_check_server
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_counters_exist 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_histograms_decoupled_exist 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_summaries_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_cache_counters_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
python3 ${PYTHON_TEST} MetricsConfigTest.test_cache_summaries_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
kill_server

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
kill_server

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
kill_server

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
kill_server

# Check setting custom summary quantiles
export SUMMARY_QUANTILES="0.1:0.0.1,0.7:0.01,0.75:0.01"
SERVER_ARGS="${BASE_SERVER_ARGS} --load-model=identity_cache_off --metrics-config summary_latencies=true --metrics-config summary_quantiles=${SUMMARY_QUANTILES}"
run_and_check_server
python3 ${PYTHON_TEST} MetricsConfigTest.test_summaries_custom_quantiles 2>&1 | tee ${CLIENT_LOG}
check_unit_test
kill_server

# Check model namespacing label with namespace on and off
REPOS_DIR="${PWD}/model_namespacing_repos"
mkdir -p "${REPOS_DIR}/addsub_repo/addsub_ensemble/1"
mkdir -p "${REPOS_DIR}/subadd_repo/subadd_ensemble/1"
# Namespace on
SERVER_ARGS="--model-repository=${REPOS_DIR}/addsub_repo --model-repository=${REPOS_DIR}/subadd_repo --model-namespacing=true --allow-metrics=true"
run_and_check_server
python3 ${PYTHON_TEST} MetricsConfigTest.test_model_namespacing_label_with_namespace_on 2>&1 | tee ${CLIENT_LOG}
check_unit_test
kill_server
# Namespace off
SERVER_ARGS="--model-repository=${REPOS_DIR}/addsub_repo --model-namespacing=false --allow-metrics=true"
run_and_check_server
python3 ${PYTHON_TEST} MetricsConfigTest.test_model_namespacing_label_with_namespace_off 2>&1 | tee ${CLIENT_LOG}
check_unit_test
kill_server

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
kill_server
expected_tests=6
check_unit_test "${expected_tests}"

### Test histogram data in ensemble decoupled model ###
MODELDIR="${PWD}/ensemble_decoupled"
SERVER_LOG="./histogram_ensemble_decoupled_server.log"
CLIENT_LOG="./histogram_ensemble_decoupled_client.log"
SERVER_ARGS="--model-repository=${MODELDIR} --metrics-config histogram_latencies=true --log-verbose=1"
mkdir -p "${MODELDIR}"/ensemble/1
cp -r "${MODELDIR}"/async_execute_decouple "${MODELDIR}"/async_execute
sed -i "s/model_transaction_policy { decoupled: True }//" "${MODELDIR}"/async_execute/config.pbtxt

run_and_check_server
python3 ${HISTOGRAM_PYTEST} TestHistogramMetrics.test_ensemble_decoupled 2>&1 | tee ${CLIENT_LOG}
kill_server
check_unit_test

### Test model metrics configuration
MODELDIR="${PWD}/model_metrics_model"
SERVER_LOG="./model_metric_config_server.log"
CLIENT_LOG="./model_metric_config_client.log"
decoupled_model="async_execute_decouple"
rm -rf "${MODELDIR}/${decoupled_model}"
mkdir -p "${MODELDIR}/${decoupled_model}/1/"
cp ../python_models/${decoupled_model}/model.py ${MODELDIR}/${decoupled_model}/1/

# Test valid model_metrics config
cp ../python_models/${decoupled_model}/config.pbtxt ${MODELDIR}/${decoupled_model}/
cat >> "${MODELDIR}/${decoupled_model}/config.pbtxt" << EOL
model_metrics {
  metric_control: [
    {
      metric_identifier: {
        family: "nv_inference_first_response_histogram_ms"
      }
      histogram_options: {
        buckets: [ -1, 0.0, 1, 2.5 ]
      }
    }
  ]
}
EOL

SERVER_ARGS="--model-repository=${MODELDIR} --model-control-mode=explicit --load-model=${decoupled_model} --metrics-config histogram_latencies=true --log-verbose=1"
run_and_check_server
export OVERRIDE_BUCKETS="-1,0,1,2.5,+Inf"
python3 ${HISTOGRAM_PYTEST} TestHistogramMetrics.test_buckets_override 2>&1 | tee ${CLIENT_LOG}
check_unit_test
kill_server

# Test valid model_metrics config with histogram disabled
PYTHON_TEST="metrics_config_test.py"
SERVER_ARGS="--model-repository=${MODELDIR} --model-control-mode=explicit --load-model=${decoupled_model} --metrics-config histogram_latencies=false --log-verbose=1"
run_and_check_server
python3 ${PYTHON_TEST} MetricsConfigTest.test_inf_histograms_decoupled_missing 2>&1 | tee ${CLIENT_LOG}
check_unit_test
kill_server

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
