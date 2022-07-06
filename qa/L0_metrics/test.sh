#!/bin/bash
# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
SERVER_ARGS="--model-repository=${MODELDIR}"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

rm -f $SERVER_LOG

RET=0

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
run_server
if [ "$SERVER_PID" == "0" ]; then
  echo -e "\n***\n*** Failed to start $SERVER\n***"
  cat $SERVER_LOG
  exit 1
fi

num_gpus=`curl -s localhost:8002/metrics | grep "nv_gpu_utilization{" | wc -l`
if [ $num_gpus -ne 3 ]; then
  echo "Found $num_gpus GPU(s) instead of 3 GPUs being monitored."
  echo -e "\n***\n*** GPU metric test failed. \n***"
  RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

export CUDA_VISIBLE_DEVICES=0
run_server
if [ "$SERVER_PID" == "0" ]; then
  echo -e "\n***\n*** Failed to start $SERVER\n***"
  cat $SERVER_LOG
  exit 1
fi

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

SERVER_ARGS="$SERVER_ARGS --metrics-interval-ms=${METRICS_INTERVAL_MS}"
run_server
if [ "$SERVER_PID" == "0" ]; then
  echo -e "\n***\n*** Failed to start $SERVER\n***"
  cat $SERVER_LOG
  exit 1
fi

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

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
