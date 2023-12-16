#!/bin/bash
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

source ../common/util.sh

REPORTER=../common/reporter.py
TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends
MODEL_REPO="${PWD}/models"
NAME="vllm_benchmarking_test"
MODEL_NAME="gpt2_vllm"
INPUT_DATA="./input_data.json"
SERVER_LOG="${NAME}_server.log"
SERVER_ARGS="--model-repository=${MODEL_REPO} --backend-directory=${BACKEND_DIR} --log-verbose=1"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:=0}
EXPORT_FILE=profile-export-vllm-model.json

pip3 install tritonclient nvidia-ml-py3
rm -rf $MODEL_REPO $EXPORT_FILE *.tjson *.json *.csv

mkdir -p $MODEL_REPO/$MODEL_NAME/1
echo '{
    "model":"gpt2",
    "disable_log_requests": "true",
    "gpu_memory_utilization": 0.5
}' >$MODEL_REPO/$MODEL_NAME/1/model.json

echo 'backend: "vllm"
instance_group [
  {
    count: 1
    kind: KIND_MODEL
  }
]' >$MODEL_REPO/$MODEL_NAME/config.pbtxt

echo '{
    "data": [
        {
            "text_input": [
                "hi hi hi hi hi hi hi hi hi hi"
            ],
            "stream": [
                true
            ],
            "sampling_parameters": [
                "{\"max_tokens\": 1024, \"ignore_eos\": true}"
            ]
        }
    ]
}' >$INPUT_DATA

RET=0
ARCH="amd64"
STATIC_BATCH=1
INSTANCE_CNT=1
CONCURRENCY=100
MODEL_FRAMEWORK="vllm"
PERF_CLIENT_PROTOCOL="grpc"
PERF_CLIENT=perf_analyzer

# Set stability-percentage 999 to bypass the stability check in PA.
# LLM generates a sequence of tokens that is unlikely to be within a reasonable bound to determine valid measurement in terms of latency.
# Using "count_windows" measurement mode, which automatically extends the window for collecting responses.
PERF_CLIENT_ARGS="-v -m $MODEL_NAME --concurrency-range=${CONCURRENCY} --measurement-mode=count_windows --measurement-request-count=10 \
                  --input-data=$INPUT_DATA --profile-export-file=$EXPORT_FILE -i $PERF_CLIENT_PROTOCOL --async --streaming --stability-percentage=999"

run_server
if (($SERVER_PID == 0)); then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
$PERF_CLIENT $PERF_CLIENT_ARGS -f ${NAME}.csv 2>&1 | tee ${NAME}_perf_analyzer.log
set +o pipefail
set -e

if [[ -n "${SERVER_PID}" ]]; then
    kill $SERVER_PID
    wait $SERVER_PID
fi

echo -e "[{\"s_benchmark_kind\":\"benchmark_perf\"," >>${NAME}.tjson
echo -e "\"s_benchmark_repo_branch\":\"${BENCHMARK_REPO_BRANCH}\"," >>${NAME}.tjson
echo -e "\"s_benchmark_name\":\"${NAME}\"," >>${NAME}.tjson
echo -e "\"s_server\":\"triton\"," >>${NAME}.tjson
echo -e "\"s_protocol\":\"${PERF_CLIENT_PROTOCOL}\"," >>${NAME}.tjson
echo -e "\"s_framework\":\"${MODEL_FRAMEWORK}\"," >>${NAME}.tjson
echo -e "\"s_model\":\"${MODEL_NAME}\"," >>${NAME}.tjson
echo -e "\"l_concurrency\":\"${CONCURRENCY}\"," >>${NAME}.tjson
echo -e "\"l_batch_size\":${STATIC_BATCH}," >>${NAME}.tjson
echo -e "\"l_instance_count\":${INSTANCE_CNT}," >>${NAME}.tjson
echo -e "\"s_architecture\":\"${ARCH}\"}]" >>${NAME}.tjson

if [ -f $REPORTER ]; then
    set +e

    URL_FLAG=
    if [ ! -z ${BENCHMARK_REPORTER_URL} ]; then
        URL_FLAG="-u ${BENCHMARK_REPORTER_URL}"
    fi

    python3 $REPORTER -v -e ${EXPORT_FILE} -o ${NAME}.json --csv ${NAME}.csv --gpu-metrics --token-latency ${URL_FLAG} ${NAME}.tjson
    if (($? != 0)); then
        RET=1
    fi

    set -e
fi

rm -rf $MODEL_REPO $INPUT_DATA

if (($RET == 0)); then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
