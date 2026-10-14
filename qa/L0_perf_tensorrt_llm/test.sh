#!/bin/bash
# Copyright 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Benchmarks the TensorRT-LLM LLM API (Python) model, the same way the C++
# backend was benchmarked, so results stay comparable run to run. Execution now
# goes through an out-of-process Python stub (shared-memory IPC, the GIL), so
# numbers from the C++ backend are not comparable and the reporting baseline
# has to be re-established.

RET=0
BASE_DIR=$(pwd)

MODEL_NAME="tensorrt_llm"
NAME="tensorrt_llm_benchmarking_test"
MODEL_REPOSITORY="$(pwd)/triton_model_repo"
TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER_LOG="${NAME}_server.log"
# A benchmark wants the KV cache the model would normally get, not the small
# share the functional tests use to leave room for other servers.
LLMAPI_KV_CACHE_FRACTION=${LLMAPI_KV_CACHE_FRACTION:="0.5"}
source ../common/trtllm_util.sh

# Install perf_analyzer
pip3 install perf_analyzer

ARCH="amd64"
STATIC_BATCH=1
INSTANCE_CNT=1
CONCURRENCY=100
MODEL_FRAMEWORK="tensorrt-llm"
PERF_CLIENT="perf_analyzer"
REPORTER=../common/reporter.py
INPUT_DATA="./input_data.json"
PERF_CLIENT_PROTOCOL="grpc"
EXPORT_FILE=profile-export-tensorrt-llm-model.json
rm -rf *.tjson *.json *.csv *log

# Streaming needs a decoupled model: model.yaml alone is not enough, the
# launcher disables auto-complete-config, so the policy goes in config.pbtxt.
prepare_llmapi_model_repository ${MODEL_REPOSITORY} 1 1 1

echo '{
  "data": [
    {
      "text_input": ["Hello, my name is"],
      "streaming": [true],
      "sampling_param_max_tokens": [16]
    }
  ]
}' >$INPUT_DATA

# Set stability-percentage 999 to bypass the stability check in PA.
# LLM generates a sequence of tokens that is unlikely to be within a reasonable bound to determine valid measurement in terms of latency.
# Using "count_windows" measurement mode, which automatically extends the window for collecting responses.
PERF_CLIENT_ARGS="-v -m $MODEL_NAME -i $PERF_CLIENT_PROTOCOL --async --streaming --input-data=$INPUT_DATA --profile-export-file=$EXPORT_FILE \
                  --shape=text_input:1 --measurement-mode=count_windows \
                  --concurrency-range=$CONCURRENCY --measurement-request-count=10 --stability-percentage=999"

set +e
run_llmapi_server ${MODEL_REPOSITORY}

if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start the server\n***"
    cat $SERVER_LOG
    exit 1
fi

$PERF_CLIENT $PERF_CLIENT_ARGS -f ${NAME}.csv 2>&1 | tee ${NAME}_perf_analyzer.log
set +o pipefail

kill_llmapi_server
set -e

echo -e "[{\"s_benchmark_kind\":\"benchmark_perf\"," >>${NAME}.tjson
echo -e "\"s_benchmark_repo_branch\":\"${BENCHMARK_REPO_BRANCH}\"," >>${NAME}.tjson
echo -e "\"s_benchmark_name\":\"${NAME}\"," >>${NAME}.tjson
echo -e "\"s_server\":\"triton\"," >>${NAME}.tjson
echo -e "\"s_protocol\":\"${PERF_CLIENT_PROTOCOL}\"," >>${NAME}.tjson
echo -e "\"s_framework\":\"${MODEL_FRAMEWORK}\"," >>${NAME}.tjson
echo -e "\"s_model\":\"${MODEL_NAME}\"," >>${NAME}.tjson
echo -e "\"l_concurrency\":${CONCURRENCY}," >>${NAME}.tjson
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

if (($RET == 0)); then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
