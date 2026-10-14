#!/bin/bash
# Copyright 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Tests the ORCA endpoint-load-metrics header (src/orca_http.cc), which the
# server computes from the nv_trt_llm_kv_cache_block_metrics gauges. The
# TensorRT-LLM LLM API backend does not publish them (the C++ backend did), so
# a small Python model publishes fixed values instead. This keeps the
# server-side header logic covered without a TensorRT-LLM engine or a GPU.

source ../common/util.sh

RET=0
NAME="orca_kv_metrics_test"
MODEL_NAME="orca_kv_metrics"
MODEL_REPOSITORY="$(pwd)/models"
TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends
SERVER_ARGS="--model-repository=${MODEL_REPOSITORY} --backend-directory=${BACKEND_DIR} --log-verbose=1"
SERVER_LOG="${NAME}_server.log"
SERVER_TIMEOUT=${SERVER_TIMEOUT:=120}
CLIENT_PY=$(pwd)/orca_http_test.py
CLIENT_LOG="${NAME}_orca_http_test.log"

rm -rf ${MODEL_REPOSITORY} ${SERVER_LOG} ${CLIENT_LOG}
mkdir -p ${MODEL_REPOSITORY}/${MODEL_NAME}/1
cp ../python_models/orca_kv_metrics/config.pbtxt ${MODEL_REPOSITORY}/${MODEL_NAME}/
cp ../python_models/orca_kv_metrics/model.py ${MODEL_REPOSITORY}/${MODEL_NAME}/1/

set +e
run_server

if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

python3 $CLIENT_PY "http://localhost:8000/v2/models/${MODEL_NAME}/generate" >>$CLIENT_LOG 2>&1

if [ $? -ne 0 ]; then
    echo "Failed: Client test had a non-zero return code."
    RET=1
fi

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** ORCA Test Passed\n***"
else
    cat $SERVER_LOG
    cat $CLIENT_LOG
    echo -e "\n***\n*** ORCA Test FAILED\n***"
fi

kill_server
set -e
exit $RET
