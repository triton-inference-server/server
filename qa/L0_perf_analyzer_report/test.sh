#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
if [ -z "${REPO_VERSION}" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi
if [ ! -z "$TEST_REPO_ARCH" ]; then
    REPO_VERSION=${REPO_VERSION}_${TEST_REPO_ARCH}
fi

source ../common/util.sh

# Setup client/perf_analyzer
CLIENT_LOG="./perf_analyzer.log"
PERF_ANALYZER=../clients/perf_analyzer

function check_perf_analyzer_error {
    ERROR_STRING="error | Request count: 0 | : 0 infer/sec"
    CLIENT_RET="$1"
    if [ ${CLIENT_RET} -ne 0 ]; then
        cat ${CLIENT_LOG}
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    if [ $(cat ${CLIENT_LOG} |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
        cat ${CLIENT_LOG}
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
}

function check_cache_output {
    # Validate cache info in perf_analyzer output
    CACHE_STRING="Cache hit count"
    if [ $(cat ${CLIENT_LOG} |  grep -i "${CACHE_STRING}" | wc -l) -eq 0 ]; then
        cat ${CLIENT_LOG}
	echo "ERROR: No cache hit count found in output"
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi

    # Validate non-zero number of cache hits
    ERROR_STRING="Cache hit count: 0"
    num_cache_hit_lines=$(cat ${CLIENT_LOG} |  grep -i "${CACHE_STRING}" | wc -l)
    num_cache_hit_zero_lines=$(cat ${CLIENT_LOG} |  grep -i "${ERROR_STRING}" | wc -l)
    # Top-level ensemble model requests do not currently support caching and
    # will always report a cache hit count of zero if any composing model
    # has caching enabled. So we check that at least one model reports
    # non-zero cache hits for now.
    # TODO: When ensemble models support cache hits, this should just fail
    #       for any occurrence of ERROR_STRING
    if [ ${num_cache_hit_lines} -eq ${num_cache_hit_zero_lines} ]; then
        cat ${CLIENT_LOG}
	echo "ERROR: All cache hit counts were zero, expected a non-zero number of cache hits"
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
}

# Setup server
export CUDA_VISIBLE_DEVICES=0
SERVER=/opt/tritonserver/bin/tritonserver
# --response-cache-byte-size must be non-zero to test models with cache enabled
SERVER_ARGS="--model-repository=`pwd`/models --response-cache-byte-size=8192"
SERVER_LOG="./inference_server.log"

# Setup model repository from existing qa_model_repository
rm -f $SERVER_LOG $CLIENT_LOG
MODEL_DIR="./models"
rm -fr ${MODEL_DIR} && mkdir ${MODEL_DIR}
ENSEMBLE_MODEL="simple_onnx_float32_float32_float32"
COMPOSING_MODEL="onnx_float32_float32_float32"
ENSEMBLE_MODEL_CACHE_ENABLED="${ENSEMBLE_MODEL}_cache_enabled"
ENSEMBLE_MODEL_CACHE_DISABLED="${ENSEMBLE_MODEL}_cache_disabled"
COMPOSING_MODEL_CACHE_ENABLED="${COMPOSING_MODEL}_cache_enabled"
COMPOSING_MODEL_CACHE_DISABLED="${COMPOSING_MODEL}_cache_disabled"
MODELS="${ENSEMBLE_MODEL_CACHE_ENABLED} ${ENSEMBLE_MODEL_CACHE_DISABLED} ${COMPOSING_MODEL_CACHE_ENABLED} ${COMPOSING_MODEL_CACHE_DISABLED}"

## Setup ensemble models, one with cache enabled and one with cache disabled
cp -r "/data/inferenceserver/${REPO_VERSION}/qa_ensemble_model_repository/qa_model_repository/${ENSEMBLE_MODEL}" "${MODEL_DIR}/${ENSEMBLE_MODEL_CACHE_ENABLED}"
cp -r "/data/inferenceserver/${REPO_VERSION}/qa_ensemble_model_repository/qa_model_repository/${ENSEMBLE_MODEL}" "${MODEL_DIR}/${ENSEMBLE_MODEL_CACHE_DISABLED}"

## Setup composing models, one with cache enabled and one with cache disabled
cp -r "/data/inferenceserver/${REPO_VERSION}/qa_model_repository/${COMPOSING_MODEL}" "${MODEL_DIR}/${COMPOSING_MODEL_CACHE_ENABLED}"
cp -r "/data/inferenceserver/${REPO_VERSION}/qa_model_repository/${COMPOSING_MODEL}" "${MODEL_DIR}/${COMPOSING_MODEL_CACHE_DISABLED}"

for model in ${MODELS}; do
    # Remove "name" line from each config to use directory name for simplicity
    sed -i "/^name:/d" "${MODEL_DIR}/${model}/config.pbtxt"
    # Add version directory to each model if non-existent
    mkdir -p "${MODEL_DIR}/${model}/1"
done

## Update "model_name" lines in each ensemble model config ensemble steps
sed -i "s/${COMPOSING_MODEL}/${COMPOSING_MODEL_CACHE_ENABLED}/g" "${MODEL_DIR}/${ENSEMBLE_MODEL_CACHE_ENABLED}/config.pbtxt"
sed -i "s/${COMPOSING_MODEL}/${COMPOSING_MODEL_CACHE_DISABLED}/g" "${MODEL_DIR}/${ENSEMBLE_MODEL_CACHE_DISABLED}/config.pbtxt"

## Append cache config to each model config 
echo -e "response_cache { enable: True }" >> "${MODEL_DIR}/${ENSEMBLE_MODEL_CACHE_ENABLED}/config.pbtxt"
echo -e "response_cache { enable: False }" >> "${MODEL_DIR}/${ENSEMBLE_MODEL_CACHE_DISABLED}/config.pbtxt"
echo -e "response_cache { enable: True }" >> "${MODEL_DIR}/${COMPOSING_MODEL_CACHE_ENABLED}/config.pbtxt"
echo -e "response_cache { enable: False }" >> "${MODEL_DIR}/${COMPOSING_MODEL_CACHE_DISABLED}/config.pbtxt"
# Force CPU memory for composing models since cache doesn't currently support GPU memory
echo -e "instance_group [{ kind: KIND_CPU, count: 1 }]" >> "${MODEL_DIR}/${COMPOSING_MODEL_CACHE_ENABLED}/config.pbtxt"
echo -e "instance_group [{ kind: KIND_CPU, count: 1 }]" >> "${MODEL_DIR}/${COMPOSING_MODEL_CACHE_DISABLED}/config.pbtxt"

# Run server
run_server
if [ "${SERVER_PID}" == "0" ]; then
    echo -e "\n***\n*** Failed to start ${SERVER}\n***"
    cat ${SERVER_LOG}
    exit 1
fi

# Run perf_analyzer
set +e
RET=0
PROTOCOLS="http grpc"
STABILITY_THRESHOLD="15"
for protocol in ${PROTOCOLS}; do
    for model in ${MODELS}; do
	echo "================================================================"
	echo "[PERMUTATION] Protocol=${protocol} Model=${model}"
	echo "================================================================"

        ${PERF_ANALYZER} -v -i ${protocol} -m ${model} -s ${STABILITY_THRESHOLD} | tee ${CLIENT_LOG} 2>&1
        check_perf_analyzer_error $?

	# Check response cache outputs
	if [[ ${model} == *"cache_enabled"* ]]; then
	  check_cache_output
	fi
    done;
done;
set -e

# Cleanup
kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo "=== START SERVER LOG ==="
  cat ${SERVER_LOG}
  echo "=== END SERVER LOG ==="
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit ${RET}
