#!/bin/bash
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

RET=0
BASE_DIR=$(pwd)
NUM_GPUS=${NUM_GPUS:=1}
TENSORRTLLM_BACKEND_REPO_TAG=${TENSORRTLLM_BACKEND_REPO_TAG:="main"}
TRT_ROOT="/usr/local/tensorrt"

MODEL_NAME="gpt2_tensorrt_llm"
NAME="tensorrt_llm_benchmarking_test"
MODEL_REPOSITORY="$(pwd)/triton_model_repo"
TENSORRTLLM_BACKEND_DIR="/workspace/tensorrtllm_backend"
GPT_DIR="$TENSORRTLLM_BACKEND_DIR/tensorrt_llm/examples/gpt"
TOKENIZER_DIR="$GPT_DIR/gpt2"
ENGINES_DIR="${BASE_DIR}/engines/inflight_batcher_llm/${NUM_GPUS}-gpu"
TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends
SERVER_LOG="${NAME}_server.log"
SERVER_TIMEOUT=${SERVER_TIMEOUT:=120}

function clone_tensorrt_llm_backend_repo {
    rm -rf $TENSORRTLLM_BACKEND_DIR && mkdir $TENSORRTLLM_BACKEND_DIR
    apt-get update && apt-get install git-lfs -y --no-install-recommends
    git clone --single-branch --depth=1 -b ${TENSORRTLLM_BACKEND_REPO_TAG} ${TRITON_REPO_ORG}/tensorrtllm_backend.git $TENSORRTLLM_BACKEND_DIR
    cd $TENSORRTLLM_BACKEND_DIR && git lfs install && git submodule update --init --recursive
}

# Update Open MPI to a version compatible with SLURM.
function upgrade_openmpi {
    local CURRENT_VERSION=$(mpirun --version 2>&1 | awk '/Open MPI/ {gsub(/rc[0-9]+/, "", $NF); print $NF}')

    if [ -n "$CURRENT_VERSION" ] && dpkg --compare-versions "$CURRENT_VERSION" lt "5.0.1"; then
        # Uninstall the current version of Open MPI
        rm -r /opt/hpcx/ompi/ /usr/local/mpi && rm -rf /usr/lib/$(gcc -print-multiarch)/openmpi || {
            echo "Failed to uninstall the existing Open MPI version $CURRENT_VERSION."
            exit 1
        }
    else
        echo "The installed Open MPI version ($CURRENT_VERSION) is 5.0.1 or higher. Skipping the upgrade."
        return
    fi

    # Install SLURM supported Open MPI version
    cd /tmp/
    wget "https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.1.tar.gz" || {
        echo "Failed to download Open MPI 5.0.1"
        exit 1
    }
    rm -rf openmpi-5.0.1 && tar -xzf openmpi-5.0.1.tar.gz && cd openmpi-5.0.1 || {
        echo "Failed to extract Open MPI 5.0.1"
        exit 1
    }
    ./configure --prefix=/opt/hpcx/ompi/ && make && make install || {
        echo "Failed to install Open MPI 5.0.1"
        exit 1
    }

    # Update environment variables
    if ! grep -q '/opt/hpcx/ompi/bin' ~/.bashrc; then
        echo 'export PATH=/opt/hpcx/ompi/bin:$PATH' >>~/.bashrc
    fi

    if ! grep -q '/opt/hpcx/ompi/lib' ~/.bashrc; then
        echo 'export LD_LIBRARY_PATH=/opt/hpcx/ompi/lib:$LD_LIBRARY_PATH' >>~/.bashrc
    fi
    ldconfig
    source ~/.bashrc
    cd "$BASE_DIR"
    mpirun --version
}

function build_gpt2_base_model {
    # Download weights from HuggingFace Transformers
    cd ${GPT_DIR} && rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2 && cd gpt2
    rm pytorch_model.bin model.safetensors
    if ! wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin; then
        echo "Downloading pytorch_model.bin failed."
        exit 1
    fi
    cd ${GPT_DIR}

    # Convert weights from HF Tranformers to FT format
    python3 convert_checkpoint.py --model_dir gpt2 --dtype float16 --tp_size ${NUM_GPUS} --output_dir "./c-model/gpt2/${NUM_GPUS}-gpu/"
    cd ${BASE_DIR}
}

function build_gpt2_tensorrt_engine {
    # Build TensorRT engines
    cd ${GPT_DIR}
    trtllm-build --checkpoint_dir "./c-model/gpt2/${NUM_GPUS}-gpu/" \
        --gpt_attention_plugin float16 \
        --remove_input_padding enable \
        --paged_kv_cache enable \
        --gemm_plugin float16 \
        --workers "${NUM_GPUS}" \
        --output_dir "${ENGINES_DIR}"

    cd ${BASE_DIR}
}

function replace_config_tags {
    tag_to_replace="${1}"
    new_value="${2}"
    config_file_path="${3}"
    sed -i "s|${tag_to_replace}|${new_value}|g" ${config_file_path}
}

function prepare_model_repository {
    rm -rf ${MODEL_REPOSITORY} && mkdir ${MODEL_REPOSITORY}
    cp -r ${TENSORRTLLM_BACKEND_DIR}/all_models/inflight_batcher_llm/* ${MODEL_REPOSITORY}
    rm -rf ${MODEL_REPOSITORY}/tensorrt_llm_bls
    mv "${MODEL_REPOSITORY}/ensemble" "${MODEL_REPOSITORY}/${MODEL_NAME}"

    replace_config_tags "model_version: -1" "model_version: 1" "${MODEL_REPOSITORY}/${MODEL_NAME}/config.pbtxt"
    replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_REPOSITORY}/${MODEL_NAME}/config.pbtxt"
    replace_config_tags 'name: "ensemble"' "name: \"$MODEL_NAME\"" "${MODEL_REPOSITORY}/${MODEL_NAME}/config.pbtxt"

    replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_REPOSITORY}/preprocessing/config.pbtxt"
    replace_config_tags '${preprocessing_instance_count}' '1' "${MODEL_REPOSITORY}/preprocessing/config.pbtxt"
    replace_config_tags '${tokenizer_dir}' "${TOKENIZER_DIR}/" "${MODEL_REPOSITORY}/preprocessing/config.pbtxt"

    replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_REPOSITORY}/postprocessing/config.pbtxt"
    replace_config_tags '${postprocessing_instance_count}' '1' "${MODEL_REPOSITORY}/postprocessing/config.pbtxt"
    replace_config_tags '${tokenizer_dir}' "${TOKENIZER_DIR}/" "${MODEL_REPOSITORY}/postprocessing/config.pbtxt"

    replace_config_tags '${triton_max_batch_size}' "128" "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${decoupled_mode}' 'true' "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${max_queue_delay_microseconds}' "1000000" "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${batching_strategy}' 'inflight_fused_batching' "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${engine_dir}' "${ENGINES_DIR}" "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${triton_backend}' "tensorrtllm" "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
    replace_config_tags '${max_queue_size}' "0" "${MODEL_REPOSITORY}/tensorrt_llm/config.pbtxt"
}

# Wait until server health endpoint shows ready. Sets WAIT_RET to 0 on
# success, 1 on failure
function wait_for_server_ready() {
    local wait_time_secs="${1:-30}"
    shift
    local spids=("$@")

    WAIT_RET=0

    for _ in $(seq "$wait_time_secs"); do
        for pid in "${spids[@]}"; do
            if ! kill -0 "$pid" >/dev/null 2>&1; then
                echo "=== Server not running."
                WAIT_RET=1
                return
            fi
        done

        sleep 1

        if curl -s --fail localhost:8000/v2/health/ready &&
            curl -s --fail -w "%{http_code}" -o /dev/null -d '{"log_verbose_level":1}' localhost:8000/v2/logging; then
            return
        fi
    done

    echo "=== Timeout $wait_time_secs secs. Server not ready."
    WAIT_RET=1
}

function run_server {
    python3 ${TENSORRTLLM_BACKEND_DIR}/scripts/launch_triton_server.py --world_size="${NUM_GPUS}" --model_repo="${MODEL_REPOSITORY}" >${SERVER_LOG} 2>&1 &
    sleep 2 # allow time to obtain the pid(s)
    # Read PIDs into an array, trimming whitespaces
    readarray -t SERVER_PID < <(pgrep "tritonserver")

    wait_for_server_ready ${SERVER_TIMEOUT} "${SERVER_PID[@]}"
    if [ "$WAIT_RET" != "0" ]; then
        # Cleanup
        kill "${SERVER_PID[@]}" >/dev/null 2>&1 || true
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi
}

function kill_server {
    pgrep tritonserver | xargs kill -SIGINT
    for pid in "${SERVER_PID[@]}"; do
        echo "Waiting for proc ${pid} to terminate..."
        while kill -0 $pid >/dev/null 2>&1; do
            sleep 1
        done
    done
}

upgrade_openmpi
clone_tensorrt_llm_backend_repo
build_gpt2_base_model
build_gpt2_tensorrt_engine
prepare_model_repository

# Install perf_analyzer
pip3 install tritonclient

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

echo '{
  "data": [
    {
      "text_input": ["Hello, my name is"],
      "stream": [true],
      "max_tokens": [16],
      "bad_words": [""],
      "stop_words": [""]
    }
  ]
}' >$INPUT_DATA

# Set stability-percentage 999 to bypass the stability check in PA.
# LLM generates a sequence of tokens that is unlikely to be within a reasonable bound to determine valid measurement in terms of latency.
# Using "count_windows" measurement mode, which automatically extends the window for collecting responses.
PERF_CLIENT_ARGS="-v -m $MODEL_NAME -i $PERF_CLIENT_PROTOCOL --async --streaming --input-data=$INPUT_DATA --profile-export-file=$EXPORT_FILE \
                  --shape=text_input:1 --shape=max_tokens:1 --shape=bad_words:1 --shape=stop_words:1 --measurement-mode=count_windows \
                  --concurrency-range=$CONCURRENCY --measurement-request-count=10 --stability-percentage=999"

set +e
run_server

$PERF_CLIENT $PERF_CLIENT_ARGS -f ${NAME}.csv 2>&1 | tee ${NAME}_perf_analyzer.log
set +o pipefail

kill_server
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
