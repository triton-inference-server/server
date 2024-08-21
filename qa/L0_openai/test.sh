#!/bin/bash

### Helpers ###

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

function install_deps() {
    pushd openai/docker
    pip install /opt/tritonserver/python/triton*.whl
    pip install -r requirements.txt
    if [ "${IMAGE_KIND}" == "TRTLLM" ]; then
        prepare_tensorrtllm
    else
        prepare_vllm
    fi
    popd
}

function prepare_vllm() {
    pip install -r requirements_vllm.txt
}

function prepare_tensorrtllm() {
    DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
    # NOTE: This default path doesn't make sense if run on non-A100 GPU
    NFS_ENGINE_DEST_PATH=${ENGINE_DEST_PATH:="${DATADIR}/trtllm_engines_A100"}
    LOCAL_ENGINE_DEST_PATH="./trtllm_engines_A100"

    MODEL="llama-3-8b-instruct"
    MODEL_REPO="../openai/tests/tensorrtllm_models"
    rm -rf ${MODEL_REPO} ${LOCAL_ENGINE_DEST_PATH}

    # FIXME: This will require an upgrade each release to match the TRT-LLM version
    # Use Triton CLI to prepare model repository for testing
    pip install git+https://github.com/triton-inference-server/triton_cli.git@0.0.10
    # Use ENGINE_DEST_PATH to re-use NFS mount when possible and skip engine build
    ENGINE_DEST_PATH="${NFS_ENGINE_DEST_PATH}" triton import \
        --model ${MODEL}  \
        --backend tensorrtllm \
        --model-repository "${MODEL_REPO}"
    rm -rf "${MODEL_REPO}"

    # To avoid too much I/O with NFS mount at test time, copy it out to a local dir first.
    time rsync -ah ${NFS_ENGINE_DEST_PATH} ${LOCAL_ENGINE_DEST_PATH}
    ENGINE_DEST_PATH="${LOCAL_ENGINE_DEST_PATH}" triton import \
        --model ${MODEL}  \
        --backend tensorrtllm \
        --model-repository "${MODEL_REPO}"

    # WAR for tests expecting default name of "tensorrt_llm_bls"
    mv "${MODEL_REPO}/${MODEL}" "${MODEL_REPO}/tensorrt_llm_bls"
}

function pre_test() {
    # Cleanup
    rm -rf openai/
    rm -f *.xml *.log

    # Prep test environment
    cp -r ../../python/openai .
    install_deps
}

function run_test() {
    pushd openai/openai/tests
    pytest -s -v --junitxml=test_openai.xml 2>&1 | tee test_openai.log
    cp *.xml *.log ../../../
    popd
}

function post_test() {
    # Placeholder
    echo "post_test"
}


### Test ###

set +e
pre_test
run_test
post_test
set -e
