#!/bin/bash

### Helpers ###

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
    MODEL="llama-3-8b-instruct"
    MODEL_REPO="../openai/tests/tensorrtllm_models"
    rm -rf ${MODEL_REPO}

    # FIXME: This will require an upgrade each release to match the TRT-LLM version
    # Use Triton CLI to prepare model repository for testing
    pip install git+https://github.com/triton-inference-server/triton_cli.git@0.0.10
    # NOTE: Could use ENGINE_DEST_PATH set to NFS mount for pre-built engines in future
    triton import \
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
    TEST_LOG="test_openai.log"

    # Capture error code without exiting to allow log collection
    set +e
    pytest -s -v --junitxml=test_openai.xml 2>&1 > ${TEST_LOG}
    if [ $? -ne 0 ]; then
        cat ${TEST_LOG}
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    set -e

    # Collect logs for error analysis when needed
    cp *.xml *.log ../../../
    popd
}

### Test ###

RET=0

pre_test
run_test

exit ${RET}
