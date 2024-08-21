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
DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
# NOTE: This default path doesn't make sense if run on non-A100 GPU
ENGINE_DEST_PATH=${ENGINE_DEST_PATH:="${DATADIR}/trtllm_engines_A100"}

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
    # Use Triton CLI to prepare model repository for testing
    pip install git+https://github.com/triton-inference-server/triton_cli.git@0.0.10
    MODEL_REPO="../openai/tests/tensorrtllm_models"
    rm -rf ${MODEL_REPO}
    # Use ENGINE_DEST_PATH to re-use NFS mount when possible
    ENGINE_DEST_PATH="${ENGINE_DEST_PATH}" triton import \
        --model llama-3-8b-instruct \
        --backend tensorrtllm \
        --model-repository "${MODEL_REPO}"
    # WAR for tests expecting default name of "tensorrt_llm_bls"
    mv "${MODEL_REPO}/llama-3-8b-instruct" "${MODEL_REPO}/tensorrt_llm_bls"
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

pre_test
run_test
post_test
