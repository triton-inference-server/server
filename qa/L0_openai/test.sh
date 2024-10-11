#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

### Helpers ###

function install_deps() {
    # Install python bindings for tritonserver and tritonfrontend
    pip install /opt/tritonserver/python/triton*.whl

    # Install application/testing requirements
    pushd openai/
    pip install -r requirements.txt
    pip install -r requirements-test.txt

    if [ "${IMAGE_KIND}" == "TRTLLM" ]; then
        prepare_tensorrtllm
    else
        prepare_vllm
    fi
    popd
}

function prepare_vllm() {
    echo "No prep needed for vllm currently"
}

function prepare_tensorrtllm() {
    MODEL="llama-3-8b-instruct"
    MODEL_REPO="tests/tensorrtllm_models"
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
    pushd openai/
    TEST_LOG="test_openai.log"

    # Capture error code without exiting to allow log collection
    set +e
    pytest -s -v --junitxml=test_openai.xml tests/ 2>&1 > ${TEST_LOG}
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
