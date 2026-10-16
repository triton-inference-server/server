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

source ../common/trtllm_util.sh

# Models served by the LLM API (IMAGE_KIND=TRTLLM) and their tokenizers. The
# tests infer the repository from the backend, see tests/conftest.py.
LLAMA_MODEL=${LLAMA_MODEL:="meta-llama/Llama-3.1-8B-Instruct"}
MISTRAL_MODEL=${MISTRAL_MODEL:="mistralai/Mistral-Nemo-Instruct-2407"}
# These servers run one model at a time and need more KV cache than the small
# share the functional TensorRT-LLM tests use.
LLMAPI_KV_CACHE_FRACTION=${LLMAPI_KV_CACHE_FRACTION:="0.5"}

### Helpers ###

function install_deps() {
    # Install python bindings for tritonserver and tritonfrontend
    # pip install /opt/tritonserver/python/triton*.whl

    # Install application/testing requirements
    pushd openai/
    # NOTE: Should be pre-installed in container, but can uncomment if needed
    # pip install -r requirements.txt
    pip install -r requirements-test.txt

    if [ "${IMAGE_KIND}" == "TRTLLM" ]; then
        prepare_llmapi ${LLAMA_MODEL} tests/llmapi_models
        prepare_llmapi ${MISTRAL_MODEL} tests/llmapi_mistral_models
    else
        prepare_vllm
    fi
    popd
}

function prepare_vllm() {
    echo "No prep needed for vllm currently"
}

# The LLM API model streams, so the repository is decoupled. The frontend loads
# it through the in-process server, which would apply model.yaml's decoupled
# flag itself; the policy is declared in config.pbtxt as well so that the
# repository is the same as the ones launch_triton_server.py serves.
function prepare_llmapi() {
    MODEL="$1"
    MODEL_REPO="$2"

    LLMAPI_HF_MODEL=${MODEL} prepare_llmapi_model_repository ${MODEL_REPO} 1 1 1
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
    TEST_XML="test_openai.xml"
    TEST_LOG_MISTRAL="test_openai_mistral.log"
    TEST_XML_MISTRAL="test_openai_mistral.xml"

    set +e
    pytest -s -v --junitxml=${TEST_XML} tests/ 2>&1 | tee ${TEST_LOG}
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    set -e

    if [ "$RET" == "0" ]; then
        # rerun the tool calling tests with mistral model to cover the mistral tool call parser
        set +e
        TEST_TOOL_CALL_PARSER="mistral" TEST_TOKENIZER="mistralai/Mistral-Nemo-Instruct-2407" \
            pytest -s -v --junitxml=${TEST_XML_MISTRAL} tests/test_tool_calling.py 2>&1 | tee ${TEST_LOG_MISTRAL}
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        set -e
    fi

    # Collect logs for error analysis when needed
    cp *.xml *.log ../../../
    popd
}

### Test ###

RET=0

pre_test
run_test

exit ${RET}
