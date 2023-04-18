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

export CUDA_VISIBLE_DEVICES=0
SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

# Create TensorRT model with invalid plan file
rm -rf models && mkdir models
mkdir models/invalid_plan_file && (cd models/invalid_plan_file && \
    echo -e "name: \"invalid_plan_file\"" >> config.pbtxt && \
    echo -e "platform: \"tensorrt_plan\"" >> config.pbtxt && \
    echo -e "input [\n {\n name: \"INPUT\"\n data_type: TYPE_FP32\n dims: [-1]\n }\n ]" >> config.pbtxt && \
    echo -e "output [\n {\n name: \"OUTPUT\"\n data_type: TYPE_FP32\n dims: [-1]\n }\n ]" >> config.pbtxt && \
    mkdir 1 && echo "----- invalid model.plan -----" >> 1/model.plan)

# Test with and without auto complete enabled
for ENABLE_AUTOCOMPLETE in "YES" "NO"; do

    if [[ "$ENABLE_AUTOCOMPLETE" == "YES" ]]; then
        TEST_NAME="test_invalid_trt_model_autocomplete"
        SERVER_ARGS="--model-repository=models --model-control-mode=explicit"
    else
        TEST_NAME="test_invalid_trt_model"
        SERVER_ARGS="--model-repository=models --model-control-mode=explicit --disable-auto-complete-config"
    fi

    SERVER_LOG="./$TEST_NAME.server.log"
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    RET=0

    set +e
    python trt_error_propagation_test.py TestTrtErrorPropagation.$TEST_NAME > $TEST_NAME.log 2>&1
    if [ $? -ne 0 ]; then
        cat $TEST_NAME.log
        echo -e "\n***\n*** Test FAILED\n***"
        RET=1
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID

    if [ $RET -ne 0 ]; then
        exit $RET
    fi

done

# Exit with success
echo -e "\n***\n*** Test Passed\n***"
exit 0
