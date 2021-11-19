#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# First need to set up enviroment
source ${TRITON_PATH}/python_backend/inferentia/scripts/setup.sh -p
echo "done setting up enviroment"

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi

CLIENT_LOG="./perf_analyzer.log"
PERF_ANALYZER=/opt/tritonserver/qa/clients/perf_analyzer

OUTPUT_JSONDATAFILE=${TEST_JSON_REPO}/validation.json
NON_ALIGNED_OUTPUT_JSONDATAFILE=${TEST_JSON_REPO}/non_aligned_validation.json
WRONG_OUTPUT_JSONDATAFILE=${TEST_JSON_REPO}/wrong_validation.json

ERROR_STRING="error | Request count: 0 | : 0 infer/sec"

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_LOG="./inference_server.log"
source /opt/tritonserver/qa/common/util.sh
TEST_TYPES="single multiple"

# Setup models
for TEST_TYPE in $TEST_TYPES; do
    DATADIR="${TRITON_PATH}/models_${TEST_TYPE}"
    rm -rf DATADIR
done
cd ${TRITON_PATH}
python ${TEST_JSON_REPO}/simple-model.py
python ${TRITON_PATH}/python_backend/inferentia/scripts/gen_triton_model.py \
    --model_type "pytorch" \
    --triton_input INPUT__0,INT64,4 INPUT__1,INT64,4 \
    --triton_output OUTPUT__0,INT64,4 OUTPUT__1,INT64,4 \
    --compiled_model $PWD/add_sub_model.pt \
    --triton_model_dir models_single/add-sub-1x4 --neuron_core_range 0:0

cd ${TRITON_PATH}
python ${TEST_JSON_REPO}/simple-model.py
python ${TRITON_PATH}/python_backend/inferentia/scripts/gen_triton_model.py \
    --model_type "pytorch" \
    --triton_input INPUT__0,INT64,4 INPUT__1,INT64,4 \
    --triton_output OUTPUT__0,INT64,4 OUTPUT__1,INT64,4 \
    --compiled_model $PWD/add_sub_model.pt \
    --triton_model_dir models_multiple/add-sub-1x4 \
    --triton_model_instance_count 3 --neuron_core_range 0:7

RET=0

for TEST_TYPE in $TEST_TYPES; do
    DATADIR="${TRITON_PATH}/models_${TEST_TYPE}"
    SERVER_ARGS="--model-repository=${DATADIR} --log-verbose=1"
    rm -f $SERVER_LOG $CLIENT_LOG

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi
    set +e
    $PERF_ANALYZER -v -m add-sub-1x4 --input-data=${NON_ALIGNED_OUTPUT_JSONDATAFILE} >$CLIENT_LOG 2>&1
    if [ $? -eq 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    if [ $(cat $CLIENT_LOG |  grep "The 'validation_data' field doesn't align with 'data' field in the json file" | wc -l) -eq 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi

    $PERF_ANALYZER -v -m add-sub-1x4 --input-data=${WRONG_OUTPUT_JSONDATAFILE} >$CLIENT_LOG 2>&1
    if [ $? -eq 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    if [ $(cat $CLIENT_LOG |  grep "Output doesn't match expected output" | wc -l) -eq 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi

    $PERF_ANALYZER -v -m add-sub-1x4 --input-data=${OUTPUT_JSONDATAFILE} >$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    set -e
    kill_server
done

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
