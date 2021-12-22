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
if [ ${USE_TENSORFLOW} == "1" ] && [ ${USE_PYTORCH} == "1" ] ; then
    echo " Unsupported test configuration. Only one of USE_TENSORFLOW and USE_PYTORCH can be set to 1."
    exit 0
elif [ ${USE_TENSORFLOW} == "1" ] ; then
    echo "Setting up enviroment with tensorflow 1"
    source ${TRITON_PATH}/python_backend/inferentia/scripts/setup.sh -t --tensorflow-version 1
elif [ ${USE_PYTORCH} == "1" ] ; then
    echo "Setting up enviroment with pytorch"
    source ${TRITON_PATH}/python_backend/inferentia/scripts/setup.sh -p
else 
    echo " Unsupported test configuration. USE_TENSORFLOW flag is: ${USE_TENSORFLOW} and USE_PYTORCH flag is: ${USE_PYTORCH}. Only one of them can be set to 1."
    exit 0
fi
echo "done setting up enviroment"

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi

CLIENT_LOG="./perf_analyzer.log"
PERF_ANALYZER=/opt/tritonserver/qa/clients/perf_analyzer

OUTPUT_NO_BATCH_JSONDATAFILE=${TEST_JSON_REPO}/validation_no_batch.json
OUTPUT_BATCHED_JSONDATAFILE=${TEST_JSON_REPO}/validation_batched.json
NON_ALIGNED_OUTPUT_NO_BATCH_JSONDATAFILE=${TEST_JSON_REPO}/non_aligned_validation_no_batch.json
NON_ALIGNED_OUTPUT_BATCHED_JSONDATAFILE=${TEST_JSON_REPO}/non_aligned_validation_batched.json
WRONG_OUTPUT_NO_BATCH_JSONDATAFILE=${TEST_JSON_REPO}/wrong_validation_no_batch.json
WRONG_OUTPUT_BATCHED_JSONDATAFILE=${TEST_JSON_REPO}/wrong_validation_batched.json

ERROR_STRING="error | Request count: 0 | : 0 infer/sec"

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_LOG="./inference_server.log"
source /opt/tritonserver/qa/common/util.sh
TEST_TYPES="single multiple"
BATCHED_FLAGS="_ _batched_"
DISABLE_DEFAULT_BATCHING_FLAGS="_default_batch _no_batch"
# Helper function for clearing out existing model directories
function clear_model_dir () {
    for DISABLE_DEFAULT_BATCHING_FLAG in ${DISABLE_DEFAULT_BATCHING_FLAGS}; do
        for BATCHED_FLAG in ${BATCHED_FLAGS}; do
            for TEST_TYPE in ${TEST_TYPES}; do
                DATADIR="${TRITON_PATH}/models_${TEST_TYPE}${BATCHED_FLAG}${TEST_FRAMEWORK}${DISABLE_DEFAULT_BATCHING_FLAG}"
                rm -rf DATADIR
            done
        done
    done
}
# Helper function for generating models
function create_inferentia_models () {
    for DISABLE_DEFAULT_BATCHING_FLAG in ${DISABLE_DEFAULT_BATCHING_FLAGS}; do
        for BATCHED_FLAG in ${BATCHED_FLAGS}; do
            for TEST_TYPE in ${TEST_TYPES}; do
                CURR_GEN_SCRIPT="${GEN_SCRIPT} --model_type ${MODEL_TYPE}  
                --triton_model_dir ${TRITON_PATH}/models_${TEST_TYPE}${BATCHED_FLAG}${TEST_FRAMEWORK}${DISABLE_DEFAULT_BATCHING_FLAG}/add-sub-1x4 
                --compiled_model ${COMPILED_MODEL}"
                if [ ${DISABLE_DEFAULT_BATCHING_FLAG} == "_no_batch" ]; then
                    CURR_GEN_SCRIPT="${CURR_GEN_SCRIPT} 
                    --disable_batch_requests_to_neuron"
                fi
                if [ ${BATCHED_FLAG} == "_batched_" ]; then
                    CURR_GEN_SCRIPT="${CURR_GEN_SCRIPT}
                    --triton_input INPUT__0,INT64,4 INPUT__1,INT64,4 
                    --triton_output OUTPUT__0,INT64,4 OUTPUT__1,INT64,4          
                    --enable_dynamic_batching 
                    --max_batch_size 1000 
                    --preferred_batch_size 8 
                    --max_queue_delay_microseconds 100"
                else
                    CURR_GEN_SCRIPT="${CURR_GEN_SCRIPT}
                    --triton_input INPUT__0,INT64,-1x4 INPUT__1,INT64,-1x4 
                    --triton_output OUTPUT__0,INT64,-1x4 OUTPUT__1,INT64,-1x4"
                fi
                if [ ${TEST_TYPE} == "single" ]; then
                    CURR_GEN_SCRIPT="${CURR_GEN_SCRIPT}   
                    --neuron_core_range 0:0"
                elif [ ${TEST_TYPE} == "multiple" ]; then
                    CURR_GEN_SCRIPT="${CURR_GEN_SCRIPT} 
                    --triton_model_instance_count 3 
                    --neuron_core_range 0:7"
                fi
                echo ${CURR_GEN_SCRIPT}
                eval ${CURR_GEN_SCRIPT}
            done
        done
    done
}

# Setup models
if [ ${USE_TENSORFLOW} == "1" ]; then
    TEST_FRAMEWORK="tf1"
    clear_model_dir
    python ${TEST_JSON_REPO}/simple_model.py \
        --name add_sub_model_tf1 \
        --model_type tensorflow \
        --tf_version 1 \
        --batch_size 1
    GEN_SCRIPT="python ${TRITON_PATH}/python_backend/inferentia/scripts/gen_triton_model.py"
    MODEL_TYPE="tensorflow"
    COMPILED_MODEL="${PWD}/add_sub_model_tf1"
    create_inferentia_models

elif [ ${USE_PYTORCH} == "1" ]; then
    TEST_FRAMEWORK="pyt"
    clear_model_dir
    python ${TEST_JSON_REPO}/simple_model.py \
        --name add_sub_model_pyt \
        --model_type pytorch \
        --batch_size 1
    GEN_SCRIPT="python ${TRITON_PATH}/python_backend/inferentia/scripts/gen_triton_model.py"
    MODEL_TYPE="pytorch"
    COMPILED_MODEL="$PWD/add_sub_model_pyt.pt"
    create_inferentia_models
fi


RET=0
for DISABLE_DEFAULT_BATCHING_FLAG in ${DISABLE_DEFAULT_BATCHING_FLAGS}; do
    for BATCHED_FLAG in ${BATCHED_FLAGS}; do
        for TEST_TYPE in $TEST_TYPES; do
            DATADIR="${TRITON_PATH}/models_${TEST_TYPE}${BATCHED_FLAG}${TEST_FRAMEWORK}${DISABLE_DEFAULT_BATCHING_FLAG}"
            SERVER_ARGS="--model-repository=${DATADIR} --log-verbose=1"
            PERF_ANALYZER_EXTRA_ARGS=""
            if [ ${BATCHED_FLAG} == "_batched_" ]; then
                PERF_ANALYZER_EXTRA_ARGS="-b 6"
                NON_ALIGNED_OUTPUT_JSONDATAFILE=${NON_ALIGNED_OUTPUT_BATCHED_JSONDATAFILE}
                WRONG_OUTPUT_JSONDATAFILE=${WRONG_OUTPUT_BATCHED_JSONDATAFILE}
                OUTPUT_JSONDATAFILE=${OUTPUT_BATCHED_JSONDATAFILE}
            else
                PERF_ANALYZER_EXTRA_ARGS=""
                NON_ALIGNED_OUTPUT_JSONDATAFILE=${NON_ALIGNED_OUTPUT_NO_BATCH_JSONDATAFILE}
                WRONG_OUTPUT_JSONDATAFILE=${WRONG_OUTPUT_NO_BATCH_JSONDATAFILE}
                OUTPUT_JSONDATAFILE=${OUTPUT_NO_BATCH_JSONDATAFILE}
            fi
            rm -f $SERVER_LOG $CLIENT_LOG

            run_server
            if [ "$SERVER_PID" == "0" ]; then
                echo -e "\n***\n*** Failed to start $SERVER\n***"
                cat $SERVER_LOG
                exit 1
            fi
            set +e
            $PERF_ANALYZER -v -m add-sub-1x4 --concurrency-range 1:10:4 --input-data=${NON_ALIGNED_OUTPUT_JSONDATAFILE} ${PERF_ANALYZER_EXTRA_ARGS} >$CLIENT_LOG 2>&1
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

            $PERF_ANALYZER -v -m add-sub-1x4 --concurrency-range 1:10:4 --input-data=${WRONG_OUTPUT_JSONDATAFILE} ${PERF_ANALYZER_EXTRA_ARGS} >$CLIENT_LOG 2>&1
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

            $PERF_ANALYZER -v -m add-sub-1x4 --concurrency-range 1:10:4 --input-data=${OUTPUT_JSONDATAFILE} ${PERF_ANALYZER_EXTRA_ARGS} >$CLIENT_LOG 2>&1
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
    done
done

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
