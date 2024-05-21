#!/bin/bash
# Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Parses default-max-batch-size log record
#
# Example log record:
# I0521 02:12:37.402353 161 backend_model.cc:503] "Adding default backend config setting: default-max-batch-size,4
parse_default_max_batch_size() {
    echo $(python3 -c "print('$1'.split(',')[1].strip('\"'))")
}

# Returns backend configuration json
# message from server log file path
#
# Example: config_map = $(get_config_map server.log)
get_config_map() {
    BACKEND_CONFIG_MAP=$(grep "backend configuration:" $1)
    echo $(python3 -c "backend_config='$BACKEND_CONFIG_MAP'.split('] \"backend configuration:\n')[1].rstrip('\"');print(backend_config)")
}

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi
if [ ! -z "$TEST_REPO_ARCH" ]; then
    REPO_VERSION=${REPO_VERSION}_${TEST_REPO_ARCH}
fi

rm -rf ./models/
mkdir -p ./models/no_config
cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/savedmodel_float32_float32_float32/1 ./models/no_config/


SERVER=/opt/tritonserver/bin/tritonserver
SERVER_TIMEOUT=20
source ../common/util.sh

SERVER_LOG_BASE="./inference_server"
rm -f $SERVER_LOG_BASE*
rm -f *.out

COMMON_ARGS="--model-repository=`pwd`/models --strict-model-config=false --log-verbose=1 "

NEGATIVE_PARSE_ARGS=("--backend-config=,default-max-batch-size=3 $COMMON_ARGS" \
                    "--backend-config=default-max-batch-size= $COMMON_ARGS" \
                    "--backend-config=default-max-batch-size $COMMON_ARGS" \
                    "--backend-config=tensorflow,default-max-batch-size= $COMMON_ARGS" \
                    "--backend-config=tensorflow,default-max-batch-size $COMMON_ARGS" \
)

POSITIVE_DEFAULT_ARGS=$COMMON_ARGS
POSITIVE_TEST_ARGS=("--backend-config=tensorflow,default-max-batch-size=5 $COMMON_ARGS" \
                    "--backend-config=default-max-batch-size=6 $COMMON_ARGS" \
                    "--backend-config=default-max-batch-size=7 --backend-config=tensorflow,default-max-batch-size=8 $COMMON_ARGS" \
)

# These integers correspond to the expected default-max-batch-size which gets set
# in the POSITIVE_TEST_ARGS
POSITIVE_TEST_ANSWERS=(5 6 8)

RET=0
# Positive tests
SERVER_ARGS=$POSITIVE_DEFAULT_ARGS
SERVER_LOG=$SERVER_LOG_BASE.backend_config_positive_default.log
run_server

if [ "$SERVER_PID" == "0" ]; then
    echo -e "*** FAILED: Server failed to start $SERVER\n"
    RET=1

else
    kill $SERVER_PID
    wait $SERVER_PID

    RESULT_LOG_LINE=$(grep -a "Adding default backend config setting:" $SERVER_LOG)
    if [ "$RESULT_LOG_LINE" != "" ]; then

        # Pick out the logged value of the default-max-batch-size which gets passed into model creation
        RESOLVED_DEFAULT_MAX_BATCH_SIZE=$(parse_default_max_batch_size "${RESULT_LOG_LINE}")

        if [ "$RESOLVED_DEFAULT_MAX_BATCH_SIZE" != "4" ]; then
            echo "*** FAILED: Found default-max-batch-size not equal to the expected default-max-batch-size. Expected: default-max-batch-size,4, Found: $RESOLVED_DEFAULT_MAX_BATCH_SIZE \n"
            RET=1
        fi
    else
        echo "*** FAILED: No log statement stating default max batch size\n"
        RET=1
    fi
fi

for ((i=0; i < ${#POSITIVE_TEST_ARGS[@]}; i++)); do
    SERVER_ARGS=${POSITIVE_TEST_ARGS[$i]}
    SERVER_LOG=$SERVER_LOG_BASE.backend_config_positive_$i.log
    run_server

    if [ "$SERVER_PID" == "0" ]; then
        echo -e "*** FAILED: Server failed to start $SERVER\n"
        RET=1

    else
        kill $SERVER_PID
        wait $SERVER_PID

        RESULT_LOG_LINE=$(grep -a "Found overwritten default setting:" $SERVER_LOG)
        if [ "$RESULT_LOG_LINE" != "" ]; then

            # Pick out the logged value of the default-max-batch-size which gets passed into model creation
            RESOLVED_DEFAULT_MAX_BATCH_SIZE=$(parse_default_max_batch_size "${RESULT_LOG_LINE}")

            if [ "$RESOLVED_DEFAULT_MAX_BATCH_SIZE" != "${POSITIVE_TEST_ANSWERS[$i]}" ]; then
                echo "*** FAILED: Found default-max-batch-size not equal to the expected default-max-batch-size. Expected: ${POSITIVE_TEST_ANSWERS[$i]}, Found: $RESOLVED_DEFAULT_MAX_BATCH_SIZE \n"
                RET=1
            fi
        else
            echo "*** FAILED: No log statement stating default max batch size\n"
            RET=1
        fi
    fi
done

# Negative tests
# Failing because the syntax is incorrect
for ((i=0; i < ${#NEGATIVE_PARSE_ARGS[@]}; i++)); do
    SERVER_ARGS=${NEGATIVE_PARSE_ARGS[$i]}
    SERVER_LOG=$SERVER_LOG_BASE.backend_config_negative_parse$i.log
    run_server

    if [ "$SERVER_PID" == "0" ]; then
        if ! grep -e "--backend-config option format is" $SERVER_LOG; then
            echo -e "*** FAILED: Expected invalid backend config parse message but found other error.\n"
            RET=1
        fi
    else
        echo -e "*** FAILED: Expected server to exit with error, but found running.\n"
        RET=1
        kill $SERVER_PID
        wait $SERVER_PID
    fi
done


#
# Specific backend tests
#

# While inference server is running, save the
# config of the 'no_config' model to the TRIAL
# file.
function save_model_config() {
    CODE=`curl -s -w %{http_code} -o ./$TRIAL.out localhost:8000/v2/models/no_config/config`
    set -e
    if [ "$CODE" != "200" ]; then
        cat $TRIAL.out
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
}

# Tensorflow 1: Batching ON
rm -rf ./models/
mkdir -p ./models/no_config
cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/savedmodel_float32_float32_float32/1 ./models/no_config/

SERVER_ARGS="--backend-config=tensorflow,default-max-batch-size=5 $COMMON_ARGS"
SERVER_LOG=$SERVER_LOG_BASE.backend_config_tensorflow_batch_5.log
run_server

TRIAL=tensorflow_batching_on
if [ "$SERVER_PID" == "0" ]; then
    echo -e "*** FAILED: Server failed to start $SERVER\n"
    RET=1
else
    save_model_config

    # Assert the max-batch-size is the command line value
    MAX_BATCH_LOG_LINE=$(grep -a "\"max_batch_size\":5" $TRIAL.out)
    if [ "$MAX_BATCH_LOG_LINE" == "" ]; then
        cat $TRIAL.out
        echo "*** FAILED: Expected max batch size to be 5 but found: $MAX_BATCH_LOG_LINE\n"
        RET=1
    fi

    # Assert we are also turning on the dynamic_batcher
    DYNAMIC_BATCHING_LOG_LINE=$(grep -a "Starting dynamic-batcher thread" $SERVER_LOG)
    if [ "$DYNAMIC_BATCHING_LOG_LINE" == "" ]; then
        echo "*** FAILED: Expected dynamic batching to be set in model config but was not found\n"
        RET=1
    fi

    kill $SERVER_PID
    wait $SERVER_PID

fi

# Tensorflow 1: Batching OFF
SERVER_ARGS="--backend-config=tensorflow,default-max-batch-size=0 $COMMON_ARGS"
SERVER_LOG=$SERVER_LOG_BASE.backend_config_tensorflow_batch_0.log
run_server

TRIAL=tensorflow_batching_off
if [ "$SERVER_PID" == "0" ]; then
    echo -e "*** FAILED: Server failed to start $SERVER\n"
    RET=1

else
    save_model_config

    # Assert the max-batch-size is 0 in the case batching is supported
    # in the model but not in the config.
    MAX_BATCH_LOG_LINE=$(grep -a "\"max_batch_size\":0" $TRIAL.out)
    if [ "$MAX_BATCH_LOG_LINE" == "" ]; then
        echo "*** FAILED: Expected max batch size to be 0 but found: $MAX_BATCH_LOG_LINE\n"
        RET=1
    fi

    # Assert batching disabled
    if [ "$(grep -a -E '\"dynamic_batching\": \{}' $SERVER_LOG)" != "" ]; then
        echo "*** FAILED: Found dynamic batching enabled in configuration when none expected.\n"
        RET=1
    fi

    kill $SERVER_PID
    wait $SERVER_PID

fi

# Onnxruntime: Batching ON
rm -rf ./models/
mkdir -p ./models/no_config
cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/onnx_float32_float32_float32/1 ./models/no_config/

SERVER_ARGS="--backend-config=onnxruntime,default-max-batch-size=5 $COMMON_ARGS"
SERVER_LOG=$SERVER_LOG_BASE.backend_config_onnxruntime_batch_5.log
run_server

TRIAL=onnxruntime_batching_on
if [ "$SERVER_PID" == "0" ]; then
    echo -e "*** FAILED: Server failed to start $SERVER\n"
    RET=1

else
    save_model_config

    # Assert the max-batch-size is the command line value
    MAX_BATCH_LOG_LINE=$(grep -a "\"max_batch_size\":5" $TRIAL.out)
    if [ "$MAX_BATCH_LOG_LINE" == "" ]; then
        echo "*** FAILED: Expected max batch size to be 5 but found: $MAX_BATCH_LOG_LINE\n"
        RET=1
    fi

    # Assert we are also turning on the dynamic_batcher
    DYNAMIC_BATCHING_LOG_LINE=$(grep -a "Starting dynamic-batcher thread" $SERVER_LOG)
    if [ "$DYNAMIC_BATCHING_LOG_LINE" == "" ]; then
        echo "*** FAILED: Expected dynamic batching to be set in model config but was not found\n"
        RET=1
    fi

    kill $SERVER_PID
    wait $SERVER_PID
fi

# Onnxruntime: Batching OFF
rm -rf ./models/
mkdir -p ./models/no_config
cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/onnx_float32_float32_float32/1 ./models/no_config/

SERVER_ARGS="--backend-config=onnxruntime,default-max-batch-size=0 $COMMON_ARGS"
SERVER_LOG=$SERVER_LOG_BASE.backend_config_onnxruntime_batch_0.log
run_server

TRIAL=onnxruntime_batching_off
if [ "$SERVER_PID" == "0" ]; then
    echo -e "*** FAILED: Server failed to start $SERVER\n"
    RET=1

else
    save_model_config

    # Assert the max-batch-size is 0 in the case batching is supported
    # in the model but not in the config.
    MAX_BATCH_LOG_LINE=$(grep -a "\"max_batch_size\":0" $TRIAL.out)
    if [ "$MAX_BATCH_LOG_LINE" == "" ]; then
        echo "*** FAILED: Expected max batch size to be 0 but found: $MAX_BATCH_LOG_LINE\n"
        RET=1
    fi

    # Assert batching disabled
    if [ "$(grep -a -E '\"dynamic_batching\": \{}' $SERVER_LOG)" != "" ]; then
        echo "*** FAILED: Found dynamic batching in configuration when none expected.\n"
        RET=1
    fi

    kill $SERVER_PID
    wait $SERVER_PID

fi

#
# General backend tests
#

# We want to make sure that backend configurations
# are not lost. For this purpose we are using only onnx backend

rm -rf ./models/
mkdir -p ./models/no_config/
cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/onnx_float32_float32_float32/1 ./models/no_config/

# First getting a baseline for the number of default configs
# added during a server set up
SERVER_ARGS="$COMMON_ARGS"
SERVER_LOG=$SERVER_LOG_BASE.default_configs.log
run_server

if [ "$SERVER_PID" == "0" ]; then
    echo -e "*** FAILED: Server failed to start $SERVER\n"
    RET=1

else
    # Count number of default configs
    BACKEND_CONFIG_MAP=$(get_config_map $SERVER_LOG)
    DEFAULT_CONFIG_COUNT=$(echo $BACKEND_CONFIG_MAP | jq -r | jq '.["cmdline"]' | jq length)
    if [ $DEFAULT_CONFIG_COUNT -lt 4 ]; then
        echo "*** FAILED: Expected number of default configs to be at least 4 but found: $DEFAULT_CONFIG_COUNT\n"
        RET=1
    fi

    kill $SERVER_PID
    wait $SERVER_PID

fi

# Now make sure that when setting specific backend configs
# default ones are not lost.
# Current logic for backend config resolution reads default configs first,
# then specific configs and overrides defaults if needed.
# We would like to make sure that none of configs are lost and
# defaults are properly overridden.
# One of defaultconfigs is `min-compute-capability`. This test
# checks if it is properlly overridden.
MIN_COMPUTE_CAPABILITY=XX
SERVER_ARGS="--backend-config=onnxruntime,min-compute-capability=$MIN_COMPUTE_CAPABILITY $COMMON_ARGS"
SERVER_LOG=$SERVER_LOG_BASE.global_configs.log
run_server

if [ "$SERVER_PID" == "0" ]; then
    echo -e "*** FAILED: Server failed to start $SERVER\n"
    RET=1

else
    # Count number of default configs
    BACKEND_CONFIG_MAP=$(get_config_map $SERVER_LOG)
    CONFIG_VALUE=$(echo $BACKEND_CONFIG_MAP | jq -r | jq '.["cmdline"]' | jq -r '.["min-compute-capability"]')

    if [ $CONFIG_VALUE != $MIN_COMPUTE_CAPABILITY ]; then
        echo "*** FAILED: Expected min-compute-capability config to be $MIN_COMPUTE_CAPABILITY but found: $CONFIG_VALUE\n"
        RET=1
    fi

    kill $SERVER_PID
    wait $SERVER_PID

fi
# Now make sure that specific backend configs are not lost.
SERVER_ARGS="--backend-config=onnxruntime,a=0 --backend-config=onnxruntime,y=0 --backend-config=onnxruntime,z=0 $COMMON_ARGS"
SERVER_LOG=$SERVER_LOG_BASE.specific_configs.log
EXPECTED_CONFIG_COUNT=$(($DEFAULT_CONFIG_COUNT+3))
run_server

if [ "$SERVER_PID" == "0" ]; then
    echo -e "*** FAILED: Server failed to start $SERVER\n"
    RET=1

else
    # Count number of default configs
    BACKEND_CONFIG_MAP=$(get_config_map $SERVER_LOG)
    TOTAL_CONFIG_COUNT=$(echo $BACKEND_CONFIG_MAP | jq -r | jq '.["cmdline"]' | jq 'length')

    if [ $TOTAL_CONFIG_COUNT -ne $EXPECTED_CONFIG_COUNT ]; then
        echo "*** FAILED: Expected number of backend configs to be $EXPECTED_CONFIG_COUNT but found: $TOTAL_CONFIG_COUNT\n"
        RET=1
    fi

    kill $SERVER_PID
    wait $SERVER_PID

fi


# Print test outcome
if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET

