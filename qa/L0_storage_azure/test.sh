#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

if [ -z "$AZURE_STORAGE_ACCOUNT" ]; then
    echo -e "azure storage account must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

if [ -z "$AZURE_STORAGE_KEY" ]; then
    echo -e "azure storage key must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

ACCOUNT_NAME=$AZURE_STORAGE_ACCOUNT
ACCOUNT_KEY=$AZURE_STORAGE_KEY
export CUDA_VISIBLE_DEVICES=0
CLIENT_LOG_BASE="./client"
INFER_TEST=infer_test.py
EXPECTED_NUM_TESTS="3"
timestamp=$(date +%s)
CONTAINER_NAME="tritonqatest${timestamp}"

# container path (Point to the container when testing cloud storage)
AS_URL="as://${ACCOUNT_NAME}/${CONTAINER_NAME}"

# create test container
az storage container create --name ${CONTAINER_NAME} --account-name ${ACCOUNT_NAME} --account-key ${ACCOUNT_KEY}
sleep 10

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_TIMEOUT=420
SERVER_LOG_BASE="./inference_server"
source ../common/util.sh

rm -f $SERVER_LOG_BASE* $CLIENT_LOG_BASE*
RET=0

# Construct model repository
mkdir -p models
for FW in graphdef savedmodel onnx libtorch plan; do
    cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/${FW}_float32_float32_float32 models/
done

# Copy models with string inputs and remove nobatch (bs=1) models
cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/*_object_object_object models/

rm -rf models/*nobatch*

KIND="KIND_GPU"
for FW in graphdef savedmodel onnx libtorch plan; do
    for MC in `ls models/${FW}*/config.pbtxt`; do
        echo "instance_group [ { kind: ${KIND} }]" >> $MC
    done
done

# now traverse the tree and create empty version directories that the CLI skips
for dir in `ls models/`; do
    for subdir in `ls models/$dir`; do
        if [ -d models/$dir/$subdir ] && [ -z "$(ls models/$dir/$subdir)" ]; then
            touch models/$dir/$subdir/$subdir
        fi
    done
done

# copy contents of /models into container.
for file in `find models -type f` ;do
    az storage blob upload --container-name ${CONTAINER_NAME} --account-name ${ACCOUNT_NAME} --account-key ${ACCOUNT_KEY} --file $file --name $file
done
sleep 10

# Test 1 Scenarios:
# 1. access blob using shared key in envs
# 2. adding more scenarios in future 
for ENV_VAR in "shared_key"; do
    SERVER_LOG=$SERVER_LOG_BASE.$ENV_VAR.log
    CLIENT_LOG=$CLIENT_LOG_BASE.$ENV_VAR.log
    MODEL_REPO="${AS_URL}/models"
    if [ "$ENV_VAR" == "sas" ]; then
        unset AZURE_STORAGE_KEY
        sas=`az storage blob generate-sas --container-name ${CONTAINER_NAME} --account-name ${ACCOUNT_NAME} --account-key ${ACCOUNT_KEY} --name models`
        sas_without_quote=$(eval echo $sas)
        export AZURE_STORAGE_SAS="?$sas_without_quote"
    fi

    # Now start model tests
    # set server arguments
    SERVER_ARGS="--model-repository=$MODEL_REPO --exit-timeout-secs=120"  

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        RET=1
        break 
    fi

    set +e

    python $INFER_TEST >$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        check_test_results $CLIENT_LOG $EXPECTED_NUM_TESTS
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi

    set -e

    kill $SERVER_PID
    wait $SERVER_PID
done
 
# Clean up container
az storage container delete --name ${CONTAINER_NAME} --account-name ${ACCOUNT_NAME} --account-key ${ACCOUNT_KEY}

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
fi

exit $RET
