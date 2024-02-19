#!/bin/bash
# Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
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
TEST_RESULT_FILE='test_results.txt'
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

export CUDA_VISIBLE_DEVICES=0

CLIENT_LOG_BASE="./client"
INFER_TEST="../common/infer_test.py"
EXPECTED_NUM_TESTS="3"
TEST_RESULT_FILE='test_results.txt'

# S3 credentials are necessary for this test. Pass via ENV variables
aws configure set default.region $AWS_DEFAULT_REGION && \
    aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID && \
    aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY

# S3 bucket path (Point to bucket when testing cloud storage)
BUCKET_URL="s3://triton-bucket-${CI_JOB_ID}"

# Cleanup and delete S3 test bucket if it already exists (due to test failure)
aws s3 rm $BUCKET_URL --recursive --include "*" && \
    aws s3 rb $BUCKET_URL || true

# Make S3 test bucket
aws s3 mb "${BUCKET_URL}"

# Remove Slash in BUCKET_URL
BUCKET_URL=${BUCKET_URL%/}
BUCKET_URL_SLASH="${BUCKET_URL}/"

# Backup S3 credentials as they will be unset during the test
AWS_DEFAULT_REGION_BACKUP=$AWS_DEFAULT_REGION
AWS_ACCESS_KEY_ID_BACKUP=$AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY_BACKUP=$AWS_SECRET_ACCESS_KEY

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_TIMEOUT=600

SERVER_LOG_BASE="./inference_server"
source ../common/util.sh

rm -f $SERVER_LOG_BASE* $CLIENT_LOG_BASE*
RET=0

# Test 3 Scenarios:
# 1. Only AWS ENV vars (Without aws configure)
# 2. AWS ENV vars + dummy values in aws configure [ENV vars have higher priority]
# 3. Only AWS configured (Without AWS ENV vars)
for ENV_VAR in "env" "env_dummy" "config"; do
    SERVER_LOG=$SERVER_LOG_BASE.$ENV_VAR.log
    CLIENT_LOG=$CLIENT_LOG_BASE.$ENV_VAR.log

    if [ "$ENV_VAR" == "config" ]; then
        unset AWS_ACCESS_KEY_ID
        unset AWS_SECRET_ACCESS_KEY
        unset AWS_DEFAULT_REGION
    elif [ "$ENV_VAR" == "env_dummy" ]; then
        aws configure set default.region "dummy_region" && \
            aws configure set aws_access_key_id "dummy_id" && \
            aws configure set aws_secret_access_key "dummy_key"
    else
        rm ~/.aws/credentials && rm ~/.aws/config
    fi

    # Construct model repository

    KIND="KIND_GPU"

    # Test coverage for extra slashes
    for MAYBE_SLASH in "" "/" "//"; do

        ROOT_REPO="$BUCKET_URL$MAYBE_SLASH"
        MODEL_REPO="${BUCKET_URL}/${MAYBE_SLASH}models${MAYBE_SLASH}"

        # copy models in model directory
        rm -rf models && mkdir -p models

        # perform empty repo tests

        SERVER_ARGS="--model-repository=$ROOT_REPO --exit-timeout-secs=120"

        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        kill $SERVER_PID
        wait $SERVER_PID

        # run with a non-root empty model repo
        touch models/dummy
        if [ "$ENV_VAR" != "config" ]; then
            aws configure set default.region $AWS_DEFAULT_REGION && \
                aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID && \
                aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
        fi
        aws s3 cp . "$BUCKET_URL_SLASH" --recursive --include "*"
        if [ "$ENV_VAR" == "env_dummy" ]; then
            aws configure set default.region "dummy_region" && \
                aws configure set aws_access_key_id "dummy_id" && \
                aws configure set aws_secret_access_key "dummy_key"
        elif [ "$ENV_VAR" == "env" ]; then
            rm ~/.aws/credentials && rm ~/.aws/config
        fi

        SERVER_ARGS="--model-repository=$MODEL_REPO --exit-timeout-secs=120"

        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        kill $SERVER_PID
        wait $SERVER_PID

        if [ "$ENV_VAR" != "config" ]; then
            aws configure set default.region $AWS_DEFAULT_REGION && \
                aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID && \
                aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
        fi
        aws s3 rm "${BUCKET_URL_SLASH}" --recursive --include "*"
        rm models/dummy

        # Now start model tests

        for FW in graphdef savedmodel onnx libtorch plan; do
            cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/${FW}_float32_float32_float32/ models/
        done

        # Copy models with string inputs and remove nobatch (bs=1) models
        cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/*_object_object_object/ models/
        rm -rf models/*nobatch*

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

        # Perform test with model repository variants
        for src in "models/" "."  ; do

            # copy contents of /models into S3 bucket.
            aws s3 cp $src $BUCKET_URL_SLASH --recursive --include "*"
            if [ "$ENV_VAR" == "env_dummy" ]; then
                aws configure set default.region "dummy_region" && \
                    aws configure set aws_access_key_id "dummy_id" && \
                    aws configure set aws_secret_access_key "dummy_key"
            elif [ "$ENV_VAR" == "env" ]; then
                rm ~/.aws/credentials && rm ~/.aws/config
            fi

            if [ "$src" == "." ]; then
                # set server arguments
                SERVER_ARGS="--model-repository=$MODEL_REPO --exit-timeout-secs=120"
            else
                # set server arguments
                SERVER_ARGS="--model-repository=$ROOT_REPO --exit-timeout-secs=120"
            fi

            run_server
            if [ "$SERVER_PID" == "0" ]; then
                echo -e "\n***\n*** Failed to start $SERVER\n***"
                cat $SERVER_LOG
                exit 1
            fi

            set +e

            python $INFER_TEST >$CLIENT_LOG 2>&1
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                echo -e "\n***\n*** Test Failed\n***"
                RET=1
            else
                check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
                if [ $? -ne 0 ]; then
                    cat $CLIENT_LOG
                    echo -e "\n***\n*** Test Result Verification Failed\n***"
                    RET=1
                fi
            fi

            set -e

            kill $SERVER_PID
            wait $SERVER_PID

            # Clean up bucket
            if [ "$ENV_VAR" != "config" ]; then
                aws configure set default.region $AWS_DEFAULT_REGION && \
                    aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID && \
                    aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
            fi
            aws s3 rm "${BUCKET_URL_SLASH}" --recursive --include "*"
        done
    done
done

# Restore S3 credentials
rm ~/.aws/credentials && rm ~/.aws/config
export AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION_BACKUP
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID_BACKUP
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY_BACKUP
aws configure set default.region $AWS_DEFAULT_REGION && \
    aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID && \
    aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY

# Test with polling enabled
SERVER_ARGS="--model-repository=$ROOT_REPO --exit-timeout-secs=120 --model-control-mode=poll"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# copy contents of /models into S3 bucket and wait for them to be loaded.
aws s3 cp models/ "${BUCKET_URL_SLASH}" --recursive --include "*"
sleep 600

set +e

python $INFER_TEST >$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi


set -e

kill $SERVER_PID
wait $SERVER_PID

# Test localization to a specified location
export TRITON_AWS_MOUNT_DIRECTORY=`pwd`/aws_localization_test

if [ -d "$TRITON_AWS_MOUNT_DIRECTORY" ]; then
  rm -rf $TRITON_AWS_MOUNT_DIRECTORY
fi

mkdir -p $TRITON_AWS_MOUNT_DIRECTORY

SERVER_LOG=$SERVER_LOG_BASE.custom_localization.log
SERVER_ARGS="--model-repository=$ROOT_REPO --exit-timeout-secs=120"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

if [ -z "$(ls -A $TRITON_AWS_MOUNT_DIRECTORY)" ]; then
    echo -e "\n***\n*** Test localization to a specified location failed. \n***"
    echo -e "\n***\n*** Specified mount folder $TRITON_AWS_MOUNT_DIRECTORY is empty \n***"
    ls -A $TRITON_AWS_MOUNT_DIRECTORY
    exit 1
fi

kill $SERVER_PID
wait $SERVER_PID

if [ -d "$TRITON_AWS_MOUNT_DIRECTORY" ] && [ ! -z "$(ls -A $TRITON_AWS_MOUNT_DIRECTORY)" ]; then
    echo -e "\n***\n*** Test localization to a specified location failed. \n***"
    echo -e "\n***\n*** Specified mount folder $TRITON_AWS_MOUNT_DIRECTORY was not cleared properly. \n***"
    ls -A $TRITON_AWS_MOUNT_DIRECTORY
    exit 1
fi

rm -rf $TRITON_AWS_MOUNT_DIRECTORY
unset TRITON_AWS_MOUNT_DIRECTORY

# Save models for AWS_SESSION_TOKEN test
rm -rf tmp_cred_test_models
mv models tmp_cred_test_models
# Clean up bucket contents
aws s3 rm "${BUCKET_URL_SLASH}" --recursive --include "*"

# Test reload of model with explicit model control
rm -rf models && mkdir -p models/libtorch_float32_float32_float32 && \
    cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/libtorch_float32_float32_float32/1 models/libtorch_float32_float32_float32/. && \
    cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/libtorch_float32_float32_float32/config.pbtxt models/libtorch_float32_float32_float32/.
    cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/libtorch_float32_float32_float32/output0_labels.txt models/libtorch_float32_float32_float32/.

# Remove version policy from config.pbtxt
sed -i '/^version_policy/d' models/libtorch_float32_float32_float32/config.pbtxt

# Copy contents of models into S3 bucket
aws s3 cp models/ "${BUCKET_URL_SLASH}" --recursive --include "*"

SERVER_ARGS="--model-repository=$BUCKET_URL --exit-timeout-secs=120 --model-control-mode=explicit"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

curl -X POST localhost:8000/v2/repository/models/libtorch_float32_float32_float32/load

CURL_LOG=$(curl -X POST localhost:8000/v2/repository/index)

if [ "$CURL_LOG" != "[{\"name\":\"libtorch_float32_float32_float32\",\"version\":\"1\",\"state\":\"READY\"}]" ]; then
    RET=1
fi

# Add new model version
aws s3 cp /data/inferenceserver/${REPO_VERSION}/qa_model_repository/libtorch_float32_float32_float32/3 "${BUCKET_URL_SLASH}libtorch_float32_float32_float32/3" --recursive --include "*"

curl -X POST localhost:8000/v2/repository/models/libtorch_float32_float32_float32/load

CURL_LOG=$(curl -X POST localhost:8000/v2/repository/index)
if [ "$CURL_LOG" != "[{\"name\":\"libtorch_float32_float32_float32\",\"version\":\"1\",\"state\":\"UNAVAILABLE\",\"reason\":\"unloaded\"},{\"name\":\"libtorch_float32_float32_float32\",\"version\":\"3\",\"state\":\"READY\"}]" ]; then
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

# Clean up bucket contents
aws s3 rm "${BUCKET_URL_SLASH}" --recursive --include "*"

# Test with temporary credential (AWS_SESSION_TOKEN)
AWS_GET_SESSION_TOKEN_RES=`aws sts get-session-token --duration-seconds 900` && \
    export AWS_ACCESS_KEY_ID=`echo $AWS_GET_SESSION_TOKEN_RES | jq -r ".Credentials.AccessKeyId"` && \
    export AWS_SECRET_ACCESS_KEY=`echo $AWS_GET_SESSION_TOKEN_RES | jq -r ".Credentials.SecretAccessKey"` && \
    export AWS_SESSION_TOKEN=`echo $AWS_GET_SESSION_TOKEN_RES | jq -r ".Credentials.SessionToken"`
rm ~/.aws/credentials && rm ~/.aws/config
aws configure set default.region $AWS_DEFAULT_REGION && \
    aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID && \
    aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY && \
    aws configure set aws_session_token $AWS_SESSION_TOKEN

# Copy models into S3 bucket
aws s3 cp tmp_cred_test_models/ "${BUCKET_URL_SLASH}" --recursive --include "*"

SERVER_LOG=$SERVER_LOG_BASE.temporary_credentials_test.log
SERVER_ARGS="--model-repository=$BUCKET_URL --exit-timeout-secs=120"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

python $INFER_TEST >$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

# Test access decline
export AWS_SECRET_ACCESS_KEY="[Invalid]" && export AWS_SESSION_TOKEN=""
SERVER_LOG=$SERVER_LOG_BASE.access_decline_test.log
SERVER_ARGS="--model-repository=$BUCKET_URL --exit-timeout-secs=120"
run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "\n***\n*** Unexpected server start $SERVER\n***"
    cat $SERVER_LOG
    kill $SERVER_PID
    wait $SERVER_PID
    RET=1
else
  # AWS S3 does not appear to reply on access decline, but other implementations
  # might provide extra messages, so make sure Triton will print the messages.
  EXPECTED_MSG="Unable to create S3 filesystem client. Check account credentials. Exception: '' Message: 'No response body.'"
  if ! grep "$EXPECTED_MSG" $SERVER_LOG; then
    echo -e "\n***\n*** Expected error message not found\n***"
    cat $SERVER_LOG
    RET=1
  fi
fi

# Restore S3 credentials
rm ~/.aws/credentials && rm ~/.aws/config
export AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION_BACKUP
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID_BACKUP
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY_BACKUP
aws configure set default.region $AWS_DEFAULT_REGION && \
    aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID && \
    aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY

# Clean up bucket contents
aws s3 rm "${BUCKET_URL_SLASH}" --recursive --include "*"

# Test case where S3 folder has >1000 files
rm -rf models

mkdir -p models/model/1
# Create Python model that reads the number of files in the
# model directory when loaded
echo "import os

class TritonPythonModel:

    def initialize(self, args):
        count = 0
        model_dir = args['model_repository']
        for path in os.listdir(model_dir):
            if os.path.isfile(os.path.join(model_dir, path)):
                count += 1
        print('Found {} files in model directory'.format(count))

    def execute(self):
        pass" > models/model/1/model.py

for i in {1..1050}; do
    touch models/model/0${i}.txt
done

# Provide extended timeout to allow >1000 files to be loaded
SERVER_ARGS="--model-repository=$BUCKET_URL --exit-timeout-secs=600 --model-control-mode=none"
SERVER_LOG=$SERVER_LOG_BASE.many_files.log

# copy contents of /models into S3 bucket and wait for them to be loaded.
aws s3 cp models/ "${BUCKET_URL_SLASH}" --recursive --include "*"

# Test that the server starts up. Files will be loaded in numerically
# ascending order, so the model file is loaded after the first 1000
# files. If AWS fails to load >1000 files, the model file will not
# be loaded and the server will fail to start.

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

kill $SERVER_PID
wait $SERVER_PID

# Confirm the correct number of files loaded
EXPECTED_MSG="Found 1050 files in model directory"
if ! grep "$EXPECTED_MSG" $SERVER_LOG; then
echo -e "\n***\n*** Expected file count message not found\n***"
cat $SERVER_LOG
RET=1
fi

# Clean up bucket contents and delete bucket
aws s3 rm "${BUCKET_URL_SLASH}" --recursive --include "*"
aws s3 rb "${BUCKET_URL}"

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
