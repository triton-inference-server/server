#!/bin/bash
# Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

export CUDA_VISIBLE_DEVICES=0

CLIENT_LOG_BASE="./client"
INFER_TEST=infer_test.py
EXPECTED_NUM_TESTS="3"

# GCS credentials are necessary for this test. Pass via ENV variables
export GOOGLE_APPLICATION_CREDENTIALS="file.json"

echo '{"type": "service_account",' >> $GOOGLE_APPLICATION_CREDENTIALS
echo '"project_id": "triton-285001",' >> $GOOGLE_APPLICATION_CREDENTIALS
echo -e "\"private_key_id\": \"$PROJECT_KEY_ID\"," >> $GOOGLE_APPLICATION_CREDENTIALS
echo -e "\"private_key\": \"$PROJECT_KEY\"," >> $GOOGLE_APPLICATION_CREDENTIALS
echo '"client_email": "hemantj@triton-285001.iam.gserviceaccount.com",' >> $GOOGLE_APPLICATION_CREDENTIALS
echo '"client_id": "106901301872481149333",' >> $GOOGLE_APPLICATION_CREDENTIALS
echo '"auth_uri": "https://accounts.google.com/o/oauth2/auth",' >> $GOOGLE_APPLICATION_CREDENTIALS
echo '"token_uri": "https://oauth2.googleapis.com/token",' >> $GOOGLE_APPLICATION_CREDENTIALS
echo '"auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",' >> $GOOGLE_APPLICATION_CREDENTIALS
echo '"client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/hemantj%40triton-285001.iam.gserviceaccount.com"}' >> $GOOGLE_APPLICATION_CREDENTIALS 

# Create .boto file (Needed to configure gsutil/google-cloud-sdk)
echo '[Credentials]' >> /root/.boto
echo -e 'gs_service_key_file = '$PWD'/'$GOOGLE_APPLICATION_CREDENTIALS >> /root/.boto
echo '[Boto]' >> /root/.boto
echo 'https_validate_certificates = True' >> /root/.boto
echo '[GSUtil]' >> /root/.boto
echo 'content_language = en' >> /root/.boto
echo 'default_api_version = 2' >> /root/.boto
echo 'default_project_id = triton-285001' >> /root/.boto

BUCKET_URL="gs://triton-bucket-${CI_PIPELINE_ID}"

# Make test bucket
gsutil mb "${BUCKET_URL}"

# Remove Slash in BUCKET_URL
BUCKET_URL=${BUCKET_URL%/}
BUCKET_URL_SLASH="${BUCKET_URL}/"

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_TIMEOUT=360

SERVER_LOG_BASE="./inference_server"
source ../common/util.sh

SERVER_LOG=$SERVER_LOG_BASE.log
CLIENT_LOG=$CLIENT_LOG_BASE.log
    
rm -f $SERVER_LOG_BASE* $CLIENT_LOG_BASE*

RET=0

# Construct model repository

KIND="KIND_GPU"

for MAYBE_SLASH in "" "/"; do

    ROOT_REPO="$BUCKET_URL$MAYBE_SLASH"
    MODEL_REPO="${BUCKET_URL_SLASH}models${MAYBE_SLASH}"

    # copy models in model directory
    rm -rf models && mkdir -p models

    # perform empty repo tests

    SERVER_ARGS="--model-repository=$ROOT_REPO --exit-timeout-secs=120 "

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
    gsutil cp -r models/ "$BUCKET_URL_SLASH"

    SERVER_ARGS="--model-repository=$MODEL_REPO --exit-timeout-secs=120 "

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    kill $SERVER_PID
    wait $SERVER_PID

    gsutil -m rm "${BUCKET_URL_SLASH}**"
    rm models/dummy

    # Now start model tests

    for FW in graphdef savedmodel netdef onnx libtorch plan; do
        cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/${FW}_float32_float32_float32/ models/
    done

    # Copy models with string inputs and remove nobatch (bs=1) models
    cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/*_object_object_object/ models/
    rm -rf models/*nobatch*

    for FW in graphdef savedmodel netdef onnx libtorch plan; do
        for MC in `ls models/${FW}*/config.pbtxt`; do
            echo "instance_group [ { kind: ${KIND} }]" >> $MC
        done
    done

    # now traverse the tree and create empty version directories that gsutil skips
    for dir in `ls models/`; do
        for subdir in `ls models/$dir`; do
            if [ -d models/$dir/$subdir ] && [ -z "$(ls models/$dir/$subdir)" ]; then
                touch models/$dir/$subdir/$subdir
            fi
        done
    done

    # Perform test with model repository variants
    for repo in "models/**" "models" ; do

        # copy contents of /models into GCS bucket.
        gsutil -m cp -r $repo $BUCKET_URL_SLASH

        if [ "$repo" == "models" ]; then
            # set server arguments
            SERVER_ARGS="--model-repository=$MODEL_REPO --exit-timeout-secs=120 "
        else
            # set server arguments
            SERVER_ARGS="--model-repository=$ROOT_REPO --exit-timeout-secs=120 "
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

        # Clean up bucket
        gsutil -m rm "${BUCKET_URL_SLASH}**"

    done
done 

# Test with polling enabled
SERVER_ARGS="--model-repository=$ROOT_REPO --exit-timeout-secs=120 --model-control-mode=poll"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# copy contents of /models into S3 bucket and wait for them to be loaded.
gsutil cp -r models/ "$BUCKET_URL_SLASH"
sleep 60

set +e

# python unittest seems to swallow ImportError and still return 0
# exit code. So need to explicitly check CLIENT_LOG to make sure
# we see some running tests
python $INFER_TEST >$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

grep -c "HTTPSocketPoolResponse status=200" $CLIENT_LOG
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed To Run\n***"
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

# Delete all contents of bucket and bucket itself
gsutil rm -r "${BUCKET_URL}"

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
fi

exit $RET
