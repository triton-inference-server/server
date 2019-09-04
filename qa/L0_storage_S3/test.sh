#!/bin/bash
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

REPO_VERSION=${NVIDIA_TENSORRT_SERVER_VERSION}
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

# S3 bucket path (Point to bucket when testing cloud storage)

BUCKET_URL="s3://bucket"

# Remove Slash in BUCKET_URL
BUCKET_URL=${BUCKET_URL%/}
BUCKET_URL_SLASH="${BUCKET_URL}/"

SERVER=/opt/tensorrtserver/bin/trtserver
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
    aws s3 cp . "$BUCKET_URL_SLASH" --recursive --include "*"

    SERVER_ARGS="--model-repository=$MODEL_REPO --exit-timeout-secs=120"

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    kill $SERVER_PID
    wait $SERVER_PID

    aws s3 rm "${BUCKET_URL_SLASH}" --recursive --include "*"
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

        # copy contents of /models into GCS bucket.
        aws s3 cp $src $BUCKET_URL_SLASH --recursive --include "*"

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

        # python unittest seems to swallow ImportError and still return 0
        # exit code. So need to explicitly check CLIENT_LOG to make sure
        # we see some running tests
        python $INFER_TEST >$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi

        grep -c "HTTP/1.1 200 OK" $CLIENT_LOG
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed To Run\n***"
            RET=1
        fi

        set -e

        kill $SERVER_PID
        wait $SERVER_PID

        # Clean up bucket
        aws s3 rm "${BUCKET_URL_SLASH}" --recursive --include "*"

    done
done 

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
fi

exit $RET
