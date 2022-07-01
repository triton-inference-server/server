#!/bin/bash
# Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
if [ ! -z "$TEST_REPO_ARCH" ]; then
    REPO_VERSION=${REPO_VERSION}_${TEST_REPO_ARCH}
fi

export CUDA_VISIBLE_DEVICES=0

CLIENT_LOG="./client.log"
ZERO_OUT_TEST=zero_out_test.py
CUDA_OP_TEST=cuda_op_test.py
MOD_OP_TEST=mod_op_test.py
VISION_OP_TEST=vision_op_test.py
ONNX_OP_TEST=onnx_op_test.py

# GCS credentials are necessary for this test. Pass via ENV variables
export GOOGLE_APPLICATION_CREDENTIALS="file.json"

echo '{"type": "service_account",' >> $GOOGLE_APPLICATION_CREDENTIALS
echo '"project_id": "triton-285001",' >> $GOOGLE_APPLICATION_CREDENTIALS
echo -e "\"private_key_id\": \"$GCP_PROJECT_KEY_ID\"," >> $GOOGLE_APPLICATION_CREDENTIALS
echo -e "\"private_key\": \"$GCP_PROJECT_KEY\"," >> $GOOGLE_APPLICATION_CREDENTIALS
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

# GCS bucket path
GCS_BUCKET_URL="gs://gcs-bucket-${CI_JOB_ID}"

# S3 credentials are necessary for this test. Pass via ENV variables
aws configure set default.region $AWS_DEFAULT_REGION && \
    aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID && \
    aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY

# S3 bucket path
S3_BUCKET_URL="s3://s3-bucket-${CI_JOB_ID}"

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_LOG="./inference_server.log"
source ../common/util.sh

rm -f $SERVER_LOG $CLIENT_LOG

RET=0

# Tensorflow
# Using Tensorflow custom ops with GCS and S3 cloud storages may cause symbol confliction
# issues. Need to add 'libtritonserver.so' library to LD_PRELOAD variable to overwrite
# the symbols so that preloaded library is not corrupting the cloud storage workflow.
for REPO in local gcs s3; do
    # Must explicitly set LD_LIBRARY_PATH so that the custom operations
    # can find libtensorflow_framework.so.
    LD_LIBRARY_PATH=/opt/tritonserver/backends/tensorflow1:$LD_LIBRARY_PATH

    if [ "$REPO" == "local" ]; then
        SERVER_ARGS="--model-repository=/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/tf_custom_ops"
        SERVER_LD_PRELOAD="/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/tf_custom_ops/libzeroout.so:/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/tf_custom_ops/libcudaop.so:/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/tf_custom_ops/libbusyop.so"
    elif [ "$REPO" == "gcs" ]; then
        # Cleanup GCS test bucket if exists
        gsutil -m rm -r ${GCS_BUCKET_URL} || true
        # Create and delete GCS test bucket if it already exists
        gsutil mb ${GCS_BUCKET_URL}
        # Copy contents of models
        gsutil -m cp -r /data/inferenceserver/${REPO_VERSION}/qa_custom_ops/tf_custom_ops ${GCS_BUCKET_URL}

        # Add 'libtritonserver.so' to LD_PRELOAD
        SERVER_LD_PRELOAD="/opt/tritonserver/lib/libtritonserver.so:/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/tf_custom_ops/libzeroout.so:/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/tf_custom_ops/libcudaop.so:/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/tf_custom_ops/libbusyop.so"
        SERVER_ARGS="--model-repository=${GCS_BUCKET_URL}/tf_custom_ops"
    else
        # Cleanup and delete S3 test bucket if it already exists
        aws s3 rm ${S3_BUCKET_URL} --recursive --include "*" && \
            aws s3 rb ${S3_BUCKET_URL} || true
        # Create S3 test bucket
        aws s3 mb ${S3_BUCKET_URL}
        # Copy contents of models
        aws s3 cp /data/inferenceserver/${REPO_VERSION}/qa_custom_ops/tf_custom_ops ${S3_BUCKET_URL} --recursive --include "*"

        # Add 'libtritonserver.so' to LD_PRELOAD
        SERVER_LD_PRELOAD="/opt/tritonserver/lib/libtritonserver.so:/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/tf_custom_ops/libzeroout.so:/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/tf_custom_ops/libcudaop.so:/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/tf_custom_ops/libbusyop.so"
        SERVER_ARGS="--model-repository=${S3_BUCKET_URL}"
    fi

    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e

    python $ZERO_OUT_TEST -v -m graphdef_zeroout >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi

    python $ZERO_OUT_TEST -v -m savedmodel_zeroout >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi

    python $CUDA_OP_TEST -v -m graphdef_cudaop >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi

    python $CUDA_OP_TEST -v -m savedmodel_cudaop >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi

    set -e

    kill $SERVER_PID
    wait $SERVER_PID
done


# Pytorch
# Must set LD_LIBRARY_PATH just for the server launch so that the
# custom operations can find libtorch.so and other pytorch dependencies.
LD_LIBRARY_PATH=/opt/tritonserver/backends/pytorch:$LD_LIBRARY_PATH

SERVER_ARGS="--model-repository=/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/libtorch_custom_ops"
SERVER_LD_PRELOAD="/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/libtorch_custom_ops/libtorch_modulo/custom_modulo.so"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

python $MOD_OP_TEST -v -m libtorch_modulo >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

python $VISION_OP_TEST -v -m libtorch_visionop >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
fi

kill $SERVER_PID
wait $SERVER_PID


# ONNX
rm -rf onnx_custom_ops && \
    mkdir -p onnx_custom_ops/custom_op/1 && \
    cp custom_op_test.onnx onnx_custom_ops/custom_op/1/model.onnx

touch onnx_custom_ops/custom_op/config.pbtxt
echo "name: \"custom_op\"" >> onnx_custom_ops/custom_op/config.pbtxt && \
echo "platform: \"onnxruntime_onnx\"" >> onnx_custom_ops/custom_op/config.pbtxt && \
echo "max_batch_size: 0" >> onnx_custom_ops/custom_op/config.pbtxt && \
echo "model_operations { op_library_filename: \"./libcustom_op_library.so\" }" >> onnx_custom_ops/custom_op/config.pbtxt

SERVER_ARGS="--model-repository=onnx_custom_ops --strict-model-config=false"
SERVER_LD_PRELOAD=""
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

python $ONNX_OP_TEST -v -m custom_op >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
fi

kill $SERVER_PID
wait $SERVER_PID

# Delete GCS bucket contents and bucket itself
gsutil -m rm -r ${GCS_BUCKET_URL}

# Delete S3 bucket contents and bucket itself
aws s3 rm ${S3_BUCKET_URL} --recursive --include "*" && \
    aws s3 rb ${S3_BUCKET_URL}

exit $RET
