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

# Cleanup GCS test bucket if exists (due to test failure)
gsutil -m rm -r ${GCS_BUCKET_URL} || true

# Create GCS test bucket
gsutil mb ${GCS_BUCKET_URL}


# S3 credentials are necessary for this test. Pass via ENV variables
aws configure set default.region $AWS_DEFAULT_REGION && \
    aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID && \
    aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY

# S3 bucket path
S3_BUCKET_URL="s3://s3-bucket-${CI_JOB_ID}"

# Cleanup and delete S3 test bucket if it already exists (due to test failure)
aws s3 rm ${S3_BUCKET_URL} --recursive --include "*" && \
    aws s3 rb ${S3_BUCKET_URL} || true

# Create S3 test bucket
aws s3 mb ${S3_BUCKET_URL}


ACCOUNT_NAME=$AZURE_STORAGE_ACCOUNT
ACCOUNT_KEY=$AZURE_STORAGE_KEY

AS_CONTAINER_NAME="azure-bucket-${CI_JOB_ID}"

# AS container path
AS_CONTAINER_URL="as://${ACCOUNT_NAME}/${AS_CONTAINER_NAME}"

# Must use setuptools version before 58.0.0 due to https://github.com/Azure/azure-cli/issues/19468
python -m pip install -U setuptools==57.5.0

# Can now install latest azure-cli (instead of 2.0.73)
python -m pip install azure-cli

# Cleanup Azure container test bucket if exists (due to test failure)
az storage container delete --name ${AS_CONTAINER_NAME} --account-name ${ACCOUNT_NAME} --account-key ${ACCOUNT_KEY}

# Create Azure test container
az storage container create --name ${AS_CONTAINER_NAME} --account-name ${ACCOUNT_NAME} --account-key ${ACCOUNT_KEY}
sleep 10

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_LOG="./inference_server.log"
source ../common/util.sh

rm -f $SERVER_LOG $CLIENT_LOG

RET=0

for CLOUD_REPO in az s3; do
    rm -rf models && mkdir -p models
    # Tensorflow
    # Must explicitly set LD_LIBRARY_PATH so that the custom operations
    # can find libtensorflow_framework.so.
    LD_LIBRARY_PATH=/opt/tritonserver/backends/tensorflow1:$LD_LIBRARY_PATH
    cp -rf /data/inferenceserver/${REPO_VERSION}/qa_custom_ops/tf_custom_ops/* models/

    if [ "$CLOUD_REPO" == "s3" ]; then
        # copy contents of models into S3 bucket and wait for them to be loaded.
        aws s3 cp models ${S3_BUCKET_URL} --recursive --include "*"
        SERVER_ARGS="--model-repository=${S3_BUCKET_URL}"
    else
        # copy contents of models into Azure Storage container.
        for file in `find models -type f`; do
            az storage blob upload --container-name ${AS_CONTAINER_NAME} --account-name ${ACCOUNT_NAME} --account-key ${ACCOUNT_KEY} --file $file --name $file
        done

        # We need a sleep here after Azure storage upload to ensure that the blobs are done uploading and available.  
        sleep 30
        SERVER_ARGS="--model-repository=${AS_CONTAINER_URL}/models"
    fi

    for PRELOAD in without_triton_lib with_triton_lib; do
        if [ "$PRELOAD" == "with_triton_lib" ]; then
            SERVER_LD_PRELOAD="/opt/tritonserver/lib/libtritonserver.so:/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/tf_custom_ops/libzeroout.so:/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/tf_custom_ops/libcudaop.so:/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/tf_custom_ops/libbusyop.so"
        else
            SERVER_LD_PRELOAD="/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/tf_custom_ops/libzeroout.so:/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/tf_custom_ops/libcudaop.so:/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/tf_custom_ops/libbusyop.so"
        fi

        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            echo -e "\n***\n*** Failed: $CLOUD_REPO $PRELOAD\n***"
            # exit 1
            RET=1
        else
            set +e

            python $ZERO_OUT_TEST -v -m graphdef_zeroout >>$CLIENT_LOG 2>&1
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                cat $SERVER_LOG
                echo -e "\n***\n*** Test Failed\n***"
                RET=1
            fi

            python $ZERO_OUT_TEST -v -m savedmodel_zeroout >>$CLIENT_LOG 2>&1
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                cat $SERVER_LOG
                echo -e "\n***\n*** Test Failed\n***"
                RET=1
            fi

            python $CUDA_OP_TEST -v -m graphdef_cudaop >>$CLIENT_LOG 2>&1
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                cat $SERVER_LOG
                echo -e "\n***\n*** Test Failed\n***"
                RET=1
            fi

            python $CUDA_OP_TEST -v -m savedmodel_cudaop >>$CLIENT_LOG 2>&1
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                cat $SERVER_LOG
                echo -e "\n***\n*** Test Failed\n***"
                RET=1
            fi

            set -e

            kill $SERVER_PID
            wait $SERVER_PID
        fi
    done

    # Clean up bucket contents
    aws s3 rm "${S3_BUCKET_URL}" --recursive --include "*"
    az storage container delete --name ${AS_CONTAINER_NAME} --account-name ${ACCOUNT_NAME} --account-key ${ACCOUNT_KEY}
    sleep 60
    az storage container create --name ${AS_CONTAINER_NAME} --account-name ${ACCOUNT_NAME} --account-key ${ACCOUNT_KEY}
    sleep 10

    # Pytorch
    # Must set LD_LIBRARY_PATH just for the server launch so that the
    # custom operations can find libtorch.so and other pytorch dependencies.
    LD_LIBRARY_PATH=/opt/tritonserver/backends/pytorch:$LD_LIBRARY_PATH
    rm -rf models && mkdir -p models
    cp -rf /data/inferenceserver/${REPO_VERSION}/qa_custom_ops/libtorch_custom_ops/* models/

    if [ "$CLOUD_REPO" == "s3" ]; then
        # copy contents of models into S3 bucket and wait for them to be loaded.
        aws s3 cp models ${S3_BUCKET_URL} --recursive --include "*"
        SERVER_ARGS="--model-repository=${S3_BUCKET_URL}"
    else
        # copy contents of models into Azure Storage container.
        for file in `find models -type f`; do
            az storage blob upload --container-name ${AS_CONTAINER_NAME} --account-name ${ACCOUNT_NAME} --account-key ${ACCOUNT_KEY} --file $file --name $file
        done

        # We need a sleep here after Azure storage upload to ensure that the blobs are done uploading and available.  
        sleep 30
        SERVER_ARGS="--model-repository=${AS_CONTAINER_URL}/models"
    fi

    for PRELOAD in without_triton_lib with_triton_lib; do
        if [ "$PRELOAD" == "with_triton_lib" ]; then
            SERVER_LD_PRELOAD="/opt/tritonserver/lib/libtritonserver.so:/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/libtorch_custom_ops/libtorch_modulo/custom_modulo.so"
        else
            SERVER_LD_PRELOAD="/data/inferenceserver/${REPO_VERSION}/qa_custom_ops/libtorch_custom_ops/libtorch_modulo/custom_modulo.so"
        fi

        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            echo -e "\n***\n*** Failed: $CLOUD_REPO $PRELOAD\n***"
            # exit 1
            RET=1
        else
            set +e

            python $MOD_OP_TEST -v -m libtorch_modulo >>$CLIENT_LOG 2>&1
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                cat $SERVER_LOG
                echo -e "\n***\n*** Test Failed\n***"
                RET=1
            fi

            python $VISION_OP_TEST -v -m libtorch_visionop >>$CLIENT_LOG 2>&1
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                cat $SERVER_LOG
                echo -e "\n***\n*** Test Failed\n***"
                RET=1
            fi

            set -e

            if [ $RET -eq 0 ]; then
            echo -e "\n***\n*** Test Passed\n***"
            fi

            kill $SERVER_PID
            wait $SERVER_PID
        fi
    done

    # Clean up bucket contents
    aws s3 rm "${S3_BUCKET_URL}" --recursive --include "*"
    az storage container delete --name ${AS_CONTAINER_NAME} --account-name ${ACCOUNT_NAME} --account-key ${ACCOUNT_KEY}
    sleep 60
    az storage container create --name ${AS_CONTAINER_NAME} --account-name ${ACCOUNT_NAME} --account-key ${ACCOUNT_KEY}
    sleep 10


    # ONNX
    rm -rf onnx_custom_ops && \
        mkdir -p onnx_custom_ops/custom_op/1 && \
        cp custom_op_test.onnx onnx_custom_ops/custom_op/1/model.onnx

    touch onnx_custom_ops/custom_op/config.pbtxt
    echo "name: \"custom_op\"" >> onnx_custom_ops/custom_op/config.pbtxt && \
    echo "platform: \"onnxruntime_onnx\"" >> onnx_custom_ops/custom_op/config.pbtxt && \
    echo "max_batch_size: 0" >> onnx_custom_ops/custom_op/config.pbtxt && \
    echo "model_operations { op_library_filename: \"./libcustom_op_library.so\" }" >> onnx_custom_ops/custom_op/config.pbtxt

    if [ "$CLOUD_REPO" == "s3" ]; then
        # copy contents of models into S3 bucket and wait for them to be loaded.
        aws s3 cp onnx_custom_ops ${S3_BUCKET_URL} --recursive --include "*"
        SERVER_ARGS="--model-repository=${S3_BUCKET_URL} --strict-model-config=false"
    else
        # copy contents of models into Azure Storage container.
        for file in `find onnx_custom_ops -type f`; do
            az storage blob upload --container-name ${AS_CONTAINER_NAME} --account-name ${ACCOUNT_NAME} --account-key ${ACCOUNT_KEY} --file $file --name $file
        done

        # We need a sleep here after Azure storage upload to ensure that the blobs are done uploading and available.  
        sleep 30
        SERVER_ARGS="--model-repository=${AS_CONTAINER_URL}/onnx_custom_ops --strict-model-config=false"
    fi

    for PRELOAD in without_triton_lib with_triton_lib; do
        if [ "$PRELOAD" == "with_triton_lib" ]; then
            SERVER_LD_PRELOAD="/opt/tritonserver/lib/libtritonserver.so"
        else
            SERVER_LD_PRELOAD=""
        fi

        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            echo -e "\n***\n*** Failed: $CLOUD_REPO $PRELOAD\n***"
            # exit 1
            RET=1
        else
            set +e

            python $ONNX_OP_TEST -v -m custom_op >>$CLIENT_LOG 2>&1
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                cat $SERVER_LOG
                echo -e "\n***\n*** Test Failed\n***"
                RET=1
            fi

            set -e
        fi
    done

    # Clean up bucket contents
    aws s3 rm "${S3_BUCKET_URL}" --recursive --include "*"
    az storage container delete --name ${AS_CONTAINER_NAME} --account-name ${ACCOUNT_NAME} --account-key ${ACCOUNT_KEY}
    sleep 60
    az storage container create --name ${AS_CONTAINER_NAME} --account-name ${ACCOUNT_NAME} --account-key ${ACCOUNT_KEY}
    sleep 10
done

# Delete all GCS bucket contents and bucket itself
gsutil -m rm -r ${GCS_BUCKET_URL}

# Delete S3 bucket contents and bucket itself
aws s3 rm ${S3_BUCKET_URL} --recursive --include "*" && \
    aws s3 rb ${S3_BUCKET_URL}

# Delete Azure storage container
az storage container delete --name ${AS_CONTAINER_NAME} --account-name ${ACCOUNT_NAME} --account-key ${ACCOUNT_KEY}


if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
fi

kill $SERVER_PID
wait $SERVER_PID

exit $RET
