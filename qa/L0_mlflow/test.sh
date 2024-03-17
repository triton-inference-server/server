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

export CUDA_VISIBLE_DEVICES=0

source ../common/util.sh

rm -fr *.log *.json

# The default version of python 3.10.6 included in
# Ubuntu 22.04 installs blinker 1.4. This doesn't
# work with the awscli which we try to install.
# Uninstalling blinker and allowing pip to install blinker 1.6
# fixes this issue. The alternative to this is to
# install a higher version of python which uses blinker 1.6,
# but it is unknown whether this test should rely on
# the default installation of python.
apt remove -y python3-blinker

RET=0

# Set up MLflow and dependencies used by the test
pip install mlflow onnx onnxruntime boto3

# Install AWS CLI
if ! command -v aws --version &> /dev/null; then
 curl "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o "awscliv2.zip"
 unzip awscliv2.zip
 ./aws/install
 rm -r ./aws/ ./awscliv2.zip
fi

# Set environment variables for MLFlow and Triton plugin
export MLFLOW_MODEL_REPO=./mlflow/artifacts
export MLFLOW_TRACKING_URI=sqlite:////tmp/mlflow-db.sqlite
export TRITON_URL=localhost:8000
export TRITON_MODEL_REPO=models
mkdir -p ./mlflow/artifacts

pip install ./mlflow-triton-plugin/

# Clear mlflow registered models if any
python - << EOF
from mlflow.tracking import MlflowClient
c = MlflowClient()
for m in c.search_registered_models():
    c.delete_registered_model(m.name)
EOF

rm -rf ./models
mkdir -p ./models
# Put some models in model repository to make sure MLFlow plugin would ignore
# model that is not registered via MLFlow
cp -r ./mlflow-triton-plugin/examples/onnx_float32_int32_int32 ./models/existing_model

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=./models --strict-model-config=false --model-control-mode=explicit --load-model=*"
SERVER_LOG="./inference_server.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** fail to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# Triton flavor with CLI
set +e
CLI_LOG=plugin_cli.log
CLI_RET=0
python ./mlflow-triton-plugin/scripts/publish_model_to_mlflow.py \
    --model_name onnx_float32_int32_int32 \
    --model_directory ./mlflow-triton-plugin/examples/onnx_float32_int32_int32/ \
    --flavor triton >>$CLI_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Expect 'triton' flavor model is logged to MLFlow\n***"
    CLI_RET=1
fi
if [ $CLI_RET -eq 0 ]; then
    mlflow deployments create -t triton --flavor triton \
        --name onnx_float32_int32_int32 -m models:/onnx_float32_int32_int32/1 >>$CLI_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Expect 'triton' flavor model is deployed via MLFlow\n***"
        CLI_RET=1
    fi
fi
if [ $CLI_RET -eq 0 ]; then
    mlflow deployments list -t triton >>$CLI_LOG 2>&1
    if [ $? -ne 0 ]; then
        CLI_RET=1
    fi
    if [ `grep -c "onnx_float32_int32_int32.*READY" $CLI_LOG` != "1" ]; then
        echo -e "\n***\n*** Expect deployed 'triton' flavor model to be listed\n***"
        CLI_RET=1
    fi
    if [ `grep -c "existing_model.*READY" $CLI_LOG` != "0" ]; then
        echo -e "\n***\n*** Unexpected non-MLflow model listed\n***"
        CLI_RET=1
    fi
fi
if [ $CLI_RET -eq 0 ]; then
    mlflow deployments get -t triton --name onnx_float32_int32_int32 >>$CLI_LOG 2>&1
    if [ $? -ne 0 ]; then
        CLI_RET=1
    fi
    if [ `grep -c "^name: onnx_float32_int32_int32" $CLI_LOG` != "1" ]; then
        echo -e "\n***\n*** Expect deployed 'triton' flavor model is found\n***"
        CLI_RET=1
    fi
fi
if [ $CLI_RET -eq 0 ]; then
    mlflow deployments predict -t triton --name onnx_float32_int32_int32 --input-path ./mlflow-triton-plugin/examples/input.json --output-path output.json >>$CLI_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Expect successful 'triton' flavor model prediction\n***"
        CLI_RET=1
    fi
    python - << EOF
import json
with open("./output.json", "r") as f:
    output = json.load(f)
with open("./mlflow-triton-plugin/examples/expected_output.json", "r") as f:
    expected_output = json.load(f)
if output == expected_output:
    exit(0)
else:
    exit(1)
EOF
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Expect 'triton' flavor model prediction matches expected output\n***"
        echo -e "Expect:\n"
        cat ./mlflow-triton-plugin/examples/expected_output.json
        echo -e "\n\nGot:\n"
        cat output.json
        CLI_RET=1
    fi
fi
if [ $CLI_RET -eq 0 ]; then
    mlflow deployments delete -t triton --name onnx_float32_int32_int32 >>$CLI_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Expect successful deletion of 'triton' flavor model\n***"
        CLI_RET=1
    fi
fi
if [ $CLI_RET -ne 0 ]; then
  cat $CLI_LOG
  echo -e "\n***\n*** MLFlow Triton plugin CLI Test FAILED\n***"
  RET=1
fi
set -e

# ONNX flavor with Python package
set +e
PY_LOG=plugin_py.log
PY_TEST=plugin_test.py
TEST_RESULT_FILE='test_results.txt'
python $PY_TEST >>$PY_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    cat $PY_LOG
    echo -e "\n***\n*** Python Test Failed\n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE 2
    if [ $? -ne 0 ]; then
        cat $PY_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

kill_server


#
# Test S3, the setup is duplicated from L0_storage_S3, except the bucket is
# created empty
#

# Clear mlflow registered models if any
python - << EOF
from mlflow.tracking import MlflowClient
c = MlflowClient()
for m in c.search_registered_models():
    c.delete_registered_model(m.name)
EOF

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

export TRITON_MODEL_REPO=${BUCKET_URL}
SERVER_ARGS="--model-repository=${TRITON_MODEL_REPO} --model-control-mode=explicit"
SERVER_LOG="./inference_server.s3.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    # Clean up bucket contents and delete bucket before exiting test
    aws s3 rm "${BUCKET_URL_SLASH}" --recursive --include "*"
    aws s3 rb "${BUCKET_URL}"
    exit 1
fi

# ONNX flavor with Python package
set +e
PY_LOG=plugin_py.s3.log
PY_TEST=plugin_test.py
TEST_RESULT_FILE='test_results.txt'
python $PY_TEST >>$PY_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    cat $PY_LOG
    echo -e "\n***\n*** Python Test Failed\n***"
    RET=1
else
    check_test_results $TEST_RESULT_FILE 2
    if [ $? -ne 0 ]; then
        cat $PY_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

kill_server

# Clean up bucket contents and delete bucket
aws s3 rm "${BUCKET_URL_SLASH}" --recursive --include "*"
aws s3 rb "${BUCKET_URL}"

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
