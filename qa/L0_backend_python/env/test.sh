#!/bin/bash
# Copyright 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CLIENT_LOG="./env_client.log"
source ../common.sh
source ../../common/util.sh

BASE_SERVER_ARGS="--model-repository=${MODELDIR}/env/models --log-verbose=1 --disable-auto-complete-config"
PYTHON_BACKEND_BRANCH=$PYTHON_BACKEND_REPO_TAG
SERVER_ARGS=$BASE_SERVER_ARGS
SERVER_LOG="./env_server.log"

RET=0

rm -fr ./models
rm -rf *.tar.gz
install_build_deps
install_conda

# Test conda env without custom Python backend stub This environment should
# always use the default Python version shipped in the container. For Ubuntu
# 24.04 it is Python 3.12, for Ubuntu 22.04 is Python 3.10 and for Ubuntu 20.04
# is 3.8.
path_to_conda_pack='$$TRITON_MODEL_DIRECTORY/python_3_12_environment.tar.gz'
create_conda_env "3.12" "python-3-12"
conda install -c conda-forge libstdcxx-ng=14 -y
TORCH_VERSION="2.6.0"
conda install numpy=1.26.4 -y
if [ $TRITON_RHEL -eq 1 ]; then
    TORCH_VERISON="2.17.0"
fi
conda install pytorch=${TORCH_VERSION} -y
PY312_VERSION_STRING="Python version is 3.12, NumPy version is 1.26.4, and PyTorch version is ${TORCH_VERSION}"
conda pack -o python3.12.tar.gz
mkdir -p models/python_3_12/1/
cp ../../python_models/python_version/config.pbtxt ./models/python_3_12
cp python3.12.tar.gz models/python_3_12/python_3_12_environment.tar.gz
(cd models/python_3_12 && \
          sed -i "s/^name:.*/name: \"python_3_12\"/" config.pbtxt && \
          echo "parameters: {key: \"EXECUTION_ENV_PATH\", value: {string_value: \"$path_to_conda_pack\"}}" >> config.pbtxt)
cp ../../python_models/python_version/model.py ./models/python_3_12/1/
conda deactivate
rm -rf ./miniconda

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

kill_server

set +e
for EXPECTED_VERSION_STRING in "$PY312_VERSION_STRING"; do
    grep "$EXPECTED_VERSION_STRING" $SERVER_LOG
    if [ $? -ne 0 ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** $EXPECTED_VERSION_STRING was not found in Triton logs. \n***"
        RET=1
    fi
done

# Test default (non set) locale in python stub processes
# NOTE: In certain pybind versions, the locale settings may not be propagated from parent to
#       stub processes correctly. See https://github.com/triton-inference-server/python_backend/pull/260.
export LC_ALL=INVALID
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

kill_server

grep "Locale is (None, None)" $SERVER_LOG
    if [ $? -ne 0 ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** Default unset Locale was not found in Triton logs. \n***"
        RET=1
    fi
set -e

rm $SERVER_LOG

# Test locale set via environment variable in python stub processes
# NOTE: In certain pybind versions, the locale settings may not be propagated from parent to
#       stub processes correctly. See https://github.com/triton-inference-server/python_backend/pull/260.
export LC_ALL=C.UTF-8
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

kill_server

set +e
grep "Locale is ('C', 'UTF-8')" $SERVER_LOG
    if [ $? -ne 0 ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** Locale UTF-8 was not found in Triton logs. \n***"
        RET=1
    fi
set -e

rm $SERVER_LOG

## Test re-extraction of environment.
SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1 --model-control-mode=explicit"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# The environment should be extracted
curl -v -X POST localhost:8000/v2/repository/models/python_3_12/load
touch -m models/python_3_12/1/model.py
# The environment should not be re-extracted
curl -v -X POST localhost:8000/v2/repository/models/python_3_12/load
touch -m models/python_3_12/python_3_12_environment.tar.gz
# The environment should be re-extracted
curl -v -X POST localhost:8000/v2/repository/models/python_3_12/load

kill_server

set +e

PY312_ENV_EXTRACTION="Extracting Python execution env"
if [ `grep -c "${PY312_ENV_EXTRACTION}" ${SERVER_LOG}` != "2" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Python execution environment should be extracted exactly twice. \n***"
    RET=1
fi
set -e

# Test execution environments with S3
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

# Test with the bucket url as model repository
aws s3 cp models/ "${BUCKET_URL_SLASH}" --recursive --include "*"

rm $SERVER_LOG
# Occasionally needs more time to load
SERVER_TIMEOUT=420

SERVER_ARGS="--model-repository=$BUCKET_URL_SLASH --log-verbose=1"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    aws s3 rb "${BUCKET_URL}" --force || true
    exit 1
fi

kill_server

set +e
grep "$PY312_VERSION_STRING" $SERVER_LOG
if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** $PY312_VERSION_STRING was not found in Triton logs. \n***"
    RET=1
fi
set -e

# Clean up bucket contents
aws s3 rm "${BUCKET_URL_SLASH}" --recursive --include "*"

# Test with EXECUTION_ENV_PATH outside the model directory
sed -i "s/\$\$TRITON_MODEL_DIRECTORY\/python_3_12_environment/s3:\/\/triton-bucket-${CI_JOB_ID}\/python_3_12_environment/" models/python_3_12/config.pbtxt
mv models/python_3_12/python_3_12_environment.tar.gz models

aws s3 cp models/ "${BUCKET_URL_SLASH}" --recursive --include "*"

rm $SERVER_LOG

SERVER_ARGS="--model-repository=$BUCKET_URL_SLASH --log-verbose=1"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    aws s3 rb "${BUCKET_URL}" --force || true
    exit 1
fi

kill_server

set +e
for EXPECTED_VERSION_STRING in "$PY312_VERSION_STRING"; do
    grep "$EXPECTED_VERSION_STRING" $SERVER_LOG
    if [ $? -ne 0 ]; then
        cat $SERVER_LOG
        echo -e "\n***\n*** $EXPECTED_VERSION_STRING was not found in Triton logs. \n***"
        RET=1
    fi
done
set -e

# Clean up bucket contents and delete bucket
aws s3 rm "${BUCKET_URL_SLASH}" --recursive --include "*"
aws s3 rb "${BUCKET_URL}"

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Env Manager Test PASSED.\n***"
else
  cat $SERVER_LOG
  echo -e "\n***\n*** Env Manager Test FAILED.\n***"
fi

exit $RET
