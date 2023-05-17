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
if [ ! -z "$TEST_REPO_ARCH" ]; then
    REPO_VERSION=${REPO_VERSION}_${TEST_REPO_ARCH}
fi

export CUDA_VISIBLE_DEVICES=0

TEST_RESULT_FILE='test_results.txt'
RET=0
rm -f *.log *.db
EXPECTED_NUM_TESTS="1"

mkdir -p models
cp -r /data/inferenceserver/${REPO_VERSION}/qa_identity_model_repository/savedmodel_zero_1_object models/

FUZZTEST=fuzztest.py
FUZZ_LOG=`pwd`/fuzz.log
DATADIR=`pwd`/models
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=$DATADIR"
source ../common/util.sh

# Remove this once foobuzz and tornado packages upgrade to work with python 3.10
# This test tests the server's ability to handle poor input and not the compatibility 
# with python 3.10. Python 3.8 is ok to use here.
function_install_python38() {
    apt-get update
    apt-get remove -y python3
    wget https://www.python.org/ftp/python/3.8.16/Python-3.8.16.tar.xz

    # md5sum is not secure. Only use for sanity check
    MD5SUM_PYTHON38=$(md5sum Python-3.8.16.tar.xz | awk '{ print $1 }' -)
    CORRECT_MD5SUM_PYTHON38=621ac153586a3152e2ab7d3a8614df9a
    if [ "$MD5SUM_PYTHON38" != "$CORRECT_MD5SUM_PYTHON38" ]; then
        echo "md5sum of downloaded Python-3.8.16.tar.xz does not match! $MD5SUM_PYTHON38 != $CORRECT_MD5SUM_PYTHON38"
        RET=1
    fi 

    # check the file size as well
    FILE_SIZE_PYTHON38=$(ls -l Python-3.8.16.tar.xz | awk '{ print $5 }' -)
    CORRECT_FILE_SIZE_PYTHON38=19046724
    if [ "$FILE_SIZE_PYTHON38" != "$CORRECT_FILE_SIZE_PYTHON38" ]; then
        echo "file size is not correct! $FILE_SIZE_PYTHON38 != $ $CORRECT_FILE_SIZE_PYTHON38" 
        RET=1
    fi
    echo "Validated md5sum and file size of Python-3.8.16.tar.xz"

    # Unpack python and install 
    tar -xf Python-3.8.16.tar.xz
    cd Python-3.8.16
    apt-get install -y libsqlite3-dev libffi-dev
    ./configure --enable-loadable-sqlite-extensions
    make 
    make install

    # Install test script dependencies
    pip3 install --upgrade wheel setuptools boofuzz==0.3.0 numpy pillow attrdict future grpcio requests gsutil \
                            awscli six grpcio-channelz prettytable virtualenv
}
WORKING_DIR=`pwd`
function_install_python38
cd $WORKING_DIR

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

# Test health
python3 $FUZZTEST -v >> ${FUZZ_LOG} 2>&1
if [ $? -ne 0 ]; then
    cat ${FUZZ_LOG}
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $TEST_RESULT_FILE
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi

set -e

kill $SERVER_PID
wait $SERVER_PID


if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
