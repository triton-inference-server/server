#!/bin/bash
# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

source ../common/util.sh

RET=0

TEST_LOG="./data_compressor_test.log"
DATA_COMPRESSOR_TEST=./data_compressor_test


export CUDA_VISIBLE_DEVICES=0

rm -fr *.log *_data

set +e

echo "All work and no play makes Jack a dull boy" >> raw_data
python3 validation.py generate_compressed_data

LD_LIBRARY_PATH=/opt/tritonserver/lib:${LD_LIBRARY_PATH} $DATA_COMPRESSOR_TEST >>$TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Data Compression Test Failed\n***"
    RET=1
fi

python3 validation.py validate_compressed_data
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Data Compression Failed\n***"
    RET=1
fi

set -e

# End-to-end testing with simple model
function run_data_compression_infer_client() {
    local client_path=$1
    local request_algorithm=$2
    local response_algorithm=$3
    local log_path=$4

    local python_or_cpp=`echo -n "$client_path" | tail -c 3`
    if [ "$python_or_cpp" == ".py" ]; then
        local infer_client="python $client_path"
        local request_cmd_option="--request-compression-algorithm $request_algorithm"
        local response_cmd_option="--response-compression-algorithm $response_algorithm"
    else  # C++ if not end with ".py"
        local infer_client=$client_path
        local request_cmd_option="-i $request_algorithm"
        local response_cmd_option="-o $response_algorithm"
    fi

    local cmd_options="-v"
    if [ "$request_algorithm" != "" ]; then
        cmd_options+=" $request_cmd_option"
    fi
    if [ "$response_algorithm" != "" ]; then
        cmd_options+=" $response_cmd_option"
    fi

    $infer_client $cmd_options >> $log_path 2>&1
    return $?
}

SIMPLE_INFER_CLIENT_PY=../clients/simple_http_infer_client.py
SIMPLE_AIO_INFER_CLIENT_PY=../clients/simple_http_aio_infer_client.py
SIMPLE_INFER_CLIENT=../clients/simple_http_infer_client

CLIENT_LOG=`pwd`/client.log
DATADIR=`pwd`/models
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=$DATADIR"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

for INFER_CLIENT in "$SIMPLE_INFER_CLIENT_PY" "$SIMPLE_AIO_INFER_CLIENT_PY" "$SIMPLE_INFER_CLIENT"; do
    for REQUEST_ALGORITHM in "deflate" "gzip" ""; do
        for RESPONSE_ALGORITHM in "deflate" "gzip" ""; do
            if [ "$REQUEST_ALGORITHM" == "$RESPONSE_ALGORITHM" ]; then
                continue
            fi

            set +e
            run_data_compression_infer_client "$INFER_CLIENT" "$REQUEST_ALGORITHM" "$RESPONSE_ALGORITHM" "$CLIENT_LOG"
            if [ $? -ne 0 ]; then
                RET=1
            fi
            set -e
        done
    done
done

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $TEST_LOG
    cat $SERVER_LOG
    cat ${CLIENT_LOG}
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
