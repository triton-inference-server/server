#!/bin/bash
# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


SIMPLE_HTTP_CLIENT=../clients/simple_http_infer_client
SIMPLE_GRPC_CLIENT=../clients/simple_grpc_infer_client

CLIENT_TEST=logging_endpoint_test.py
CLIENT_LOG="client.log"
TEST_RESULT_FILE="test_results.txt"
EXPECTED_NUM_TESTS="4"

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

DATADIR=/data/inferenceserver/${REPO_VERSION}/qa_model_repository
ENSEMBLEDIR=$DATADIR/../qa_ensemble_model_repository/qa_model_repository/
MODELBASE=onnx_int32_int32_int32

MODELSDIR=`pwd`/log_models

SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

rm -f *.log
rm -fr $MODELSDIR && mkdir -p $MODELSDIR

# set up simple and global_simple model using MODELBASE
rm -fr $MODELSDIR && mkdir -p $MODELSDIR && \
    cp -r $DATADIR/$MODELBASE $MODELSDIR/simple && \
    rm -r $MODELSDIR/simple/2 && rm -r $MODELSDIR/simple/3 && \
    (cd $MODELSDIR/simple && \
            sed -i "s/^name:.*/name: \"simple\"/" config.pbtxt) && \
    cp -r $MODELSDIR/simple $MODELSDIR/global_simple && \
    (cd $MODELSDIR/global_simple && \
            sed -i "s/^name:.*/name: \"global_simple\"/" config.pbtxt) && \

RET=0

#Run Default Server
SERVER_ARGS="--model-repository=$MODELSDIR"
SERVER_LOG="./server.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# Check Default Settings
rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out localhost:8000/v2/logging`

# Check if the current setting is returned
if [ `grep -c "\"log_file\":\""\" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log File Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_info\":true" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Info Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_warnings\":true" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Warn Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_errors\":true" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Error Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_verbose_level\":0" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Verbose Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_format\":\"default\"" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Format Setting\n***"
    RET=1
fi

$SIMPLE_HTTP_CLIENT >> client_default.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

$SIMPLE_GRPC_CLIENT >> client_default.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi


set -e

kill $SERVER_PID
wait $SERVER_PID

# Test Log File
SERVER_ARGS="--log-file=log_file.log --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_log_file.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out localhost:8000/v2/logging`
set -e

if [ `grep -c "\"log_file\":\"log_file.log"\" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log File Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_info\":true" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Info Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_warnings\":true" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Warn Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_errors\":true" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Error Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_verbose_level\":0" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Verbose Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_format\":\"default\"" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Format Setting\n***"
    RET=1
fi

$SIMPLE_HTTP_CLIENT >> client_test_log_file.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

$SIMPLE_GRPC_CLIENT >> client_test_log_file.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

# Test Log Info
SERVER_ARGS="--log-info=false --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_log_file.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out localhost:8000/v2/logging`
set -e

if [ `grep -c "\"log_file\":\""\" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log File Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_info\":false" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Info Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_warnings\":true" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Warn Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_errors\":true" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Error Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_verbose_level\":0" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Verbose Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_format\":\"default\"" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Format Setting\n***"
    RET=1
fi


$SIMPLE_HTTP_CLIENT >> client_test_log_info.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

$SIMPLE_GRPC_CLIENT >> client_test_log_info.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

# Test Log Warning
SERVER_ARGS="--log-warning=false --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_log_file.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out localhost:8000/v2/logging`
set -e
if [ `grep -c "\"log_file\":\""\" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log File Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_info\":true" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Info Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_warnings\":false" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Warn Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_errors\":true" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Error Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_verbose_level\":0" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Verbose Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_format\":\"default\"" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Format Setting\n***"
    RET=1
fi

$SIMPLE_HTTP_CLIENT >> client_test_log_warning.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

$SIMPLE_GRPC_CLIENT >> client_test_log_warning.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

# Test Log Error
SERVER_ARGS="--log-error=false --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_log_file.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out localhost:8000/v2/logging`
set -e

if [ `grep -c "\"log_file\":\""\" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log File Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_info\":true" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Info Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_warnings\":true" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Warn Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_errors\":false" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Error Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_verbose_level\":0" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Verbose Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_format\":\"default\"" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Format Setting\n***"
    RET=1
fi

$SIMPLE_HTTP_CLIENT >> client_test_log_error.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

$SIMPLE_GRPC_CLIENT >> client_test_log_error.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

# Test Log Verbose Level
SERVER_ARGS="--log-verbose=1 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_log_file.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out localhost:8000/v2/logging`
set -e

if [ `grep -c "\"log_file\":\""\" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log File Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_info\":true" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Info Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_warnings\":true" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Warn Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_errors\":true" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Error Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_verbose_level\":1" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Verbose Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_format\":\"default\"" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Format Setting\n***"
    RET=1
fi

$SIMPLE_HTTP_CLIENT >> client_test_log_verbose.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

$SIMPLE_GRPC_CLIENT >> client_test_log_verbose.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

# Test Log Format
SERVER_ARGS="--log-format=ISO8601 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_log_file.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f ./curl.out
set +e
code=`curl -s -w %{http_code} -o ./curl.out localhost:8000/v2/logging`
set -e

if [ `grep -c "\"log_file\":\""\" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log File Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_info\":true" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Info Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_warnings\":true" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Warn Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_errors\":true" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Error Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_verbose_level\":0" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Verbose Setting\n***"
    RET=1
fi
if [ `grep -c "\"log_format\":\"ISO8601\"" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed Incorrect Log Format Setting\n***"
    RET=1
fi

$SIMPLE_HTTP_CLIENT >> client_test_log_format.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

$SIMPLE_GRPC_CLIENT >> client_test_log_format.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

# Test that log file was created and has contents
count=$(grep -c "http_server.cc" ./log_file.log)
if [ $count -le 0 ]; then
    echo -e "\n***\n*** Test Failed Log File Error\n***"
    RET=1
fi

#Test Negative Test Cases
SERVER_ARGS="--log-warn="false" --model-repository=$MODELSDIR"
SERVER_LOG="./server.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

BOOL_PARAMS=${BOOL_PARAMS:="log_info log_warnings log_errors"}
for BOOL_PARAM in $BOOL_PARAMS; do
    code=`curl -s -w %{http_code} -o ./curl.out -d'{"'"$BOOL_PARAM"'":1}' localhost:8000/v2/logging`
    if [ "$code" != "400" ]; then
        echo $code
        cat ./curl.out
        echo -e "\n***\n*** Test Failed Line: $LINENO\n***"
        RET=1
    fi
    code=`curl -s -w %{http_code} -o ./curl.out -d'{"'"$BOOL_PARAM"'":False}' localhost:8000/v2/logging`
    if [ "$code" != "400" ]; then
        cat ./curl.out
        echo -e "\n***\n*** Test Failed Line: $LINENO\n***"
        RET=1
    fi
    code=`curl -s -w %{http_code} -o ./curl.out -d'{"'"$BOOL_PARAM"'":"false"}' localhost:8000/v2/logging`
    if [ "$code" != "400" ]; then
        echo $code
        cat ./curl.out
        echo -e "\n***\n*** Test Failed Line: $LINENO\n***"
        RET=1
    fi
    code=`curl -s -w %{http_code} -o ./curl.out -d'{"'"$BOOL_PARAM"'":true}' localhost:8000/v2/logging`
    if [ "$code" != "200" ]; then
        cat ./curl.out
        echo -e "\n***\n*** Test Failed Line: $LINENO\n***"
        RET=1
    fi
done

code=`curl -s -w %{http_code} -o ./curl.out -d'{"log_verbose_level":-1}' localhost:8000/v2/logging`
if [ "$code" != "400" ]; then
    echo $code
    cat ./curl.out
    echo -e "\n***\n*** Test Failed Line: $LINENO\n***"
    RET=1
fi
code=`curl -s -w %{http_code} -o ./curl.out -d'{"log_verbose_level":"1"}' localhost:8000/v2/logging`
if [ "$code" != "400" ]; then
    echo $code
    cat ./curl.out
    echo -e "\n***\n*** Test Failed Line: $LINENO\n***"
    RET=1
fi
code=`curl -s -w %{http_code} -o ./curl.out -d'{"log_verbose_level":0}' localhost:8000/v2/logging`
if [ "$code" != "200" ]; then
    echo $code
    cat ./curl.out
    echo -e "\n***\n*** Test Failed Line: $LINENO\n***"
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

# Test Python client library
SERVER_ARGS="--model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_unittest.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

python $CLIENT_TEST >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
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



if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi


exit $RET
