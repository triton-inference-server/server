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
MODELBASE=onnx_int32_int32_int32

MODELSDIR=`pwd`/log_models

SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

rm -f *.log
rm -fr $MODELSDIR && mkdir -p $MODELSDIR

# set up simple repository MODELBASE
rm -fr $MODELSDIR && mkdir -p $MODELSDIR && \
    cp -r $DATADIR/$MODELBASE $MODELSDIR/simple && \
    rm -r $MODELSDIR/simple/2 && rm -r $MODELSDIR/simple/3 && \
    (cd $MODELSDIR/simple && \
            sed -i "s/^name:.*/name: \"simple\"/" config.pbtxt)
RET=0

function verify_correct_settings () {
  log_file_expected=$1
  log_info_expected=$2
  log_warn_expected=$3
  log_error_expected=$4
  log_verbose_expected=$5
  log_format_expected=$6
  code=`curl -s -w %{http_code} -o ./curl.out localhost:8000/v2/logging`

  if [ `grep -c "\"log_file\":\"$log_file_expected"\" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed: Incorrect Log File Setting\n***"
    RET=1
  fi
  if [ `grep -c "\"log_info\":$log_info_expected" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed: Incorrect Log Info Setting\n***"
    RET=1
  fi
  if [ `grep -c "\"log_warning\":$log_warn_expected" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed: Incorrect Log Warn Setting\n***"
    RET=1
  fi
  if [ `grep -c "\"log_error\":$log_error_expected" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed: Incorrect Log Error Setting\n***"
    RET=1
  fi
  if [ `grep -c "\"log_verbose_level\":$log_verbose_expected" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed: Incorrect Log Verbose Setting\n***"
    RET=1
  fi
  if [ `grep -c "\"log_format\":\"$log_format_expected\"" ./curl.out` != "1" ]; then
    echo -e "\n***\n*** Test Failed: Incorrect Log Format Setting\n***"
    RET=1
  fi
}

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

# Check if the current settings are returned [ file | info | warn | error | verbosity |format ]
verify_correct_settings "" "true" "true" "true" "0" "default"

$SIMPLE_HTTP_CLIENT >> client_default.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

$SIMPLE_GRPC_CLIENT >> client_default.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

# Check log is streaming to console by default
console_count=($(wc -l ./server.log))
if [ $console_count -le 30 ]; then
    echo -e "\n***\n*** Test Failed: Log File Error\n***"
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

# Test Log File (Argument)
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

verify_correct_settings "log_file.log" "true" "true" "true" "0" "default"

$SIMPLE_HTTP_CLIENT >> client_test_log_file.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

$SIMPLE_GRPC_CLIENT >> client_test_log_file.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi
expected_log_count=19
actual_log_count=$(grep -c ^[IWEV][0-9][0-9][0-9][0-9].* ./log_file.log)
if [ $actual_log_count -lt $expected_log_count ]; then
    echo $actual_log_count
    echo $expected_log_count
    echo -e "\n***\n*** Test Failed: Less Log Messages Than Expected $LINENO\n***"
    RET=1
fi
expected_server_count=0
actual_server_count=$(grep -c ^[IWEV][0-9][0-9][0-9][0-9].* inference_server_log_file.log)
if [ $actual_server_count -gt $expected_server_count ]; then
    echo $actual_server_count
    echo $expected_server_count
    echo -e "\n***\n*** Test Failed: More Log Messages Than Expected $LINENO\n***"
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# Test Log File (Dynamic)
rm -f log_file.log
SERVER_ARGS="--log-file=log_file.log --log-verbose=1 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_log_file.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

rm -f ./curl.out
code=`curl -s -w %{http_code} -o ./curl.out -d'{"log_file":"other_log.log"}' localhost:8000/v2/logging`
set +e

verify_correct_settings "other_log.log" "true" "true" "true" "1" "default"

$SIMPLE_HTTP_CLIENT >> client_test_log_file.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

$SIMPLE_GRPC_CLIENT >> client_test_log_file.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

# Check redirection worked properly (server log has tolerance of 40 due to
# unavoidable onnx framework logging)
expected_log_count=75
actual_log_count=$(grep -c ^[IWEV][0-9][0-9][0-9][0-9].* ./log_file.log)
if [ $actual_log_count -lt $expected_log_count ]; then
    echo $actual_log_count
    echo $expected_log_count
    echo -e "\n***\n*** Test Failed: Less Log Messages Than Expected $LINENO\n***"
    RET=1
fi
expected_other_log_count=31
actual_other_log_count=$(grep -c ^[IWEV][0-9][0-9][0-9][0-9].* ./other_log.log)
if [ $actual_other_log_count -lt $expected_other_log_count ]; then
    echo $actual_other_log_count
    echo $expected_other_log_count
    echo -e "\n***\n*** Test Failed: Less Log Messages Than Expected $LINENO\n***"
    RET=1
fi
expected_server_count=0
actual_server_count=$(grep -c ^[IWEV][0-9][0-9][0-9][0-9].* inference_server_log_file.log)
if [ $actual_server_count -gt $expected_server_count ]; then
    echo $actual_server_count
    echo $expected_server_count
    echo -e "\n***\n*** Test Failed: More Log Messages Than Expected $LINENO\n***"
    RET=1
fi

set -e
kill $SERVER_PID
wait $SERVER_PID

# Test Log Info (Argument)
rm -f log_file.log
SERVER_ARGS="--log-file=log_file.log --log-info=false --log-verbose=1 --model-repository=$MODELSDIR"
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

verify_correct_settings "log_file.log" "false" "true" "true" "1" "default"

$SIMPLE_HTTP_CLIENT >> client_test_log_info.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

$SIMPLE_GRPC_CLIENT >> client_test_log_info.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

# Test against guaranteed info message
count=$(grep -c "Started HTTPService at" ./log_file.log)
if [ $count -gt 0 ]; then
    echo -e "\n***\n*** Test Failed: Info Message Not Expected $LINENO\n***"
    RET=1
fi

set -e

# Test Log Info (Dynamic)
set +e
rm -f ./curl.out
code=`curl -s -w %{http_code} -o ./curl.out -d'{"log_info":true}' localhost:8000/v2/logging`

verify_correct_settings "log_file.log" "true" "true" "true" "1" "default"

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

set +e
# Test against guaranteed info message
count=$(grep -c "Waiting for in-flight requests to complete" ./log_file.log)
if [ $count -ne 1 ]; then
    echo -e "\n***\n*** Test Failed: Info Message Expected $LINENO\n***"
    RET=1
fi
set -e

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

verify_correct_settings "" "true" "false" "true" "0" "default"

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

# Check if the current settings are returned [ file | info | warn | error | verbosity |format ]
verify_correct_settings "" "true" "true" "false" "0" "default"

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

# Test Log Verbose Level (Argument)
rm -f log_file.log
SERVER_ARGS="--log-file=log_file.log --log-verbose=1 --model-repository=$MODELSDIR"
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

verify_correct_settings "log_file.log" "true" "true" "true" "1" "default"

$SIMPLE_HTTP_CLIENT >> client_test_log_verbose.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

$SIMPLE_GRPC_CLIENT >> client_test_log_verbose.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

count=$(grep -c "/v2/logging" ./log_file.log)
if [ $count -ne 2 ]; then
    echo -e "\n***\n*** Test Failed: Verbose Message Expected $LINENO\n***"
    RET=1
fi

code=`curl -s -w %{http_code} -o ./curl.out -d'{"log_verbose_level":0}' localhost:8000/v2/logging`
verify_correct_settings "log_file.log" "true" "true" "true" "0" "default"

code=`curl -s -w %{http_code} -o ./curl.out localhost:8000/v2/logging`
count=$(grep -c "/v2/logging" ./log_file.log)
if [ $count -gt 3 ]; then
    echo -e "\n***\n*** Test Failed: Too Many Verbose Messages $LINENO\n***"
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

# Test Log Format (Argument)
rm -f log_file.log
SERVER_ARGS="--log-file=log_file.log --log-verbose=1 --log-format=ISO8601 --model-repository=$MODELSDIR"
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
verify_correct_settings "log_file.log" "true" "true" "true" "1" "ISO8601"

$SIMPLE_HTTP_CLIENT >> client_test_log_format.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

$SIMPLE_GRPC_CLIENT >> client_test_log_format.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

line=$(head -n 1 log_file.log)
date=$(date '+%m%d')
final_date="I${date}"
format_date=$(echo $line | head -n1 | awk '{print $1;}')
if [[ $final_date == $format_date ]]; then
    echo -e "\n***\n*** Test Failed: Unexpected Log Format $LINENO\n***"
    RET=1
fi

set -e

# Test Log Format (Dynamic)
set +e
rm -f ./curl.out
code=`curl -s -w %{http_code} -o ./curl.out -d'{"log_format":"default"}' localhost:8000/v2/logging`
verify_correct_settings "log_file.log" "true" "true" "true" "1" "default"

line=$(tail -n 1 log_file.log)
date=$(date '+%m%d')
final_date="I${date}"
format_date=$(echo $line | head -n1 | awk '{print $1;}')
if [[ $final_date != $format_date ]]; then
    echo -e "\n***\n*** Test Failed: Unexpected Log Format $LINENO\n***"
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

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

BOOL_PARAMS=${BOOL_PARAMS:="log_info log_warning log_error"}
for BOOL_PARAM in $BOOL_PARAMS; do
    # Attempt to use integer instead of bool
    code=`curl -s -w %{http_code} -o ./curl.out -d'{"'"$BOOL_PARAM"'":1}' localhost:8000/v2/logging`
    if [ "$code" == "200" ]; then
        echo $code
        cat ./curl.out
        echo -e "\n***\n*** Test Failed: Line: $LINENO\n***"
        RET=1
    fi
    # Attempt to use upper-case bool
    code=`curl -s -w %{http_code} -o ./curl.out -d'{"'"$BOOL_PARAM"'":False}' localhost:8000/v2/logging`
    if [ "$code" == "200" ]; then
        cat ./curl.out
        echo -e "\n***\n*** Test Failed: Line: $LINENO\n***"
        RET=1
    fi
    # Attempt to use string bool
    code=`curl -s -w %{http_code} -o ./curl.out -d'{"'"$BOOL_PARAM"'":"false"}' localhost:8000/v2/logging`
    if [ "$code" == "200" ]; then
        echo $code
        cat ./curl.out
        echo -e "\n***\n*** Test Failed: Line: $LINENO\n***"
        RET=1
    fi
    # Positive test case
    code=`curl -s -w %{http_code} -o ./curl.out -d'{"'"$BOOL_PARAM"'":true}' localhost:8000/v2/logging`
    if [ "$code" != "200" ]; then
        cat ./curl.out
        echo -e "\n***\n*** Test Failed: Line: $LINENO\n***"
        RET=1
    fi
done

code=`curl -s -w %{http_code} -o ./curl.out -d'{"log_verbose_level":-1}' localhost:8000/v2/logging`
if [ "$code" == "200" ]; then
    echo $code
    cat ./curl.out
    echo -e "\n***\n*** Test Failed: Line: $LINENO\n***"
    RET=1
fi
code=`curl -s -w %{http_code} -o ./curl.out -d'{"log_verbose_level":"1"}' localhost:8000/v2/logging`
if [ "$code" == "200" ]; then
    echo $code
    cat ./curl.out
    echo -e "\n***\n*** Test Failed: Line: $LINENO\n***"
    RET=1
fi
code=`curl -s -w %{http_code} -o ./curl.out -d'{"log_verbose_level":0}' localhost:8000/v2/logging`
if [ "$code" != "200" ]; then
    echo $code
    cat ./curl.out
    echo -e "\n***\n*** Test Failed: Line: $LINENO\n***"
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
