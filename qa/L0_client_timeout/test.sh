#!/bin/bash
# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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
TIMEOUT_VALUE=100000000
SHORT_TIMEOUT_VALUE=1000
RET=0

CLIENT_INFER_TIMEOUT_TEST=client_infer_timeout_test.py
CLIENT_NON_INFER_TIMEOUT_TEST=client_non_infer_timeout_test.py
CLIENT_TIMEOUT_TEST_CPP=../clients/client_timeout_test
TEST_RESULT_FILE='test_results.txt'

rm -f *.log
rm -f *.log.*

CLIENT_LOG=`pwd`/client.log
CLIENT_GRPC_TIMEOUTS_LOG=`pwd`/client.log.grpc
DATADIR=`pwd`/models
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=$DATADIR --model-control-mode=explicit --load-model=custom_identity_int32 --log-verbose 2"
source ../common/util.sh

mkdir -p $DATADIR/custom_identity_int32/1

# Test all APIs apart from Infer.
export TRITONSERVER_SERVER_DELAY_GRPC_RESPONSE_SEC=2
run_server
if [ $? -eq 1 ]; then
    echo -e "\n***\n*** Test Failed: GRPC non-infer APIs\n***"
    RET=1
fi
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
# Expect timeout for everything
$CLIENT_TIMEOUT_TEST_CPP -t $SHORT_TIMEOUT_VALUE -v -i grpc -p >> ${CLIENT_LOG}.c++.grpc_non_infer_apis 2>&1
if [ `grep -c "Deadline Exceeded" ${CLIENT_LOG}.c++.grpc_non_infer_apis` != "18" ]; then
    cat ${CLIENT_LOG}.c++.grpc_non_infer_apis
    echo -e "\n***\n*** Test Failed. Expected 18 failed\n***"
    RET=1
fi
# Test all APIs with long timeout
$CLIENT_TIMEOUT_TEST_CPP -t $TIMEOUT_VALUE -v -i grpc -p >> ${CLIENT_LOG} 2>&1
if [ $? -eq 0 ]; then
    echo -e "\n***\n*** Test Failed: GRPC non-infer APIs\n***"
    RET=1
fi

set -e
kill $SERVER_PID
wait $SERVER_PID

# Test infer APIs
unset TRITONSERVER_SERVER_DELAY_GRPC_RESPONSE_SEC
SERVER_ARGS="--model-repository=$DATADIR --log-verbose 2"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
set +e

# CASE 1: Provide too small a timeout and expect a failure.
# Note, the custom_identity_int32 is configured with a delay
# of 3 sec.
# Test request timeout in grpc synchronous inference
$CLIENT_TIMEOUT_TEST_CPP -t $SHORT_TIMEOUT_VALUE -v -i grpc >> ${CLIENT_LOG}.c++.grpc_infer 2>&1
if [ $? -eq 0 ]; then
    RET=1
fi
if [ `grep -c "Deadline Exceeded" ${CLIENT_LOG}.c++.grpc_infer` != "1" ]; then
    cat ${CLIENT_LOG}.c++.grpc_infer
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

# Test request timeout in grpc asynchronous inference
$CLIENT_TIMEOUT_TEST_CPP -t $SHORT_TIMEOUT_VALUE -v -i grpc -a >> ${CLIENT_LOG}.c++.grpc_async_infer 2>&1
if [ $? -eq 0 ]; then
    RET=1
fi
if [ `grep -c "Deadline Exceeded" ${CLIENT_LOG}.c++.grpc_async_infer` != "1" ]; then
    cat ${CLIENT_LOG}.c++.grpc_async_infer
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

# Test stream timeout in grpc asynchronous streaming inference
$CLIENT_TIMEOUT_TEST_CPP -t $SHORT_TIMEOUT_VALUE -v -i grpc -s >> ${CLIENT_LOG}.c++.grpc_async_stream_infer 2>&1
if [ $? -eq 0 ]; then
    RET=1
fi
if [ `grep -c "Stream has been closed" ${CLIENT_LOG}.c++.grpc_async_stream_infer` != "1" ]; then
    cat ${CLIENT_LOG}.c++.grpc_async_stream_infer
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

# Test request timeout in http synchronous inference
$CLIENT_TIMEOUT_TEST_CPP -t $SHORT_TIMEOUT_VALUE -v >> ${CLIENT_LOG}.c++.http_infer 2>&1
if [ $? -eq 0 ]; then
    RET=1
fi
if [ `grep -c "Deadline Exceeded" ${CLIENT_LOG}.c++.http_infer` == "0" ]; then
    cat ${CLIENT_LOG}.c++.http_infer
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi


# Test request timeout in http asynchronous inference
$CLIENT_TIMEOUT_TEST_CPP -t $SHORT_TIMEOUT_VALUE -v -a >> ${CLIENT_LOG}.c++.http_async_infer 2>&1
if [ $? -eq 0 ]; then
    RET=1
fi
if [ `grep -c "Deadline Exceeded" ${CLIENT_LOG}.c++.http_async_infer` == "0" ]; then
    cat ${CLIENT_LOG}.c++.http_async_infer
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

if [ $RET -eq 1 ]; then
    # Return if CASE 1 failed
    kill $SERVER_PID
    wait $SERVER_PID
    exit $RET
fi


# CASE 2: Provide sufficiently large timeout value
set +e

echo "TEST:  GRPC Synchronous" >> ${CLIENT_LOG}
$CLIENT_TIMEOUT_TEST_CPP -t $TIMEOUT_VALUE -v -i grpc >> ${CLIENT_LOG} 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed: GRPC Synchronous\n***"
    RET=1
fi

echo "TEST:  GRPC Asynchronous" >> ${CLIENT_LOG}
$CLIENT_TIMEOUT_TEST_CPP -t $TIMEOUT_VALUE -v -i grpc -a >> ${CLIENT_LOG}.c++.grpc_async_infer 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed: GRPC Asynchronous\n***"
    RET=1
fi

echo "TEST:  GRPC Streaming" >> ${CLIENT_LOG}
$CLIENT_TIMEOUT_TEST_CPP -t $TIMEOUT_VALUE -v -i grpc -s >> ${CLIENT_LOG}.c++.grpc_async_stream_infer 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed: GRPC Streaming\n***"
    RET=1
fi

echo "TEST:  HTTP Synchronous" >> ${CLIENT_LOG}
$CLIENT_TIMEOUT_TEST_CPP -t $TIMEOUT_VALUE -v >> ${CLIENT_LOG}.c++.http_infer 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed: HTTP Synchronous\n***"
    RET=1
fi

echo "TEST:  HTTP Asynchronous" >> ${CLIENT_LOG}
$CLIENT_TIMEOUT_TEST_CPP -t $TIMEOUT_VALUE -v -a >> ${CLIENT_LOG}.c++.http_async_infer 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed: HTTP Asynchronous\n***"
    RET=1
fi

echo "TEST:  Python Library" >> ${CLIENT_LOG}

# CASE 3: Python Library

for i in test_grpc_infer \
    test_grpc_async_infer \
    test_grpc_stream_infer \
    test_http_infer \
    test_http_async_infer \
   ; do
    python $CLIENT_INFER_TIMEOUT_TEST ClientInferTimeoutTest.$i >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test $i Failed\n***" >>$CLIENT_LOG
            echo -e "\n***\n*** Test $i Failed\n***"
            RET=1
    else
        check_test_results $TEST_RESULT_FILE 1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi
done

set -e
kill $SERVER_PID
wait $SERVER_PID

# Test all APIs other than infer
export TRITONSERVER_SERVER_DELAY_GRPC_RESPONSE_SEC=2
SERVER_ARGS="${SERVER_ARGS} --model-control-mode=explicit --load-model=custom_identity_int32 --log-verbose 2"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi
set +e

python $CLIENT_NON_INFER_TIMEOUT_TEST >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test $i Failed\n***" >>$CLIENT_LOG
    echo -e "\n***\n*** Test $i Failed\n***"
    RET=1
fi

set -e
kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat ${CLIENT_LOG}
    echo -e "\n***\n*** Test FAILED\n***"
fi

set +e
exit $RET
