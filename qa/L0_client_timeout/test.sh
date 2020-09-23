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

export CUDA_VISIBLE_DEVICES=0

RET=0

CLIENT_TIMEOUT_TEST=client_timeout_test.py

rm -f *.log
rm -f *.log.*

CLIENT_LOG=`pwd`/client.log
DATADIR=`pwd`/models
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=$DATADIR"
source ../common/util.sh

mkdir -p $DATADIR/custom_identity_int32/1
cp libidentity.so $DATADIR/custom_identity_int32/1/.

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

# CASE 1: Provide too small a timeout and expect a failure.

# Test request timeout in grpc synchronous inference
CLIENT=../clients/simple_grpc_infer_client
$CLIENT -t 1000 -v >> ${CLIENT_LOG}.c++.grpc_infer 2>&1
if [ $? -eq 0 ]; then
    RET=1
fi
if [ `grep -c "Deadline Exceeded" ${CLIENT_LOG}.c++.grpc_infer` != "1" ]; then
    cat ${CLIENT_LOG}.c++.grpc_infer
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

# Test request timeout in grpc asynchronous inference
CLIENT=../clients/simple_grpc_async_infer_client
$CLIENT -t 1000 -v >> ${CLIENT_LOG}.c++.grpc_async_infer 2>&1
if [ $? -eq 0 ]; then
    RET=1
fi
if [ `grep -c "Deadline Exceeded" ${CLIENT_LOG}.c++.grpc_async_infer` != "1" ]; then
    cat ${CLIENT_LOG}.c++.grpc_async_infer
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

# Test stream timeout in grpc asynchronous streaming inference
CLIENT=../clients/simple_grpc_sequence_stream_infer_client
$CLIENT -t 100 -v >> ${CLIENT_LOG}.c++.grpc_async_stream_infer 2>&1
if [ $? -eq 0 ]; then
    RET=1
fi
if [ `grep -c "Stream has been closed" ${CLIENT_LOG}.c++.grpc_async_stream_infer` != "1" ]; then
    cat ${CLIENT_LOG}.c++.grpc_async_stream_infer
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

# Test request timeout in http synchronous inference
CLIENT=../clients/simple_http_infer_client
$CLIENT -t 1000 -v >> ${CLIENT_LOG}.c++.http_infer 2>&1
if [ $? -eq 0 ]; then
    RET=1
fi
if [ `grep -c "Deadline Exceeded" ${CLIENT_LOG}.c++.http_infer` == "0" ]; then
    cat ${CLIENT_LOG}.c++.http_infer
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi


# Test request timeout in http asynchronous inference
CLIENT=../clients/simple_http_async_infer_client
$CLIENT -t 1000 -v >> ${CLIENT_LOG}.c++.http_async_infer 2>&1
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
for i in simple_grpc_infer_client \
    simple_grpc_async_infer_client \
    simple_grpc_sequence_stream_infer_client \
    simple_http_infer_client \
    simple_http_async_infer_client \
   ; do
   echo "TEST:  $i" >> ${CLIENT_LOG}
   ../clients/$i -v -t 100000000 >> ${CLIENT_LOG} 2>&1
   if [ $? -ne 0 ]; then
        RET=1
    fi
done

echo "TEST:  Python Library" >> ${CLIENT_LOG}

# CASE 3: Python Library

for i in test_grpc_infer \
    test_grpc_async_infer \
    test_grpc_stream_infer \
    test_http_infer \
    test_http_async_infer \
   ; do
    python $CLIENT_TIMEOUT_TEST ClientTimeoutTest.$i >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test $i Failed\n***" >>$CLIENT_LOG
            echo -e "\n***\n*** Test $i Failed\n***"
            RET=1
    else
        check_test_results $CLIENT_LOG 1
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

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat ${CLIENT_LOG}
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
