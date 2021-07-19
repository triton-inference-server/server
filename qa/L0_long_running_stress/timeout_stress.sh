#!/bin/bash
# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TIMEOUT_CLIENT_LOG_BASE="./timeout_client"
CLIENT_TIMEOUT_TEST_CPP=../clients/client_timeout_test

# Provide too small a timeout and expect a failure.
# Note, the custom_identity_int32 is configured with a delay of 3 sec.
# Test request timeout in grpc synchronous inference
$CLIENT_TIMEOUT_TEST_CPP -t 1000 -v -i grpc >> ${TIMEOUT_CLIENT_LOG_BASE}.c++.grpc_infer.log 2>&1
if [ $? -eq 0 ]; then
    RET=1
    echo "$RET" > RET.txt
fi
if [ `grep -c "Deadline Exceeded" ${TIMEOUT_CLIENT_LOG_BASE}.c++.grpc_infer.log` != "1" ]; then
    cat ${TIMEOUT_CLIENT_LOG_BASE}.c++.grpc_infer
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
    echo "$RET" > RET.txt
fi

# Test request timeout in grpc asynchronous inference
$CLIENT_TIMEOUT_TEST_CPP -t 1000 -v -i grpc -a >> ${TIMEOUT_CLIENT_LOG_BASE}.c++.grpc_async_infer.log 2>&1
if [ $? -eq 0 ]; then
    RET=1
    echo "$RET" > RET.txt
fi
if [ `grep -c "Deadline Exceeded" ${TIMEOUT_CLIENT_LOG_BASE}.c++.grpc_async_infer.log` != "1" ]; then
    cat ${TIMEOUT_CLIENT_LOG_BASE}.c++.grpc_async_infer
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
    echo "$RET" > RET.txt
fi

# Test stream timeout in grpc asynchronous streaming inference
$CLIENT_TIMEOUT_TEST_CPP -t 1000 -v -i grpc -s >> ${TIMEOUT_CLIENT_LOG_BASE}.c++.grpc_async_stream_infer.log 2>&1
if [ $? -eq 0 ]; then
    RET=1
    echo "$RET" > RET.txt
fi
if [ `grep -c "Stream has been closed" ${TIMEOUT_CLIENT_LOG_BASE}.c++.grpc_async_stream_infer.log` != "1" ]; then
    cat ${TIMEOUT_CLIENT_LOG_BASE}.c++.grpc_async_stream_infer
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
    echo "$RET" > RET.txt
fi

# Test request timeout in http synchronous inference
$CLIENT_TIMEOUT_TEST_CPP -t 1000 -v >> ${TIMEOUT_CLIENT_LOG_BASE}.c++.http_infer.log 2>&1
if [ $? -eq 0 ]; then
    RET=1
    echo "$RET" > RET.txt
fi
if [ `grep -c "Deadline Exceeded" ${TIMEOUT_CLIENT_LOG_BASE}.c++.http_infer.log` == "0" ]; then
    cat ${TIMEOUT_CLIENT_LOG_BASE}.c++.http_infer
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
    echo "$RET" > RET.txt
fi

# Test request timeout in http asynchronous inference
$CLIENT_TIMEOUT_TEST_CPP -t 1000 -v -a >> ${TIMEOUT_CLIENT_LOG_BASE}.c++.http_async_infer.log 2>&1
if [ $? -eq 0 ]; then
    RET=1
    echo "$RET" > RET.txt
fi
if [ `grep -c "Deadline Exceeded" ${TIMEOUT_CLIENT_LOG_BASE}.c++.http_async_infer.log` == "0" ]; then
    cat ${TIMEOUT_CLIENT_LOG_BASE}.c++.http_async_infer
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
    echo "$RET" > RET.txt
fi
