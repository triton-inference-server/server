#!/bin/bash
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

SIMPLE_CLIENT=../clients/simple_http_infer_client
SIMPLE_CLIENT_PY=../clients/simple_http_infer_client.py

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=`pwd`/models"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

rm -f *.log

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

RET=0

set +e

# Run with default host header...
$SIMPLE_CLIENT -v >>client_c++.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

if [ `grep -c "localhost:8000" client_c++.log` != "2" ]; then
    echo -e "\n***\n*** Failed. Expected 2 Host:localhost:8000 headers for C++ client\n***"
    RET=1
fi

python $SIMPLE_CLIENT_PY -v >>client_py.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

if [ `grep -c "HTTPSocketPoolResponse status=200" client_py.log` != "3" ]; then
    echo -e "\n***\n*** Failed. Expected 3 Host:HTTPSocketPoolResponse status=200 headers for Python client\n***"
    RET=1
fi

# Run with custom host header...
$SIMPLE_CLIENT -v -H"Host:my_host_" >>client_c++_host.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

if [ `grep -c my_host_ client_c++_host.log` != "1" ]; then
    echo -e "\n***\n*** Failed. Expected 1 Host:my_host_ headers for C++ client\n***"
    RET=1
fi

python $SIMPLE_CLIENT_PY -v -H"Host:my_host_" >>client_py_host.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

if [ `grep -c my_host_ client_py_host.log` != "3" ]; then
    echo -e "\n***\n*** Failed. Expected 3 Host:my_host_ headers for Python client\n***"
    RET=1
fi

# Run with multiple headers...
$SIMPLE_CLIENT -v -H"abc:xyz" -H"123:456" >>client_c++_multi.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

if [ `grep -c "abc: xyz" client_c++_multi.log` != "1" ]; then
    echo -e "\n***\n*** Failed. Expected 1 abc:xyz headers for C++ client\n***"
    RET=1
fi
if [ `grep -c "123: 456" client_c++_multi.log` != "1" ]; then
    echo -e "\n***\n*** Failed. Expected 1 123:456 headers for C++ client\n***"
    RET=1
fi

python $SIMPLE_CLIENT_PY -v -H"abc:xyz" -H"123:456" >>client_py_multi.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

if [ `grep -c "'abc': 'xyz'" client_py_multi.log` != "3" ]; then
    echo -e "\n***\n*** Failed. Expected 3 abc:xyz headers for Python client\n***"
    RET=1
fi
if [ `grep -c "'123': '456'" client_py_multi.log` != "3" ]; then
    echo -e "\n***\n*** Failed. Expected 3 123:456 headers for Python client\n***"
    RET=1
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
