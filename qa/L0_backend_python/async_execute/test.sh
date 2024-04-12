#!/bin/bash
# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

source ../../common/util.sh

RET=0

#
# Test execution overlapping on the same instance
#
rm -rf models && mkdir models
mkdir -p models/async_execute_decouple/1 && \
    cp ../../python_models/async_execute_decouple/model.py models/async_execute_decouple/1 && \
    cp ../../python_models/async_execute_decouple/config.pbtxt models/async_execute_decouple
mkdir -p models/async_execute_decouple_bls/1 && \
    cp ../../python_models/async_execute_decouple_bls/model.py models/async_execute_decouple_bls/1 && \
    cp ../../python_models/async_execute_decouple_bls/config.pbtxt models/async_execute_decouple_bls

TEST_LOG="concurrency_test.log"
SERVER_LOG="concurrency_test.server.log"
SERVER_ARGS="--model-repository=${MODELDIR}/async_execute/models --backend-directory=${BACKEND_DIR} --log-verbose=1"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
SERVER_LOG=$SERVER_LOG python3 -m pytest --junitxml=concurrency_test.report.xml concurrency_test.py > $TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** async execute concurrency test FAILED\n***"
    cat $TEST_LOG
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 1 ]; then
    echo -e "\n***\n*** Async execute test FAILED\n***"
else
    echo -e "\n***\n*** Async execute test Passed\n***"
fi
exit $RET
