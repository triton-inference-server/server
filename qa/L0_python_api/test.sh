#!/bin/bash
# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

BINDING_TEST_LOG="./python_binding.log"

RET=0

rm -f $BINDING_TEST_LOG

set +e

python -m pytest --junitxml=test_binding_report.xml test_binding.py > $BINDING_TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $BINDING_TEST_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

API_TEST_LOG="./python_api.log"

python -m pytest --junitxml=test_api_report.xml test_api.py > $API_TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $API_TEST_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi


FRONTEND_TEST_LOG="./python_kserve.log"
# TODO: [DLIS-7735] Run tritonclient.grpc as separate process
# Currently, running tritonclient.grpc with tritonserver in the same process,
# it will non-deterministically abort/crash without being able to be caught by pytest.
# This is because fork() is called by tritonserver on model load,
# which attempts to fork the imported libraries and their internal states,
# and cygrpc (dependency of tritonclient.grpc) does not officially support fork().
# Reference: https://github.com/grpc/grpc/blob/master/doc/fork_support.md
python -m pytest --junitxml=test_kserve.xml test_kserve.py -k "not KServeGrpc" > $FRONTEND_TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $FRONTEND_TEST_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
