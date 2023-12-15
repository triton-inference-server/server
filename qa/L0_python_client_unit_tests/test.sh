#!/bin/bash
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

TEST_LOG="./python_client_unit_tests.log"
PYTHON_CLIENT_UNIT_TESTS_DIR=/opt/tritonserver/qa/python_client_unit_tests/
PYTHON_CLIENT_UNIT_TESTS_CMD="python3 -m unittest discover -v -s $PYTHON_CLIENT_UNIT_TESTS_DIR -t $PYTHON_CLIENT_UNIT_TESTS_DIR"

# DLPack test requires Torch to validate GPU tensor
pip3 install torch

RET=0

rm -f $TEST_LOG

set +e

$PYTHON_CLIENT_UNIT_TESTS_CMD > $TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $TEST_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
