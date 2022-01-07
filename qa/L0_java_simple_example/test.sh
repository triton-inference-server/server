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

# Set up test files
git clone https://github.com/bytedeco/javacpp-presets.git
cp -f `pwd`/pom.xml `pwd`/javacpp-presets/tritonserver/samples/pom.xml

MODEL_REPO=`pwd`/../L0_simple_ensemble/models
SAMPLES_REPO=`pwd`/javacpp-presets/tritonserver/samples
BASE_COMMAND="mvn clean compile -f $SAMPLES_REPO exec:java -Djavacpp.platform=linux-x86_64"
source ../common/util.sh

rm -f *.log
RET=0

# Run with default settings
$BASE_COMMAND -Dexec.args="-r $MODEL_REPO" >>client.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

if [ `grep -c "1 - 1 = 0" client.log` != "18" ]; then
    echo -e "\n***\n*** Failed. Expected 18 '1 - 1 = 0'\n***"
    RET=1
fi

# Run with verbose logging
$BASE_COMMAND -Dexec.args="-r $MODEL_REPO -v" >>client.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

if [ `grep -c "Server side auto-completed config" client.log` != "2" ]; then
    echo -e "\n***\n*** Failed. Expected 'Server side auto-completed config'\n***"
    RET=1
fi

# Run with memory set to system
$BASE_COMMAND -Dexec.args="-r $MODEL_REPO -m system" >>client.log 2>&1
if [ $? -ne 0 ]; then
    RET=1
fi

if [ `grep -c "OUTPUT0 is stored in system memory" client.log` != "9" ]; then
    echo -e "\n***\n*** Failed. Expected 9 'OUTPUT0 is stored in system memory'\n***"
    RET=1
fi

# Run with FP32 datatype
sed -i 's/TYPE_INT32/TYPE_FP32/g' $MODEL_REPO/simple/config.pbtxt
sed -i 's/TYPE_INT32/TYPE_FP32/g' $MODEL_REPO/ensemble_add_sub_int32_int32_int32/config.pbtxt
$BASE_COMMAND -Dexec.args="-r $MODEL_REPO -v" >>client.log 2>&1
sed -i 's/TYPE_FP32/TYPE_INT32/g' $MODEL_REPO/simple/config.pbtxt
sed -i 's/TYPE_FP32/TYPE_INT32/g' $MODEL_REPO/ensemble_add_sub_int32_int32_int32/config.pbtxt
if [ $? -ne 0 ]; then
    RET=1
fi

if [ `grep -c "data_type: TYPE_FP32" client.log` != "8" ]; then
    echo -e "\n***\n*** Failed. Expected 4 'data_type: TYPE_FP32'\n***"
    RET=1
fi

# Run ensemble
sed -i 's/"simple"/"ensemble_add_sub_int32_int32_int32"/g' $SAMPLES_REPO/Simple.java
$BASE_COMMAND -Dexec.args="-r $MODEL_REPO -v" >>client.log 2>&1
sed -i 's/"ensemble_add_sub_int32_int32_int32"/"simple"/g' $SAMPLES_REPO/Simple.java
if [ $? -ne 0 ]; then
    RET=1
fi

if [ `grep -c "request id: my_request_id, model: ensemble_add_sub_int32_int32_int32" client.log` != "3" ]; then
    echo -e "\n***\n*** Failed. Expected 3 'request id: my_request_id, model: ensemble_add_sub_int32_int32_int32'\n***"
    RET=1
fi

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
