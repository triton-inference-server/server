#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Set up test files based on installation instructions
# https://github.com/bytedeco/javacpp-presets/blob/master/tritonserver/README.md
rm -r javacpp-presets
git clone https://github.com/bytedeco/javacpp-presets.git
cd javacpp-presets
mvn clean install --projects .,tritonserver
mvn clean install -f platform --projects ../tritonserver/platform -Djavacpp.platform.host
cd ..

CLIENT_LOG="client_cpu_only.log"
MODEL_REPO=`pwd`/models
SAMPLES_REPO=`pwd`/javacpp-presets/tritonserver/samples
BASE_COMMAND="mvn clean compile -f $SAMPLES_REPO exec:java -Djavacpp.platform=linux-x86_64"
source ../common/util.sh

# Create local model repository
rm -r models
cp -r `pwd`/../L0_simple_ensemble/models .
mkdir ${MODEL_REPO}/ensemble_add_sub_int32_int32_int32/1

rm -f *.log
RET=0

function run_cpu_tests() {
    # Run with default settings
    $BASE_COMMAND -Dexec.args="-r $MODEL_REPO" >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    if [ `grep -c "1 - 1 = 0" ${CLIENT_LOG}` != "18" ]; then
        echo -e "\n***\n*** Failed. Expected 18 '1 - 1 = 0'\n***"
        RET=1
    fi

    # Run with verbose logging
    $BASE_COMMAND -Dexec.args="-r $MODEL_REPO -v" >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    if [ `grep -c "Server side auto-completed config" ${CLIENT_LOG}` != "2" ]; then
        echo -e "\n***\n*** Failed. Expected 'Server side auto-completed config'\n***"
        RET=1
    fi

    # Run with memory set to system
    $BASE_COMMAND -Dexec.args="-r $MODEL_REPO -m system" >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    if [ `grep -c "OUTPUT0 is stored in system memory" ${CLIENT_LOG}` != "9" ]; then
        echo -e "\n***\n*** Failed. Expected 9 'OUTPUT0 is stored in system memory'\n***"
        RET=1
    fi

    # Run with FP32 datatype
    sed -i 's/TYPE_INT32/TYPE_FP32/g' $MODEL_REPO/simple/config.pbtxt
    sed -i 's/TYPE_INT32/TYPE_FP32/g' $MODEL_REPO/ensemble_add_sub_int32_int32_int32/config.pbtxt
    $BASE_COMMAND -Dexec.args="-r $MODEL_REPO -v" >>$CLIENT_LOG 2>&1
    sed -i 's/TYPE_FP32/TYPE_INT32/g' $MODEL_REPO/simple/config.pbtxt
    sed -i 's/TYPE_FP32/TYPE_INT32/g' $MODEL_REPO/ensemble_add_sub_int32_int32_int32/config.pbtxt
    if [ $? -ne 0 ]; then
        RET=1
    fi

    if [ `grep -c "data_type: TYPE_FP32" ${CLIENT_LOG}` != "8" ]; then
        echo -e "\n***\n*** Failed. Expected 4 'data_type: TYPE_FP32'\n***"
        RET=1
    fi
}

# Run tests on CPU-only simple example
echo -e "\nRunning Simple CPU-Only Tests\n"

sed -i 's/Simple/SimpleCPUOnly/g' $SAMPLES_REPO/pom.xml
run_cpu_tests

# Run ensemble
sed -i 's/"simple"/"ensemble_add_sub_int32_int32_int32"/g' $SAMPLES_REPO/SimpleCPUOnly.java
cat $SAMPLES_REPO/pom.xml >>$CLIENT_LOG 2>&1
$BASE_COMMAND -Dexec.args="-r $MODEL_REPO -v" >>$CLIENT_LOG 2>&1
sed -i 's/"ensemble_add_sub_int32_int32_int32"/"simple"/g' $SAMPLES_REPO/SimpleCPUOnly.java
if [ $? -ne 0 ]; then
    RET=1
fi

if [ `grep -c "request id: my_request_id, model: ensemble_add_sub_int32_int32_int32" ${CLIENT_LOG}` != "3" ]; then
    echo -e "\n***\n*** Failed. Expected 3 'request id: my_request_id, model: ensemble_add_sub_int32_int32_int32'\n***"
    RET=1
fi

sed -i 's/SimpleCPUOnly/Simple/g' $SAMPLES_REPO/pom.xml

# Run tests on full simple example
echo -e "\nRunning Simple Tests\n"
CLIENT_LOG="client.log"
run_cpu_tests

INDEX=1
for MEMORY_TYPE in pinned gpu; do
    $BASE_COMMAND -Dexec.args="-r $MODEL_REPO -m $MEMORY_TYPE" >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    if [ `grep -ic "OUTPUT0 is stored in ${MEMORY_TYPE} memory" ${CLIENT_LOG}` != "2" ]; then
        echo -e "\n***\n*** Failed. Expected 2 'OUTPUT0 is stored in ${MEMORY_TYPE} memory'\n***"
        RET=1
    fi

    sed -i 's/"simple"/"ensemble_add_sub_int32_int32_int32"/g' $SAMPLES_REPO/Simple.java
    $BASE_COMMAND -Dexec.args="-r $MODEL_REPO -v -m $MEMORY_TYPE" >>$CLIENT_LOG 2>&1
    sed -i 's/"ensemble_add_sub_int32_int32_int32"/"simple"/g' $SAMPLES_REPO/Simple.java
    if [ $? -ne 0 ]; then
        RET=1
    fi

    if [ `grep -c "request id: my_request_id, model: ensemble_add_sub_int32_int32_int32" ${CLIENT_LOG}` != $((INDEX*2)) ]; then
        echo -e "\n***\n*** Failed. Expected $((INDEX*2)) 'request id: my_request_id, model: ensemble_add_sub_int32_int32_int32'\n***"
        RET=1
    fi
    let INDEX++
done


if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
