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
JAVACPP_BRANCH=${JAVACPP_BRANCH:="https://github.com/bytedeco/javacpp-presets.git"}
JAVACPP_BRANCH_TAG=${JAVACPP_BRANCH_TAG:="master"}
set -e
git clone --single-branch --depth=1 -b ${TRITON_CLIENT_REPO_TAG} https://github.com/triton-inference-server/client.git
source client/src/java-api-bindings/scripts/install_dependencies_and_build.sh -b $PWD --javacpp-branch ${JAVACPP_BRANCH} --javacpp-tag ${JAVACPP_BRANCH_TAG} --keep-build-dependencies
cd ..

export MAVEN_OPTS="-XX:MaxGCPauseMillis=40"
MODEL_REPO=`pwd`/models
SAMPLES_REPO=`pwd`/javacpp-presets/tritonserver/samples/simple
BASE_COMMAND="mvn clean compile -f $SAMPLES_REPO exec:java -Djavacpp.platform=linux-x86_64"
source ../common/util.sh

# Create local model repository
rm -rf ${MODEL_REPO}
mkdir ${MODEL_REPO}
cp -r `pwd`/../L0_simple_ensemble/models/simple ${MODEL_REPO}/.

cp MemoryGrowthTest.java $SAMPLES_REPO
sed -i 's/Simple/MemoryGrowthTest/g' $SAMPLES_REPO/pom.xml

rm -f *.log
RET=0


# Sanity test: check accuracy
ITERS=200000

LOG_IDX=0
CLIENT_LOG="./client_$LOG_IDX.log"

echo -e "\nRunning Sanity Test (accuracy checking)\n"
$BASE_COMMAND -Dexec.args="-r $MODEL_REPO -i $ITERS" >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to run sanity test to complete\n***"
    RET=1
fi

if [ `grep -c "Memory growth test passed" $CLIENT_LOG` != "1" ]; then
    echo -e "\n***\n*** Failed. Expected 1 'Memory growth test passed' in $CLIENT_LOG\n***"
    cat $CLIENT_LOG
    RET=1
fi

LOG_IDX=$((LOG_IDX+1))
CLIENT_LOG="./client_$LOG_IDX.log"

# Longer-running memory growth test
ITERS=1000000
MAX_MEM_GROWTH_MB=10
if [ "$TRITON_PERF_LONG" == 1 ]; then
    # ~1 day
    ITERS=150000000
    MAX_MEM_GROWTH_MB=25
fi

echo -e "\nRunning Memory Growth Test, $ITERS Iterations\n"
$BASE_COMMAND -Dexec.args="-r $MODEL_REPO -c -i $ITERS --max-growth $MAX_MEM_GROWTH_MB" >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to run memory growth test to complete\n***"
    RET=1
fi

if [ `grep -c "Memory growth test passed" $CLIENT_LOG` != "1" ]; then
    echo -e "\n***\n*** Failed. Expected 1 'Memory growth test passed' in $CLIENT_LOG\n***"
    cat $CLIENT_LOG
    RET=1
fi

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
