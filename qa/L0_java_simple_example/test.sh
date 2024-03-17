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
REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

JAVACPP_BRANCH=${JAVACPP_BRANCH:="https://github.com/bytedeco/javacpp-presets.git"}
JAVACPP_BRANCH_TAG=${JAVACPP_BRANCH_TAG:="master"}
set -e
git clone --single-branch --depth=1 -b ${TRITON_CLIENT_REPO_TAG} https://github.com/triton-inference-server/client.git
source client/src/java-api-bindings/scripts/install_dependencies_and_build.sh -b $PWD --javacpp-branch ${JAVACPP_BRANCH} --javacpp-tag ${JAVACPP_BRANCH_TAG} --keep-build-dependencies
cd ..

CLIENT_LOG="client_cpu_only.log"
DATADIR=/data/inferenceserver/${REPO_VERSION}/qa_model_repository
MODEL_REPO=`pwd`/models

SAMPLES_REPO=`pwd`/javacpp-presets/tritonserver/samples/simple
BASE_COMMAND="mvn clean compile -f $SAMPLES_REPO exec:java -Djavacpp.platform=linux-x86_64"
source ../common/util.sh


rm -f *.log
RET=0

function run_cpu_tests_int32() {
    # Create local model repository
    set +e
    rm -r ${MODEL_REPO}
    cp -r `pwd`/../L0_simple_ensemble/models .
    mkdir ${MODEL_REPO}/ensemble_add_sub_int32_int32_int32/1
    set -e

    # Run with default settings
    $BASE_COMMAND -Dexec.args="-r $MODEL_REPO" >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "Failed to run: ${BASE_COMMAND} -Dexec.args=\"-r ${MODEL_REPO}\""
        RET=1
    fi

    if [ `grep -c "1 - 1 = 0" ${CLIENT_LOG}` != "18" ]; then
        echo -e "\n***\n*** Failed. Expected 18 '1 - 1 = 0'\n***"
        RET=1
    fi

    # Run with verbose logging
    $BASE_COMMAND -Dexec.args="-r $MODEL_REPO -v" >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "Failed to run: ${BASE_COMMAND} -Dexec.args=\"-r ${MODEL_REPO} -v\""
        RET=1
    fi

    if [ `grep -c "Server side auto-completed config" ${CLIENT_LOG}` != "2" ]; then
        echo -e "\n***\n*** Failed. Expected 'Server side auto-completed config'\n***"
        RET=1
    fi

    # Run with memory set to system
    $BASE_COMMAND -Dexec.args="-r $MODEL_REPO -m system" >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "Failed to run: ${BASE_COMMAND} -Dexec.args=\"-r ${MODEL_REPO} -m system\""
        RET=1
    fi

    if [ `grep -c "OUTPUT0 is stored in system memory" ${CLIENT_LOG}` != "9" ]; then
        echo -e "\n***\n*** Failed. Expected 9 'OUTPUT0 is stored in system memory'\n***"
        RET=1
    fi

}

function run_cpu_tests_fp32() {
    for trial in graphdef savedmodel; do
        full=${trial}_float32_float32_float32
        set +e
        rm -rf ${MODEL_REPO}
        mkdir -p ${MODEL_REPO}/simple/1 && \
            cp -r $DATADIR/${full}/1/* ${MODEL_REPO}/simple/1/. && \
            cp $DATADIR/${full}/config.pbtxt ${MODEL_REPO}/simple/. && \
            (cd ${MODEL_REPO}/simple && \
                    sed -i "s/^name:.*/name: \"simple\"/" config.pbtxt && \
                    sed -i "s/label_filename:.*//" config.pbtxt)


        # No memory type enforcement
        $BASE_COMMAND -Dexec.args="-r $MODEL_REPO -v" >>$CLIENT_LOG.$full.log 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG.$full.log
            echo -e "Failed to run: ${BASE_COMMAND} -Dexec.args=\"-r ${MODEL_REPO} -v\" for ${full}"
            RET=1
        fi

        # Enforce I/O to be in specific memory type
        for MEM_TYPE in system; do
            $BASE_COMMAND -Dexec.args="-r $MODEL_REPO -m ${MEM_TYPE}" >>$CLIENT_LOG.$full.${MEM_TYPE}.log 2>&1
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG.$full.$MEM_TYPE.log
                echo -e "Failed to run: ${BASE_COMMAND} -Dexec.args=\"-r ${MODEL_REPO} -v -m ${MEM_TYPE}\" for ${full}"
                RET=1
            fi
        done
    done
    set -e
}


# Run ensemble
function run_ensemble_tests() {
    set +e
    rm -r ${MODEL_REPO}
    cp -r `pwd`/../L0_simple_ensemble/models .
    mkdir -p ${MODEL_REPO}/ensemble_add_sub_int32_int32_int32/1
    sed -i 's/"simple"/"ensemble_add_sub_int32_int32_int32"/g' $SAMPLES_REPO/Simple.java
    cat $SAMPLES_REPO/pom.xml >>$CLIENT_LOG 2>&1
    set -e

    $BASE_COMMAND -Dexec.args="-r $MODEL_REPO -v" >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "Failed to run ensemble model: ${BASE_COMMAND} -Dexec.args=\"-r ${MODEL_REPO} -v\""
        RET=1
    fi
    sed -i 's/"ensemble_add_sub_int32_int32_int32"/"simple"/g' $SAMPLES_REPO/Simple.java

    if [ `grep -c "request id: my_request_id, model: ensemble_add_sub_int32_int32_int32" ${CLIENT_LOG}` != "3" ]; then
        echo -e "\n***\n*** Failed. Expected 3 'request id: my_request_id, model: ensemble_add_sub_int32_int32_int32'\n***"
        RET=1
    fi
}

# Run tests on simple example
echo -e "\nRunning Simple Tests\n"

run_cpu_tests_fp32
run_cpu_tests_int32
run_ensemble_tests

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
