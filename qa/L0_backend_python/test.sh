#!/bin/bash
# Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# On windows the paths invoked by the script (running in WSL) must use
# /mnt/c when needed but the paths on the tritonserver command-line
# must be C:/ style.
export TEST_WINDOWS=0
if [[ -v WSL_DISTRO_NAME ]] || [[ -v MSYSTEM ]]; then
    export DATADIR=${DATADIR:="/c/data/inferenceserver/${REPO_VERSION}"}
    export TRITON_DIR=${TRITON_DIR:=c:/tritonserver}
    # This will run in WSL, but Triton will run in windows, so environment
    # variables meant for loaded models must be exported using WSLENV.
    # The /w flag indicates the value should only be included when invoking
    # Win32 from WSL.
    export WSLENV=TRITON_DIR
    export SERVER=${SERVER:=c:/tritonserver/bin/tritonserver.exe}
    export BACKEND_DIR=${BACKEND_DIR:=c:/tritonserver/backends}
    export MODELDIR=${MODELDIR:=c:/}
    export TEST_WINDOWS=1
else
    export DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
    export TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
    export SERVER=${TRITON_DIR}/bin/tritonserver
    export BACKEND_DIR=${TRITON_DIR}/backends
    export MODELDIR=${MODELDIR:=`pwd`}
fi
export REPO_VERSION=$REPO_VERSION
export TEST_JETSON=${TEST_JETSON:=0}
export CUDA_VISIBLE_DEVICES=0
export PYTHON_ENV_VERSION=${PYTHON_ENV_VERSION:="12"}
export PYTHON_BACKEND_REPO_TAG=$PYTHON_BACKEND_REPO_TAG

BASE_SERVER_ARGS="--model-repository=${MODELDIR}/models --backend-directory=${BACKEND_DIR} --log-verbose=1"
# Set the default byte size to 5MBs to avoid going out of shared memory. The
# environment that this job runs on has only 1GB of shared-memory available.
SERVER_ARGS="$BASE_SERVER_ARGS --backend-config=python,shm-default-byte-size=5242880"

CLIENT_PY=./python_test.py
CLIENT_LOG="./client.log"
TEST_RESULT_FILE='test_results.txt'
SERVER_LOG="./inference_server.log"
source ../common/util.sh
source ./common.sh

rm -fr *.log ./models

python3 --version | grep "3.12" > /dev/null
if [ $? -ne 0 ]; then
    echo -e "Expecting Python default version to be: Python 3.12 but actual version is $(python3 --version)"
    exit 1
fi

(bash -ex setup_python_enviroment.sh)

python3 --version | grep "3.${PYTHON_ENV_VERSION}" > /dev/null
if [ $? -ne 0 ]; then
    echo -e "Expecting Python version to be: Python 3.${PYTHON_ENV_VERSION} but actual version is $(python3 --version)"
    exit 1
fi

mkdir -p models/identity_fp32/1/
cp ../python_models/identity_fp32/model.py ./models/identity_fp32/1/model.py
cp ../python_models/identity_fp32/config.pbtxt ./models/identity_fp32/config.pbtxt
mkdir -p models/identity_bf16/1/
cp ../python_models/identity_bf16/model.py ./models/identity_bf16/1/model.py
cp ../python_models/identity_bf16/config.pbtxt ./models/identity_bf16/config.pbtxt
RET=0

cp -r ./models/identity_fp32 ./models/identity_uint8
(cd models/identity_uint8 && \
          sed -i "s/^name:.*/name: \"identity_uint8\"/" config.pbtxt && \
          sed -i "s/TYPE_FP32/TYPE_UINT8/g" config.pbtxt && \
          sed -i "s/^max_batch_size:.*/max_batch_size: 8/" config.pbtxt && \
          echo "dynamic_batching { preferred_batch_size: [8], max_queue_delay_microseconds: 12000000 }" >> config.pbtxt)

cp -r ./models/identity_fp32 ./models/identity_uint8_nobatch
(cd models/identity_uint8_nobatch && \
          sed -i "s/^name:.*/name: \"identity_uint8_nobatch\"/" config.pbtxt && \
          sed -i "s/TYPE_FP32/TYPE_UINT8/g" config.pbtxt && \
          sed -i "s/^max_batch_size:.*//" config.pbtxt >> config.pbtxt)

cp -r ./models/identity_fp32 ./models/identity_uint32
(cd models/identity_uint32 && \
          sed -i "s/^name:.*/name: \"identity_uint32\"/" config.pbtxt && \
          sed -i "s/TYPE_FP32/TYPE_UINT32/g" config.pbtxt)

cp -r ./models/identity_fp32 ./models/identity_bool
(cd models/identity_bool && \
          sed -i "s/^name:.*/name: \"identity_bool\"/" config.pbtxt && \
          sed -i "s/TYPE_FP32/TYPE_BOOL/g" config.pbtxt)

# Test models with `default_model_filename` variable set.
cp -r ./models/identity_fp32 ./models/default_model_name
mv ./models/default_model_name/1/model.py ./models/default_model_name/1/mymodel.py
(cd models/default_model_name && \
    sed -i "s/^name:.*/name: \"default_model_name\"/" config.pbtxt && \
    echo "default_model_filename: \"mymodel.py\"" >> config.pbtxt )

mkdir -p models/pytorch_fp32_fp32/1/
    cp -r ../python_models/pytorch_fp32_fp32/model.py ./models/pytorch_fp32_fp32/1/
    cp ../python_models/pytorch_fp32_fp32/config.pbtxt ./models/pytorch_fp32_fp32/
    (cd models/pytorch_fp32_fp32 && \
            sed -i "s/^name:.*/name: \"pytorch_fp32_fp32\"/" config.pbtxt)

mkdir -p models/delayed_model/1/
cp -r ../python_models/delayed_model/model.py ./models/delayed_model/1/
cp ../python_models/delayed_model/config.pbtxt ./models/delayed_model/
mkdir -p models/init_args/1/
cp ../python_models/init_args/model.py ./models/init_args/1/
cp ../python_models/init_args/config.pbtxt ./models/init_args/
sed -i "s|TRITON_DIR_PATH|${TRITON_DIR}|" ./models/init_args/config.pbtxt


mkdir -p models/optional/1/
cp ../python_models/optional/model.py ./models/optional/1/
cp ../python_models/optional/config.pbtxt ./models/optional/

mkdir -p models/non_contiguous/1/
cp ../python_models/non_contiguous/model.py ./models/non_contiguous/1/
cp ../python_models/non_contiguous/config.pbtxt ./models/non_contiguous/config.pbtxt

# Unicode Characters
mkdir -p models/string/1/
cp ../python_models/string/model.py ./models/string/1/
cp ../python_models/string/config.pbtxt ./models/string

# More string tests
mkdir -p models/string_fixed/1/
cp ../python_models/string_fixed/model.py ./models/string_fixed/1/
cp ../python_models/string_fixed/config.pbtxt ./models/string_fixed

mkdir -p models/dlpack_identity/1/
cp ../python_models/dlpack_identity/model.py ./models/dlpack_identity/1/
cp ../python_models/dlpack_identity/config.pbtxt ./models/dlpack_identity


if [ "$TEST_JETSON" == "0" ] && [[ ${TEST_WINDOWS} == 0 ]]; then
  pip3 install torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
else
  # GPU tensor tests are disabled on jetson
  pip3 install torch==2.3.1 -f https://download.pytorch.org/whl/torch_stable.html
fi

pip3 install pytest requests virtualenv

# Set up the tests to execute.
declare -A subtest_properties
subtest_properties["shared_memory"]=""
subtest_properties["lifecycle"]=""
subtest_properties["argument_validation"]=""
subtest_properties["logging"]=""
subtest_properties["custom_metrics"]=""

# Add tests depending on which environment is being run. 

# If not running in igpu mode add these
# Disable env test for Jetson since cloud storage repos are not supported
# Disable ensemble, io and bls tests for Jetson since GPU Tensors are not supported
# Disable variants test for Jetson since already built without GPU Tensor support
# Disable decoupled test because it uses GPU tensors
if [ "$TEST_JETSON" == "0" ]; then
    subtest_properties["ensemble"]=""
    subtest_properties["bls"]=""
    subtest_properties["decoupled"]=""
    subtest_properties["response_sender"]=""
    subtest_properties["env"]="locale"
fi

# If not running on windows add these 
# [DLIS-6093] Disable variants test for Windows since tests are not executed in docker container (cannot apt update/install)
# [DLIS-5970] Disable io tests for Windows since GPU Tensors are not supported
# [DLIS-6124] Disable restart test for Windows since it requires more investigation
# [DLIS-6122] Disable model_control & request_rescheduling tests for Windows since they require load/unload
# [DLIS-6123] Disable examples test for Windows since it requires updates to the example clients
if [[ ${TEST_WINDOWS} == 0 ]]; then
    subtest_properties["variants"]=""
    subtest_properties["io"]=""
    subtest_properties["python_based_backends"]=""
    subtest_properties["async_execute"]=""
    subtest_properties["model_control"]=""
    subtest_properties["examples"]=""
    subtest_properties["request_rescheduling"]=""
fi  

ALL_SUBTESTS=$(echo "${!subtest_properties[@]}")
if [[ -n "${SUBTESTS}" ]]; then
    for subtest in $(echo "${!subtest_properties[@]}"); do 
        if [[ ! "${SUBTESTS}" =~ "${subtest}" ]]; then
            unset "subtest_properties[${subtest}]"
        fi
    done
fi
ALL_SUBTESTS=$(echo "${!subtest_properties[@]}")

echo "Executing the following subtests: "
for subtest in $(echo "${!subtest_properties[@]}"); do
    echo "  ${subtest}: ${subtest_properties[${subtest}]}"
done

if [[ "${ALL_SUBTETS}" =~ "shared_memory" ]]; then
    bash -ex test_shared_memory.sh
fi

for TEST in ${ALL_SUBTESTS}; do 
    if [[ "${TEST}" == "env" ]]; then 
        continue
    fi

    # Run each subtest in a separate virtual environment to avoid conflicts
    # between dependencies.
    setup_virtualenv

    set +e
    (cd ${TEST} && PROPERTIES="${subtest_properties["${TEST}"]}" bash -ex test.sh)
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Subtest ${TEST} FAILED"
        RET=$EXIT_CODE

        # In bls test, it is allowed to fail with a strict memory leak of 480 bytes with exit code '123'.
        # Propagate the exit code to make sure it's not overwritten by other tests.
        if [[ ${TEST} == "bls" ]]  && [[ $EXIT_CODE -ne 1 ]] ; then
            BLS_RET=$RET
        fi
    fi
    set -e

    deactivate_virtualenv
done

if [[ "${ALL_SUBTESTS}" =~ "env" ]]; then
    # In 'env' test we use miniconda for dependency management. No need to run
    # the test in a virtual environment.
    set +e
    (cd env && PROPERTIES="${subtest_properties["env"]}" bash -ex test.sh)
    if [ $? -ne 0 ]; then
        echo "Subtest env FAILED"
        RET=1
    fi
    set -e
fi

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

# Exit with RET if it is 1, meaning that the test failed.
# Otherwise, exit with BLS_RET if it is set, meaning that the known memory leak is captured.
if [ $RET -eq 1 ]; then
    exit $RET
else
    if [ -z "$BLS_RET" ]; then
        exit $RET
    else
        exit $BLS_RET
    fi
fi
