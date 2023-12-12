#!/bin/bash
# Copyright 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
export REPO_VERSION=$REPO_VERSION
export DATADIR=$DATADIR

export TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
export BACKEND_DIR=${TRITON_DIR}/backends
export TEST_JETSON=${TEST_JETSON:=0}
export CUDA_VISIBLE_DEVICES=0
export PYTHON_ENV_VERSION=${PYTHON_ENV_VERSION:="10"}

BASE_SERVER_ARGS="--model-repository=`pwd`/models --backend-directory=${BACKEND_DIR} --log-verbose=1"
# Set the default byte size to 5MBs to avoid going out of shared memory. The
# environment that this job runs on has only 1GB of shared-memory available.
SERVER_ARGS="$BASE_SERVER_ARGS --backend-config=python,shm-default-byte-size=5242880"

PYTHON_BACKEND_BRANCH=$PYTHON_BACKEND_REPO_TAG
CLIENT_PY=./python_test.py
CLIENT_LOG="./client.log"
EXPECTED_NUM_TESTS="11"
TEST_RESULT_FILE='test_results.txt'
SERVER_LOG="./inference_server.log"
source ../common/util.sh
source ./common.sh

rm -fr *.log ./models

python3 --version | grep "3.10" > /dev/null
if [ $? -ne 0 ]; then
    echo -e "Expecting Python default version to be: Python 3.10 but actual version is $(python3 --version)"
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

if [ "$TEST_JETSON" == "0" ]; then
  pip3 install torch==1.13.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
else
  pip3 install torch==1.13.0 -f https://download.pytorch.org/whl/torch_stable.html
  # GPU tensor tests are disabled on jetson
  EXPECTED_NUM_TESTS=9
fi

prev_num_pages=`get_shm_pages`
run_server
if [ "$SERVER_PID" == "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    exit 1
fi

set +e
python3 $CLIENT_PY >> $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

current_num_pages=`get_shm_pages`
if [ $current_num_pages -ne $prev_num_pages ]; then
    ls /dev/shm
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed. Shared memory pages where not cleaned properly.
Shared memory pages before starting triton equals to $prev_num_pages
and shared memory pages after starting triton equals to $current_num_pages \n***"
    RET=1
fi

prev_num_pages=`get_shm_pages`
# Triton non-graceful exit
run_server
if [ "$SERVER_PID" == "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    exit 1
fi

sleep 5

triton_procs=`pgrep --parent $SERVER_PID`

set +e

# Trigger non-graceful termination of Triton
kill -9 $SERVER_PID

# Wait 10 seconds so that Python stub can detect non-graceful exit
sleep 10

for triton_proc in $triton_procs; do
    kill -0 $triton_proc > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Python backend non-graceful exit test failed \n***"
        RET=1
        break
    fi
done
set -e

#
# Test KIND_GPU
# Disable env test for Jetson since GPU Tensors are not supported
if [ "$TEST_JETSON" == "0" ]; then
  rm -rf models/
  mkdir -p models/add_sub_gpu/1/
  cp ../python_models/add_sub/model.py ./models/add_sub_gpu/1/
  cp ../python_models/add_sub_gpu/config.pbtxt ./models/add_sub_gpu/

  prev_num_pages=`get_shm_pages`
  run_server
  if [ "$SERVER_PID" == "0" ]; then
      cat $SERVER_LOG
      echo -e "\n***\n*** Failed to start $SERVER\n***"
      exit 1
  fi

  if [ $? -ne 0 ]; then
      cat $SERVER_LOG
      echo -e "\n***\n*** KIND_GPU model test failed \n***"
      RET=1
  fi

  kill $SERVER_PID
  wait $SERVER_PID

  current_num_pages=`get_shm_pages`
  if [ $current_num_pages -ne $prev_num_pages ]; then
      cat $CLIENT_LOG
      ls /dev/shm
      echo -e "\n***\n*** Test Failed. Shared memory pages where not cleaned properly.
  Shared memory pages before starting triton equals to $prev_num_pages
  and shared memory pages after starting triton equals to $current_num_pages \n***"
      exit 1
  fi
fi

# Test Multi file models
rm -rf models/
mkdir -p models/multi_file/1/
cp ../python_models/multi_file/*.py ./models/multi_file/1/
cp ../python_models/identity_fp32/config.pbtxt ./models/multi_file/
(cd models/multi_file && \
          sed -i "s/^name:.*/name: \"multi_file\"/" config.pbtxt)

prev_num_pages=`get_shm_pages`
run_server
if [ "$SERVER_PID" == "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    exit 1
fi

if [ $? -ne 0 ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** multi-file model test failed \n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

current_num_pages=`get_shm_pages`
if [ $current_num_pages -ne $prev_num_pages ]; then
    cat $SERVER_LOG
    ls /dev/shm
    echo -e "\n***\n*** Test Failed. Shared memory pages where not cleaned properly.
Shared memory pages before starting triton equals to $prev_num_pages
and shared memory pages after starting triton equals to $current_num_pages \n***"
    exit 1
fi

# Test environment variable propagation
rm -rf models/
mkdir -p models/model_env/1/
cp ../python_models/model_env/model.py ./models/model_env/1/
cp ../python_models/model_env/config.pbtxt ./models/model_env/

export MY_ENV="MY_ENV"
prev_num_pages=`get_shm_pages`
run_server
if [ "$SERVER_PID" == "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    echo -e "\n***\n*** Environment variable test failed \n***"
    exit 1
fi

kill $SERVER_PID
wait $SERVER_PID

current_num_pages=`get_shm_pages`
if [ $current_num_pages -ne $prev_num_pages ]; then
    cat $CLIENT_LOG
    ls /dev/shm
    echo -e "\n***\n*** Test Failed. Shared memory pages where not cleaned properly.
Shared memory pages before starting triton equals to $prev_num_pages
and shared memory pages after starting triton equals to $current_num_pages \n***"
    exit 1
fi

rm -fr ./models
mkdir -p models/identity_fp32/1/
cp ../python_models/identity_fp32/model.py ./models/identity_fp32/1/model.py
cp ../python_models/identity_fp32/config.pbtxt ./models/identity_fp32/config.pbtxt

shm_default_byte_size=$((1024*1024*4))
SERVER_ARGS="$BASE_SERVER_ARGS --backend-config=python,shm-default-byte-size=$shm_default_byte_size"

run_server
if [ "$SERVER_PID" == "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    exit 1
fi

for shm_page in `ls /dev/shm/`; do
    if [[ $shm_page !=  triton_python_backend_shm* ]]; then
        continue
    fi
    page_size=`ls -l /dev/shm/$shm_page 2>&1 | awk '{print $5}'`
    if [ $page_size -ne $shm_default_byte_size ]; then
        echo -e "Shared memory region size is not equal to
$shm_default_byte_size for page $shm_page. Region size is
$page_size."
        RET=1
    fi
done

kill $SERVER_PID
wait $SERVER_PID

# Test model getting killed during initialization
rm -fr ./models
mkdir -p models/init_exit/1/
cp ../python_models/init_exit/model.py ./models/init_exit/1/model.py
cp ../python_models/init_exit/config.pbtxt ./models/init_exit/config.pbtxt

ERROR_MESSAGE="Stub process 'init_exit_0_0' is not healthy."

prev_num_pages=`get_shm_pages`
run_server
if [ "$SERVER_PID" != "0" ]; then
    echo -e "*** FAILED: unexpected success starting $SERVER" >> $CLIENT_LOG
    RET=1
    kill $SERVER_PID
    wait $SERVER_PID
else
    if grep "$ERROR_MESSAGE" $SERVER_LOG; then
        echo -e "Found \"$ERROR_MESSAGE\"" >> $CLIENT_LOG
    else
        echo $CLIENT_LOG
        echo -e "Not found \"$ERROR_MESSAGE\"" >> $CLIENT_LOG
        RET=1
    fi
fi

current_num_pages=`get_shm_pages`
if [ $current_num_pages -ne $prev_num_pages ]; then
    cat $SERVER_LOG
    ls /dev/shm
    echo -e "\n***\n*** Test Failed. Shared memory pages where not cleaned properly.
Shared memory pages before starting triton equals to $prev_num_pages
and shared memory pages after starting triton equals to $current_num_pages \n***"
    exit 1
fi

# Disable env test for Jetson since cloud storage repos are not supported
# Disable ensemble, io and bls tests for Jetson since GPU Tensors are not supported
# Disable variants test for Jetson since already built without GPU Tensor support
# Disable decoupled test because it uses GPU tensors
if [ "$TEST_JETSON" == "0" ]; then
    SUBTESTS="ensemble io bls decoupled variants python_based_backends"
    for TEST in ${SUBTESTS}; do
        # Run each subtest in a separate virtual environment to avoid conflicts
        # between dependencies.
        virtualenv --system-site-packages venv
        source venv/bin/activate

        (cd ${TEST} && bash -ex test.sh)
        if [ $? -ne 0 ]; then
        echo "Subtest ${TEST} FAILED"
        RET=1
        fi

        deactivate
        rm -fr venv
    done

    if [ ${PYTHON_ENV_VERSION} = "10" ]; then
        # In 'env' test we use miniconda for dependency management. No need to run
        # the test in a virtual environment.
        (cd env && bash -ex test.sh)
        if [ $? -ne 0 ]; then
            echo "Subtest env FAILED"
            RET=1
        fi
    fi
fi

SUBTESTS="lifecycle restart model_control examples argument_validation logging custom_metrics request_rescheduling"
for TEST in ${SUBTESTS}; do
    # Run each subtest in a separate virtual environment to avoid conflicts
    # between dependencies.
    virtualenv --system-site-packages venv
    source venv/bin/activate

    (cd ${TEST} && bash -ex test.sh)

    if [ $? -ne 0 ]; then
        echo "Subtest ${TEST} FAILED"
        RET=1
    fi

    deactivate
    rm -fr venv
done

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
