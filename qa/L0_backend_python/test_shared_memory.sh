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

prev_num_pages=`get_shm_pages`
run_server
if [ "$SERVER_PID" == "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    exit 1
fi

set +e
python3 -m pytest --junitxml=L0_backend_python.report.xml $CLIENT_PY >> $CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    RET=1
fi
set -e

kill_server

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

readarray -t triton_procs < <(pgrep --parent ${SERVER_PID})

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
# Disable env test for Jetson & Windows since GPU Tensors are not supported
if [ "$TEST_JETSON" == "0" ] && [[ ${TEST_WINDOWS} == 0 ]]; then
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

  kill_server

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

kill_server

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
if [[ ${TEST_WINDOWS} == 1 ]]; then
    # This will run in WSL, but Triton will run in windows, so environment
    # variables meant for loaded models must be exported using WSLENV.
    # The /w flag indicates the value should only be included when invoking
    # Win32 from WSL.
    export WSLENV=MY_ENV/w
fi

prev_num_pages=`get_shm_pages`
run_server
if [ "$SERVER_PID" == "0" ]; then
    cat $SERVER_LOG
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    echo -e "\n***\n*** Environment variable test failed \n***"
    exit 1
fi

kill_server

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

kill_server

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
    kill_server
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
