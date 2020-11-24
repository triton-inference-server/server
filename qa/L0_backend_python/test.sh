#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

CLIENT_PY=./python_test.py
CLIENT_LOG="./client.log"
EXPECTED_NUM_TESTS="6"

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=`pwd`/models --log-verbose=1"
SERVER_LOG="./inference_server.log"
source ../common/util.sh

rm -fr *.log ./models

cp /opt/tritonserver/backends/python/triton_python_backend_utils.py .

mkdir -p models/identity_fp32/1/
cp ../python_models/identity_fp32/model.py ./models/identity_fp32/1/model.py
cp ../python_models/identity_fp32/config.pbtxt ./models/identity_fp32/config.pbtxt

cp -r ./models/identity_fp32 ./models/identity_uint8
(cd models/identity_uint8 && \
          sed -i "s/^name:.*/name: \"identity_uint8\"/" config.pbtxt && \
          sed -i "s/TYPE_FP32/TYPE_UINT8/g" config.pbtxt && \
          sed -i "s/^max_batch_size:.*/max_batch_size: 8/" config.pbtxt && \
          echo "dynamic_batching { preferred_batch_size: [8], max_queue_delay_microseconds: 12000000 }" >> config.pbtxt)

cp -r ./models/identity_fp32 ./models/identity_uint32
(cd models/identity_uint32 && \
          sed -i "s/^name:.*/name: \"identity_uint32\"/" config.pbtxt && \
          sed -i "s/TYPE_FP32/TYPE_UINT32/g" config.pbtxt)

mkdir -p models/wrong_model/1/
cp ../python_models/wrong_model/model.py ./models/wrong_model/1/
cp ../python_models/wrong_model/config.pbtxt ./models/wrong_model/
(cd models/wrong_model && \
          sed -i "s/^name:.*/name: \"wrong_model\"/" config.pbtxt && \
          sed -i "s/TYPE_FP32/TYPE_UINT32/g" config.pbtxt)

mkdir -p models/pytorch_fp32_fp32/1/
cp -r ../python_models/pytorch_fp32_fp32/model.py ./models/pytorch_fp32_fp32/1/
cp ../python_models/pytorch_fp32_fp32/config.pbtxt ./models/pytorch_fp32_fp32/
(cd models/pytorch_fp32_fp32 && \
          sed -i "s/^name:.*/name: \"pytorch_fp32_fp32\"/" config.pbtxt)

mkdir -p models/execute_error/1/
cp ../python_models/execute_error/model.py ./models/execute_error/1/
cp ../python_models/execute_error/config.pbtxt ./models/execute_error/

mkdir -p models/init_args/1/
cp ../python_models/init_args/model.py ./models/init_args/1/
cp ../python_models/init_args/config.pbtxt ./models/init_args/

# Ensemble Model
mkdir -p models/ensemble/1/
cp ../python_models/ensemble/config.pbtxt ./models/ensemble

mkdir -p models/add_sub_1/1/
cp ../python_models/add_sub/config.pbtxt ./models/add_sub_1
(cd models/add_sub_1 && \
          sed -i "s/^name:.*/name: \"add_sub_1\"/" config.pbtxt)
cp ../python_models/add_sub/model.py ./models/add_sub_1/1/

mkdir -p models/add_sub_2/1/
cp ../python_models/add_sub/config.pbtxt ./models/add_sub_2/
(cd models/add_sub_2 && \
          sed -i "s/^name:.*/name: \"add_sub_2\"/" config.pbtxt)
cp ../python_models/add_sub/model.py ./models/add_sub_2/1/

pip3 install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

RET=0

set +e
python3 $CLIENT_PY >>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    RET=1
else
    check_test_results $CLIENT_LOG $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# Triton non-graceful exit

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

sleep 5

triton_procs=`pgrep --parent $SERVER_PID`

set +e
# Trigger non-graceful termination of Triton
kill -9 $SERVER_PID

# Wait 10 seconds so that Python gRPC server can detect non-graceful exit
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

# These models have errors in the initialization and finalization
# steps and we want to ensure that correct error is being returned

rm -rf models/
mkdir -p models/init_error/1/
cp ../python_models/init_error/model.py ./models/init_error/1/
cp ../python_models/init_error/config.pbtxt ./models/init_error/

set +e
run_server_nowait
wait $SERVER_PID

grep "name 'lorem_ipsum' is not defined" $SERVER_LOG

if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** init_error model test failed \n***"
    RET=1
fi
set -e

rm -rf models/
mkdir -p models/fini_error/1/
cp ../python_models/fini_error/model.py ./models/fini_error/1/
cp ../python_models/fini_error/config.pbtxt ./models/fini_error/

set +e
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

kill $SERVER_PID
wait $SERVER_PID

grep "name 'undefined_variable' is not defined" $SERVER_LOG

if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** fini_error model test failed \n***"
    RET=1
fi
set -e

# Test KIND_GPU
rm -rf models/
mkdir -p models/add_sub_gpu/1/
cp ../python_models/add_sub/model.py ./models/add_sub_gpu/1/
cp ../python_models/add_sub_gpu/config.pbtxt ./models/add_sub_gpu/

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** KIND_GPU model test failed \n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

# Test Multi file models
rm -rf models/
mkdir -p models/multi_file/1/
cp ../python_models/multi_file/*.py ./models/multi_file/1/
cp ../python_models/identity_fp32/config.pbtxt ./models/multi_file/
(cd models/multi_file && \
          sed -i "s/^name:.*/name: \"multi_file\"/" config.pbtxt)

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** multi-file model test failed \n***"
    RET=1
fi

kill $SERVER_PID
wait $SERVER_PID

# Test environment variable propagation
rm -rf models/
mkdir -p models/model_env/1/
cp ../python_models/model_env/model.py ./models/model_env/1/
cp ../python_models/model_env/config.pbtxt ./models/model_env/

export MY_ENV="MY_ENV"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    echo -e "\n***\n*** Environment variable test failed \n***"
    cat $SERVER_LOG
    exit 1
fi

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET

