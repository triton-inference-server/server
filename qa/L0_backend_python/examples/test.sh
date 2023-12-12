#!/bin/bash
# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

source ../common.sh
source ../../common/util.sh

TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
SERVER=${TRITON_DIR}/bin/tritonserver
BACKEND_DIR=${TRITON_DIR}/backends
SERVER_ARGS="--model-repository=`pwd`/python_backend/models --backend-directory=${BACKEND_DIR} --log-verbose=1"
SERVER_LOG="./examples_server.log"

RET=0
rm -fr *.log python_backend/

# Install torch
pip3 uninstall -y torch
if [ "$TEST_JETSON" == "0" ]; then
    pip3 install torch==2.0.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html torchvision==0.15.0+cu117
else
    pip3 install torch==2.0.0 -f https://download.pytorch.org/whl/torch_stable.html torchvision==0.15.0
fi

# Install `validators` for Model Instance Kind example
pip3 install validators

# Install JAX
if [ "$TEST_JETSON" == "0" ]; then
    pip3 install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

git clone https://github.com/triton-inference-server/python_backend -b $PYTHON_BACKEND_REPO_TAG
cd python_backend

# Example 1
CLIENT_LOG="./examples_add_sub_client.log"
mkdir -p models/add_sub/1/
cp examples/add_sub/model.py models/add_sub/1/model.py
cp examples/add_sub/config.pbtxt models/add_sub/config.pbtxt
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

set +e
python3 examples/add_sub/client.py > $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify add_sub example. \n***"
    RET=1
fi

grep "PASS" $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify add_sub example. \n***"
    cat $CLIENT_LOG
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# Example 2
CLIENT_LOG="./examples_pytorch_client.log"
mkdir -p models/pytorch/1/
cp examples/pytorch/model.py models/pytorch/1/model.py
cp examples/pytorch/config.pbtxt models/pytorch/config.pbtxt
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

set +e
python3 examples/pytorch/client.py > $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify pytorch example. \n***"
    RET=1
fi

grep "PASS" $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify pytorch example. \n***"
    cat $CLIENT_LOG
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# Example 3

# JAX AddSub
# JAX is not supported on Jetson
if [ "$TEST_JETSON" == "0" ]; then
    CLIENT_LOG="./examples_jax_client.log"
    mkdir -p models/jax/1/
    cp examples/jax/model.py models/jax/1/model.py
    cp examples/jax/config.pbtxt models/jax/config.pbtxt
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        RET=1
    fi

    set +e
    python3 examples/jax/client.py > $CLIENT_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed to verify jax example. \n***"
        RET=1
    fi

    grep "PASS" $CLIENT_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed to verify jax example. \n***"
        cat $CLIENT_LOG
        RET=1
    fi
    set -e

    kill $SERVER_PID
    wait $SERVER_PID
fi

# Example 4

# BLS Sync
CLIENT_LOG="./examples_sync_client.log"
mkdir -p models/bls_sync/1
cp examples/bls/sync_model.py models/bls_sync/1/model.py
cp examples/bls/sync_config.pbtxt models/bls_sync/config.pbtxt
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

set +e
python3 examples/bls/sync_client.py > $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify BLS sync example. \n***"
    RET=1
fi

grep "PASS" $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify BLS sync example. \n***"
    cat $CLIENT_LOG
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# Example 5

# Decoupled Repeat
CLIENT_LOG="./examples_repeat_client.log"
mkdir -p models/repeat_int32/1/
cp examples/decoupled/repeat_model.py models/repeat_int32/1/model.py
cp examples/decoupled/repeat_config.pbtxt models/repeat_int32/config.pbtxt
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

set +e
python3 examples/decoupled/repeat_client.py > $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify repeat_int32 example. \n***"
    RET=1
fi

grep "PASS" $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify repeat_int32 example. \n***"
    cat $CLIENT_LOG
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# Example 6

# Decoupled Square
CLIENT_LOG="./examples_square_client.log"
mkdir -p models/square_int32/1/
cp examples/decoupled/square_model.py models/square_int32/1/model.py
cp examples/decoupled/square_config.pbtxt models/square_int32/config.pbtxt
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

set +e
python3 examples/decoupled/square_client.py > $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify square_int32 example. \n***"
    RET=1
fi

grep "PASS" $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify square_int32 example. \n***"
    cat $CLIENT_LOG
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

#
# BLS Async
#
# Skip async BLS on Jetson since it is not supported with python3.6
# Having multiple python versions lead to build issues.
# Anaconda is not officially supported on Jetson.
if [ "$TEST_JETSON" == "0" ]; then
    CLIENT_LOG="./examples_async_client.log"
    mkdir -p models/bls_async/1
    cp examples/bls/async_model.py models/bls_async/1/model.py
    cp examples/bls/async_config.pbtxt models/bls_async/config.pbtxt
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        RET=1
    fi

    set +e
    python3 examples/bls/async_client.py > $CLIENT_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed to verify BLS async example. \n***"
        RET=1
    fi

    grep "PASS" $CLIENT_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed to verify BLS async example. \n***"
        cat $CLIENT_LOG
        RET=1
    fi

    set -e

    kill $SERVER_PID
    wait $SERVER_PID
fi

# Auto Complete Model Configuration Example
CLIENT_LOG="./examples_auto_complete_client.log"
mkdir -p models/nobatch_auto_complete/1/
mkdir -p models/batch_auto_complete/1/
cp examples/auto_complete/nobatch_model.py models/nobatch_auto_complete/1/model.py
cp examples/auto_complete/batch_model.py models/batch_auto_complete/1/model.py

SERVER_ARGS="$SERVER_ARGS --strict-model-config=false"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

set +e
python3 examples/auto_complete/client.py > $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify auto_complete example. \n***"
    RET=1
fi

grep "PASS" $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify auto_complete example. \n***"
    cat $CLIENT_LOG
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# BLS Decoupled Sync
CLIENT_LOG="./examples_bls_decoupled_sync_client.log"
mkdir -p models/bls_decoupled_sync/1
cp examples/bls_decoupled/sync_model.py models/bls_decoupled_sync/1/model.py
cp examples/bls_decoupled/sync_config.pbtxt models/bls_decoupled_sync/config.pbtxt
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

set +e
python3 examples/bls_decoupled/sync_client.py > $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify BLS Decoupled Sync example. \n***"
    RET=1
fi

grep "PASS" $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify BLS Decoupled Sync example. \n***"
    cat $CLIENT_LOG
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# BLS Decoupled Async
if [ "$TEST_JETSON" == "0" ]; then
    CLIENT_LOG="./examples_bls_decoupled_async_client.log"
    mkdir -p models/bls_decoupled_async/1
    cp examples/bls_decoupled/async_model.py models/bls_decoupled_async/1/model.py
    cp examples/bls_decoupled/async_config.pbtxt models/bls_decoupled_async/config.pbtxt
    run_server
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        RET=1
    fi

    set +e
    python3 examples/bls_decoupled/async_client.py > $CLIENT_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed to verify BLS Decoupled Async example. \n***"
        RET=1
    fi

    grep "PASS" $CLIENT_LOG
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Failed to verify BLS Decoupled Async example. \n***"
        cat $CLIENT_LOG
        RET=1
    fi

    set -e

    kill $SERVER_PID
    wait $SERVER_PID
fi

# Example 7

# Model Instance Kind
CLIENT_LOG="./examples_model_instance_kind.log"
mkdir -p models/resnet50/1
cp examples/instance_kind/model.py models/resnet50/1/
cp examples/instance_kind/config.pbtxt models/resnet50/
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

set +e
python3 examples/instance_kind/client.py --label_file examples/instance_kind/resnet50_labels.txt > $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify Model instance Kind example. \n***"
    RET=1
fi

grep "PASS" $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify Model Instance Kind example. Example failed to pass. \n***"
    cat $CLIENT_LOG
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# Custom Metrics
CLIENT_LOG="./examples_custom_metrics_client.log"
mkdir -p models/custom_metrics/1
cp examples/custom_metrics/model.py models/custom_metrics/1/model.py
cp examples/custom_metrics/config.pbtxt models/custom_metrics/config.pbtxt
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    RET=1
fi

set +e
python3 examples/custom_metrics/client.py > $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify Custom Metrics example. \n***"
    RET=1
fi

grep "PASS" $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to verify Custom Metrics example. \n***"
    cat $CLIENT_LOG
    RET=1
fi
set -e

kill $SERVER_PID
wait $SERVER_PID


if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Example verification test PASSED.\n***"
else
    echo -e "\n***\n*** Example verification test FAILED.\n***"
fi

exit $RET
