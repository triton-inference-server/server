#!/bin/bash
# Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

SIMPLE_HTTP_CLIENT=../clients/simple_http_infer_client
SIMPLE_GRPC_CLIENT=../clients/simple_grpc_infer_client
TRACE_SUMMARY=../common/trace_summary.py

REPO_VERSION=${NVIDIA_TRITON_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi
if [ ! -z "$TEST_REPO_ARCH" ]; then
    REPO_VERSION=${REPO_VERSION}_${TEST_REPO_ARCH}
fi

export CUDA_VISIBLE_DEVICES=0

DATADIR=/data/inferenceserver/${REPO_VERSION}/qa_model_repository
ENSEMBLEDIR=$DATADIR/../qa_ensemble_model_repository/qa_model_repository/
MODELBASE=onnx_int32_int32_int32

MODELSDIR=`pwd`/trace_models

SERVER=/opt/tritonserver/bin/tritonserver
source ../common/util.sh

rm -f *.log
rm -fr $MODELSDIR && mkdir -p $MODELSDIR

# set up simple model using MODELBASE, this test needs gradually update as
# backends are ported to use backend API as backend API not yet support tracing.
rm -fr $MODELSDIR && mkdir -p $MODELSDIR && \
    cp -r $DATADIR/$MODELBASE $MODELSDIR/simple && \
    rm -r $MODELSDIR/simple/2 && rm -r $MODELSDIR/simple/3 && \
    (cd $MODELSDIR/simple && \
            sed -i "s/^name:.*/name: \"simple\"/" config.pbtxt)

RET=0

# trace-level=OFF make sure no tracing
SERVER_ARGS="--trace-file=trace_off.log --trace-level=OFF --trace-rate=1 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_off.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

for p in {1..10}; do
    $SIMPLE_HTTP_CLIENT >> client_off.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_GRPC_CLIENT >> client_off.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

set -e

kill $SERVER_PID
wait $SERVER_PID

set +e

if [ -f ./trace_off.log ]; then
    echo -e "\n***\n*** Test Failed, unexpected generation of trace_off.log\n***"
    RET=1
fi

set -e

# trace-rate == 1, trace-level=MIN make sure every request is traced
SERVER_ARGS="--trace-file=trace_min.log --trace-level=MIN --trace-rate=1 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_min.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

for p in {1..10}; do
    $SIMPLE_HTTP_CLIENT >> client_min.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_GRPC_CLIENT >> client_min.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

set -e

kill $SERVER_PID
wait $SERVER_PID

set +e

$TRACE_SUMMARY -t trace_min.log > summary_min.log

if [ `grep -c "COMPUTE_INPUT_END" summary_min.log` != "20" ]; then
    cat summary_min.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

if [ `grep -c ^simple summary_min.log` != "20" ]; then
    cat summary_min.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

# trace-rate == 9, trace-level=MAX
SERVER_ARGS="--http-thread-count=1 --trace-file=trace_max.log \
             --trace-level=MAX --trace-rate=9 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_max.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

for p in {1..10}; do
    $SIMPLE_HTTP_CLIENT >> client_max.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_GRPC_CLIENT >> client_max.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

set -e

kill $SERVER_PID
wait $SERVER_PID

set +e

$TRACE_SUMMARY -t trace_max.log > summary_max.log

if [ `grep -c "COMPUTE_INPUT_END" summary_max.log` != "2" ]; then
    cat summary_max.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

if [ `grep -c ^simple summary_max.log` != "2" ]; then
    cat summary_max.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

# trace-rate == 1, trace-level=TIMESTAMPS make sure every request is traced
SERVER_ARGS="--trace-file=trace_1.log --trace-level=TIMESTAMPS --trace-rate=1 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_1.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

for p in {1..10}; do
    $SIMPLE_HTTP_CLIENT >> client_1.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_GRPC_CLIENT >> client_1.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

set -e

kill $SERVER_PID
wait $SERVER_PID

set +e

$TRACE_SUMMARY -t trace_1.log > summary_1.log

if [ `grep -c "COMPUTE_INPUT_END" summary_1.log` != "20" ]; then
    cat summary_1.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

if [ `grep -c ^simple summary_1.log` != "20" ]; then
    cat summary_1.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

# trace-rate == 6, trace-level=TIMESTAMPS
SERVER_ARGS="--http-thread-count=1 --trace-file=trace_6.log \
             --trace-level=TIMESTAMPS --trace-rate=6 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_6.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

for p in {1..10}; do
    $SIMPLE_HTTP_CLIENT >> client_6.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_GRPC_CLIENT >> client_6.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

set -e

kill $SERVER_PID
wait $SERVER_PID

set +e

$TRACE_SUMMARY -t trace_6.log > summary_6.log

if [ `grep -c "COMPUTE_INPUT_END" summary_6.log` != "3" ]; then
    cat summary_6.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

if [ `grep -c ^simple summary_6.log` != "3" ]; then
    cat summary_6.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

# trace-rate == 6, trace-level=TIMESTAMPS, trace-log-frequency == 2
SERVER_ARGS="--http-thread-count=1 --trace-file=trace_frequency.log \
             --trace-level=TIMESTAMPS --trace-rate=6 \
             --trace-log-frequency=2 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_frequency.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

for p in {1..10}; do
    $SIMPLE_HTTP_CLIENT >> client_frequency.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_GRPC_CLIENT >> client_frequency.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

set -e

kill $SERVER_PID
wait $SERVER_PID

set +e

# Two trace files
$TRACE_SUMMARY -t trace_frequency.log.0 > summary_frequency.log.0
if [ `grep -c "COMPUTE_INPUT_END" summary_frequency.log.0` != "2" ]; then
    cat summary_frequency.log.0
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

if [ `grep -c ^simple summary_frequency.log.0` != "2" ]; then
    cat summary_frequency.log.0
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

$TRACE_SUMMARY -t trace_frequency.log.1 > summary_frequency.log.1
if [ `grep -c "COMPUTE_INPUT_END" summary_frequency.log.1` != "1" ]; then
    cat summary_frequency.log.1
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

if [ `grep -c ^simple summary_frequency.log.1` != "1" ]; then
    cat summary_frequency.log.1
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

# trace-rate == 9, trace-level=TIMESTAMPS
SERVER_ARGS="--http-thread-count=1 --trace-file=trace_9.log \
             --trace-level=TIMESTAMPS --trace-rate=9 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_9.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

for p in {1..10}; do
    $SIMPLE_HTTP_CLIENT >> client_9.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi

    $SIMPLE_GRPC_CLIENT >> client_9.log 2>&1
    if [ $? -ne 0 ]; then
        RET=1
    fi
done

set -e

kill $SERVER_PID
wait $SERVER_PID

set +e

$TRACE_SUMMARY -t trace_9.log > summary_9.log

if [ `grep -c "COMPUTE_INPUT_END" summary_9.log` != "2" ]; then
    cat summary_9.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

if [ `grep -c ^simple summary_9.log` != "2" ]; then
    cat summary_9.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

# Demonstrate trace for ensemble
# set up "addsub" nested ensemble
rm -fr $MODELSDIR && mkdir -p $MODELSDIR && \
    cp -r $DATADIR/$MODELBASE $MODELSDIR/$MODELBASE && \
    rm -r $MODELSDIR/$MODELBASE/2 && rm -r $MODELSDIR/$MODELBASE/3

# nested ensemble
mkdir -p $MODELSDIR/fan_$MODELBASE/1 && \
    cp $ENSEMBLEDIR/fan_$MODELBASE/config.pbtxt $MODELSDIR/fan_$MODELBASE/. && \
        (cd $MODELSDIR/fan_$MODELBASE && \
                sed -i "s/label_filename:.*//" config.pbtxt)

mkdir -p $MODELSDIR/simple/1 && \
    cp $ENSEMBLEDIR/fan_$MODELBASE/config.pbtxt $MODELSDIR/simple/. && \
        (cd $MODELSDIR/simple && \
                sed -i "s/^name:.*/name: \"simple\"/" config.pbtxt && \
                sed -i "s/$MODELBASE/fan_$MODELBASE/" config.pbtxt && \
                sed -i "s/label_filename:.*//" config.pbtxt)

cp -r $ENSEMBLEDIR/nop_TYPE_INT32_-1 $MODELSDIR/. && \
    mkdir -p $MODELSDIR/nop_TYPE_INT32_-1/1

# trace-rate == 1, trace-level=TIMESTAMPS
SERVER_ARGS="--http-thread-count=1 --trace-file=trace_ensemble.log \
             --trace-level=TIMESTAMPS --trace-rate=1 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_ensemble.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

$SIMPLE_HTTP_CLIENT >> client_ensemble.log 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

set +e

$TRACE_SUMMARY -t trace_ensemble.log > summary_ensemble.log

# Check if the traces are captured with proper hierarchy
if [ `grep -c "COMPUTE_INPUT_END" summary_ensemble.log` != "7" ]; then
    echo -e "Ensemble trace log expects 7 compute"
    RET=1
fi

for trace_str in \
        "{\"id\":1,\"model_name\":\"simple\",\"model_version\":1}" \
        "{\"id\":2,\"model_name\":\"nop_TYPE_INT32_-1\",\"model_version\":1,\"parent_id\":1}" \
        "{\"id\":3,\"model_name\":\"fan_${MODELBASE}\",\"model_version\":1,\"parent_id\":1}" \
        "{\"id\":4,\"model_name\":\"nop_TYPE_INT32_-1\",\"model_version\":1,\"parent_id\":3}" \
        "{\"id\":5,\"model_name\":\"${MODELBASE}\",\"model_version\":1,\"parent_id\":3}" \
        "{\"id\":6,\"model_name\":\"nop_TYPE_INT32_-1\",\"model_version\":1,\"parent_id\":3}" \
        "{\"id\":7,\"model_name\":\"nop_TYPE_INT32_-1\",\"model_version\":1,\"parent_id\":3}" \
        "{\"id\":8,\"model_name\":\"nop_TYPE_INT32_-1\",\"model_version\":1,\"parent_id\":1}" \
        "{\"id\":9,\"model_name\":\"nop_TYPE_INT32_-1\",\"model_version\":1,\"parent_id\":1}" ; do
    if [ `grep -c ${trace_str} trace_ensemble.log` != "1" ]; then
        echo -e "Ensemble trace log expects trace: ${trace_str}"
        RET=1
    fi
done

if [ `grep -c ^simple summary_ensemble.log` != "1" ]; then
    cat summary_ensemble.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e


if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

# trace-rate == 1, trace-level=TIMESTAMPS, trace-level=TENSORS
SERVER_ARGS="--http-thread-count=1 --trace-file=trace_ensemble_tensor.log \
             --trace-level=TIMESTAMPS --trace-level=TENSORS --trace-rate=1 --model-repository=$MODELSDIR"
SERVER_LOG="./inference_server_ensemble_tensor.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e

$SIMPLE_HTTP_CLIENT >> client_ensemble_tensor.log 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

set +e

$TRACE_SUMMARY -t trace_ensemble_tensor.log > summary_ensemble_tensor.log

# Check if the traces are captured with proper hierarchy
if [ `grep -c "COMPUTE_INPUT_END" summary_ensemble_tensor.log` != "7" ]; then
    echo -e "Ensemble trace tensors log expects 7 compute"
    RET=1
fi
for trace_str in \
        "{\"id\":1,\"model_name\":\"simple\",\"model_version\":1}" \
        "{\"id\":2,\"model_name\":\"nop_TYPE_INT32_-1\",\"model_version\":1,\"parent_id\":1}" \
        "{\"id\":3,\"model_name\":\"fan_${MODELBASE}\",\"model_version\":1,\"parent_id\":1}" \
        "{\"id\":4,\"model_name\":\"nop_TYPE_INT32_-1\",\"model_version\":1,\"parent_id\":3}" \
        "{\"id\":5,\"model_name\":\"${MODELBASE}\",\"model_version\":1,\"parent_id\":3}" \
        "{\"id\":6,\"model_name\":\"nop_TYPE_INT32_-1\",\"model_version\":1,\"parent_id\":3}" \
        "{\"id\":7,\"model_name\":\"nop_TYPE_INT32_-1\",\"model_version\":1,\"parent_id\":3}" \
        "{\"id\":8,\"model_name\":\"nop_TYPE_INT32_-1\",\"model_version\":1,\"parent_id\":1}" \
        "{\"id\":9,\"model_name\":\"nop_TYPE_INT32_-1\",\"model_version\":1,\"parent_id\":1}" ; do
    if [ `grep -c ${trace_str} trace_ensemble_tensor.log` != "1" ]; then
        echo -e "Ensemble trace tensors log expects trace: ${trace_str}"
        RET=1
    fi
done

if [ `grep -c ^simple summary_ensemble_tensor.log` != "1" ]; then
    cat summary_ensemble_tensor.log
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

if [ `grep -o TENSOR_QUEUE_INPUT trace_ensemble_tensor.log | wc -l` != "18" ]; then
    echo -e "Ensemble trace tensors log expects 18 TENSOR_QUEUE_INPUTs"
    RET=1
fi

if [ `grep -o TENSOR_BACKEND_OUTPUT trace_ensemble_tensor.log | wc -l` != "14" ]; then
    echo -e "Ensemble trace tensors log expects 14 TENSOR_BACKEND_OUTPUTs"
    RET=1
fi

for trace_str in \
        "{\"id\":1,\"activity\":\"TENSOR_QUEUE_INPUT\",\"tensor\":{\"name\":\"INPUT0\",\"data\":\"0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15\",\"shape\":\"1,16\",\"dtype\":\"INT32\"}}" \
        "{\"id\":1,\"activity\":\"TENSOR_QUEUE_INPUT\",\"tensor\":{\"name\":\"INPUT1\",\"data\":\"1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\",\"shape\":\"1,16\",\"dtype\":\"INT32\"}}" \
        "{\"id\":1,\"activity\":\"TENSOR_BACKEND_OUTPUT\",\"tensor\":{\"name\":\"OUTPUT0\",\"data\":\"1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16\",\"shape\":\"1,16\",\"dtype\":\"INT32\"}}" \
        "{\"id\":1,\"activity\":\"TENSOR_BACKEND_OUTPUT\",\"tensor\":{\"name\":\"OUTPUT1\",\"data\":\"-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14\",\"shape\":\"1,16\",\"dtype\":\"INT32\"}}" ; do
    if [ `grep -c ${trace_str} trace_ensemble_tensor.log` != "1" ]; then
        echo -e "Ensemble trace tensors log expects trace: ${trace_str}"
        RET=1
    fi
done

set -e


if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi


exit $RET
