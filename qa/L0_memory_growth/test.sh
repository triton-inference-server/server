#!/bin/bash
# Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
if [ ! -z "$TEST_REPO_ARCH" ]; then
    REPO_VERSION=${REPO_VERSION}_${TEST_REPO_ARCH}
fi

# Single GPU
export CUDA_VISIBLE_DEVICES=0

# Clients
PERF_ANALYZER=../clients/perf_analyzer
IMAGE=../images/vulture.jpeg

# Models
CAFFE2PLAN=../common/caffe2plan
DATADIR=/data/inferenceserver/${REPO_VERSION}

# Server
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_TIMEOUT=1200

# Valgrind massif
LEAKCHECK=/usr/bin/valgrind
LEAKCHECK_ARGS_BASE="--tool=massif --time-unit=B"
MASSIF_TEST=../common/check_massif_log.py

source ../common/util.sh

# Function that checks the massif logs
function check_massif_log () {
    local massif_out=$1
}

rm -rf *.log models/ *.massif

# Test parameters
STATIC_BATCH=128
INSTANCE_CNT=2
CONCURRENCY=32
CLIENT_BS=8

# Set the number of repetitions in nightly and weekly tests
# Set the email subject for nightly and weekly tests
if [ "$TRITON_PERF_WEEKLY" == 1 ]; then
    if [ "$TRITON_PERF_LONG" == 1 ]; then
        # ~ 2.5 days for system under test
        REPETITION=1400
        EMAIL_SUBJECT="Weekly Long"
    else
        # Run the test for each model approximately 1.5 hours
        # All tests are run cumulatively for 7 hours
        REPETITION=200
        EMAIL_SUBJECT="Weekly"
    fi
else
    REPETITION=3
    EMAIL_SUBJECT="Nightly"
fi

# Threshold memory growth in MB
MAX_ALLOWED_ALLOC="150"
export MAX_ALLOWED_ALLOC

# Create local model repository
mkdir -p models/
cp -r $DATADIR/perf_model_store/resnet50* models/

# Copy and prepare trt model
cp -r $DATADIR/caffe_models/trt_model_store/resnet50_plan models/resnet50_fp16_plan
mkdir -p models/resnet50_fp16_plan/1
sed -i "s/^name:.*/name: \"resnet50_fp16_plan\"/" models/resnet50_fp16_plan/config.pbtxt

set +e

# Create the PLAN
$CAFFE2PLAN -h -b ${STATIC_BATCH} \
    -n prob -o models/resnet50_fp16_plan/1/model.plan \
    $DATADIR/caffe_models/resnet50.prototxt $DATADIR/caffe_models/resnet50.caffemodel
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Failed to generate resnet50 PLAN\n***"
    exit 1
fi

set -e

RET=0

for MODEL in $(ls models); do
    # Create temporary model repository and copy only the model being tested
    rm -rf test_repo && mkdir test_repo
    cp -r models/$MODEL test_repo/

    # Set server, client and valgrind arguments
    SERVER_ARGS="--model-repository=`pwd`/test_repo"
    LEAKCHECK_LOG="test_${MODEL}.valgrind.log"
    MASSIF_LOG="test_${MODEL}.massif"
    GRAPH_LOG="memory_growth_${MODEL}.log"
    LEAKCHECK_ARGS="$LEAKCHECK_ARGS_BASE --massif-out-file=$MASSIF_LOG --max-threads=3000 --log-file=$LEAKCHECK_LOG"
    SERVER_LOG="test_$MODEL.server.log"
    CLIENT_LOG="test_$MODEL.client.log"

    # Enable dynamic batching, set max batch size and instance count
    if [ "$MODEL" == "resnet50_fp32_libtorch" ]; then
        sed -i "s/^max_batch_size:.*/max_batch_size: 32/" test_repo/$MODEL/config.pbtxt
    else
        sed -i "s/^max_batch_size:.*/max_batch_size: ${STATIC_BATCH}/" test_repo/$MODEL/config.pbtxt
    fi
    echo "dynamic_batching {}" >> test_repo/$MODEL/config.pbtxt
    echo "instance_group [{ count: ${INSTANCE_CNT} }]" >> test_repo/$MODEL/config.pbtxt

    # Run the server
    run_server_leakcheck
    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    set +e

    TEMP_CLIENT_LOG=temp_client.log
    TEMP_RET=0

    SECONDS=0
    # Run the perf analyzer 'REPETITION' times
    for ((i=1; i<=$REPETITION; i++)); do
        $PERF_ANALYZER -v -m $MODEL -i grpc --concurrency-range $CONCURRENCY -b $CLIENT_BS > $TEMP_CLIENT_LOG 2>&1
        # Only record failure log for unexpected perf analyzer error
        # [TMA-625] Currently check failure log as WAR, should check for specific
        # code once perf analyzer returns different code for different error
        if [ $? -ne 0 ] && [ `grep -c "^No valid requests recorded" $TEMP_CLIENT_LOG` == "0" ]; then
            cat $TEMP_CLIENT_LOG >> $CLIENT_LOG
            echo -e "\n***\n*** perf_analyzer for $MODEL failed on iteration $i\n***" >> $CLIENT_LOG
            TEMP_RET=1
        fi
    done
    TEST_DURATION=$SECONDS

    if [ $TEMP_RET -ne 0 ]; then
        cat $CLIENT_LOG
        RET=1
    fi

    set -e

    # Stop Server
    kill $SERVER_PID
    wait $SERVER_PID

    set +e

    # Log test duration and the graph for memory growth
    hrs=$(printf "%02d" $((TEST_DURATION / 3600)))
    mins=$(printf "%02d" $(((TEST_DURATION / 60) % 60)))
    secs=$(printf "%02d" $((TEST_DURATION % 60)))
    echo -e "Test Duration: $hrs:$mins:$secs (HH:MM:SS)" >> ${GRAPH_LOG}
    ms_print ${MASSIF_LOG} | head -n35 >> ${GRAPH_LOG}
    cat ${GRAPH_LOG}
    # Check the massif output
    python $MASSIF_TEST $MASSIF_LOG $MAX_ALLOWED_ALLOC --start-from-middle >> $CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test for $MODEL Failed\n***"
        RET=1
    fi
    set -e
done

# Next perform a test that has unbound memory growth. Use the busy op model
# with a high delay in order to force requests to sit in the queue, and result
# in memory growth.
BUSY_OP_TEST=busy_op_test.py
DELAY_CYCLES=2100000000
NUM_REQUESTS=100

rm -rf test_repo && mkdir test_repo
cp -r ${DATADIR}/qa_custom_ops/tf_custom_ops/graphdef_busyop test_repo/

# Explicitly set library path so custom ops can find TF
LD_LIBRARY_PATH=/opt/tritonserver/backends/tensorflow2
SERVER_ARGS="--model-repository=`pwd`/test_repo"
SERVER_LD_PRELOAD="${DATADIR}/qa_custom_ops/tf_custom_ops/libbusyop.so"

LEAKCHECK_LOG="test_busyop.valgrind.log"
MASSIF_LOG="test_busyop.massif"
GRAPH_LOG="memory_growth_busyop.log"
LEAKCHECK_ARGS="$LEAKCHECK_ARGS_BASE --massif-out-file=$MASSIF_LOG --max-threads=3000 --log-file=$LEAKCHECK_LOG"
SERVER_LOG="test_busyop.server.log"
CLIENT_LOG="test_busyop.client.log"
SKIP_BUSYOP=0

# Run server
run_server_leakcheck
if [ "$SERVER_PID" == "0" ]; then
    cat $SERVER_LOG
    if [ `grep -c "provided PTX was compiled" $SERVER_LOG` != "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER due to PTX issue\n***"
        SKIP_BUSYOP=1
    else
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        exit 1
    fi
fi

set +e

# Run the busy_op test if no PTX issue was observed when launching server
if [ $SKIP_BUSYOP -ne 1 ]; then
    SECONDS=0
    python $BUSY_OP_TEST -v -m graphdef_busyop -d $DELAY_CYCLES -n $NUM_REQUESTS > $CLIENT_LOG 2>&1
    TEST_DURATION=$SECONDS
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test graphdef_busyop Failed\n***"
        RET=1
    fi
    set -e

    # Stop Server
    kill $SERVER_PID
    wait $SERVER_PID

    set +e

    # Log test duration and the graph for memory growth
    hrs=$(printf "%02d" $((TEST_DURATION / 3600)))
    mins=$(printf "%02d" $(((TEST_DURATION / 60) % 60)))
    secs=$(printf "%02d" $((TEST_DURATION % 60)))
    echo -e "Test Duration: $hrs:$mins:$secs (HH:MM:SS)" >> ${GRAPH_LOG}
    ms_print ${MASSIF_LOG} | head -n35 >> ${GRAPH_LOG}
    cat ${GRAPH_LOG}
    # Check the massif output
    python $MASSIF_TEST $MASSIF_LOG $MAX_ALLOWED_ALLOC --start-from-middle >> $CLIENT_LOG 2>&1
    if [ $? -ne 1 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test for graphdef_busyop Failed\n***"
        RET=1
    fi
fi
set -e

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

# Run only if both TRITON_FROM and TRITON_TO_DL are set
if [[ ! -z "$TRITON_FROM" ]] && [[ ! -z "$TRITON_TO_DL" ]]; then
    python server_memory_mail.py "$EMAIL_SUBJECT"
fi

exit $RET
