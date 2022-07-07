#!/bin/bash
# Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Must run on a single device or else the TRITONSERVER_DELAY_SCHEDULER
# can fail when the requests are distributed to multiple devices.
export CUDA_VISIBLE_DEVICES=0

LEAKCHECK=/usr/bin/valgrind
LEAKCHECK_ARGS_BASE="--max-threads=3000 --tool=massif --time-unit=B"
SERVER_TIMEOUT=3600
rm -f *.log *.massif

MEMORY_GROWTH_TEST_CPP=../clients/memory_leak_test
MEMORY_GROWTH_TEST_PY=../clients/memory_growth_test.py
MASSIF_TEST=../common/check_massif_log.py

DATADIR=`pwd`/models
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=$DATADIR"
source ../common/util.sh

# Set the number of repetitions in nightly and weekly tests
# Set the email subject for nightly and weekly tests
if [ "$TRITON_PERF_WEEKLY" == 1 ]; then
    if [ "$TRITON_PERF_LONG" == 1 ]; then
        # ~ 12 hours
        # GRPC cycles are reduced as there is high fluctuation in time spent
        REPETITION_HTTP_CPP=2220000
        REPETITION_HTTP_PY=3600000
        REPETITION_GRPC_CPP=8000000
        REPETITION_GRPC_PY=1500000
        EMAIL_SUBJECT="Weekly Long"
    else
        # Run the test for each case approximately 1.5 hours
        # All tests are run cumulatively for 7 hours
        REPETITION_HTTP_CPP=1300000
        REPETITION_HTTP_PY=2100000
        REPETITION_GRPC_CPP=6600000
        REPETITION_GRPC_PY=1000000
        EMAIL_SUBJECT="Weekly"
    fi
else
    REPETITION_CPP=100000
    REPETITION_PY=10000
    EMAIL_SUBJECT="Nightly"
fi

mkdir -p $DATADIR/custom_identity_int32/1

RET=0

# Run test for both HTTP and GRPC, not re-using client object.
for PROTOCOL in http grpc; do
    for LANG in c++ python; do
        LEAKCHECK_LOG="./valgrind.${PROTOCOL}.${LANG}.log"
        CLIENT_LOG="./client.${PROTOCOL}.${LANG}.log"
        GRAPH_LOG="./client_memory_growth.${PROTOCOL}.${LANG}.log"
        MASSIF_LOG="./${PROTOCOL}.${LANG}.massif"
        LEAKCHECK_ARGS="$LEAKCHECK_ARGS_BASE --log-file=$LEAKCHECK_LOG --massif-out-file=$MASSIF_LOG"

        if [ "$TRITON_PERF_WEEKLY" == 1 ]; then
            if [ $PROTOCOL ==  http ]; then
                REPETITION_CPP=$REPETITION_HTTP_CPP
                REPETITION_PY=$REPETITION_HTTP_PY
            else
                REPETITION_CPP=$REPETITION_GRPC_CPP
                REPETITION_PY=$REPETITION_GRPC_PY
            fi
        fi

        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        # MAX_ALLOWED_ALLOC is the threshold memory growth in MB
        if [ "$LANG" == "c++" ]; then
            MEMORY_GROWTH_TEST=$MEMORY_GROWTH_TEST_CPP
            MAX_ALLOWED_ALLOC="10"
            EXTRA_ARGS="-r ${REPETITION_CPP} -i ${PROTOCOL}"
        else
            MEMORY_GROWTH_TEST="python $MEMORY_GROWTH_TEST_PY"
            MAX_ALLOWED_ALLOC="1"
            EXTRA_ARGS="-r ${REPETITION_PY} -i ${PROTOCOL}"
        fi

        set +e
        SECONDS=0
        $LEAKCHECK $LEAKCHECK_ARGS $MEMORY_GROWTH_TEST $EXTRA_ARGS >> ${CLIENT_LOG} 2>&1
        TEST_DURATION=$SECONDS
        set -e
        if [ $? -ne 0 ]; then
            cat ${CLIENT_LOG}
            RET=1
            echo -e "\n***\n*** Test FAILED\n***"
        else
            python3 ../common/check_valgrind_log.py -f $LEAKCHECK_LOG
            if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Memory leak detected\n***"
            RET=1
            fi

            set +e
            # Check for memory growth
            python $MASSIF_TEST $MASSIF_LOG $MAX_ALLOWED_ALLOC >> ${CLIENT_LOG}.massif 2>&1
            if [ $? -ne 0 ]; then
                echo -e "\n***\n*** Massif Test for ${PROTOCOL} ${LANG} Failed\n***"
                RET=1
            fi

            # Log test duration, the graph for memory growth and the change between Average and Max memory usage
            hrs=$(printf "%02d" $((TEST_DURATION / 3600)))
            mins=$(printf "%02d" $(((TEST_DURATION / 60) % 60)))
            secs=$(printf "%02d" $((TEST_DURATION % 60)))
            echo -e "Test Duration: $hrs:$mins:$secs (HH:MM:SS)" >> ${GRAPH_LOG}
            cat ${CLIENT_LOG}.massif
            ms_print ${MASSIF_LOG} | head -n35 >> ${GRAPH_LOG}
            cat ${GRAPH_LOG}
            set -e
        fi

        # Stop Server
        kill $SERVER_PID
        wait $SERVER_PID
    done
done

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

# Run only if both TRITON_FROM and TRITON_TO_DL are set
if [[ ! -z "$TRITON_FROM" ]] && [[ ! -z "$TRITON_TO_DL" ]]; then
    python client_memory_mail.py "$EMAIL_SUBJECT"
fi

exit $RET
