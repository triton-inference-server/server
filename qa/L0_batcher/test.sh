#!/bin/bash
# Copyright 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

CLIENT_LOG="./client.log"
BATCHER_TEST=batcher_test.py
VERIFY_TIMESTAMPS=verify_timestamps.py
TEST_RESULT_FILE='test_results.txt'

if [ -z "$TEST_VALGRIND" ]; then
    TEST_VALGRIND="0"
fi

if [ -z "$TEST_CUDA_SHARED_MEMORY" ]; then
    TEST_CUDA_SHARED_MEMORY="0"
fi

# Add valgrind flag check
if [ "$TEST_VALGRIND" -eq 1 ]; then
    LEAKCHECK=/usr/bin/valgrind
    LEAKCHECK_ARGS_BASE="--leak-check=full --show-leak-kinds=definite --max-threads=3000"
    SERVER_TIMEOUT=3600
    rm -f *.valgrind.log

    NO_DELAY_TESTS="test_static_batch_preferred \
                        test_multi_batch_sum_gt_max_preferred \
                        test_multi_same_output0 \
                        test_multi_different_output_order"

    DELAY_TESTS="test_multi_batch_use_biggest_preferred \
                    test_multi_batch_use_best_preferred"

    DIFFERENT_SHAPE_TESTS="test_multi_batch_not_preferred_different_shape \
                                test_multi_batch_different_shape_allow_ragged"
fi

TF_VERSION=${TF_VERSION:=1}

# On windows the paths invoked by the script (running in WSL) must use
# /mnt/c when needed but the paths on the tritonserver command-line
# must be C:/ style.
if [[ "$(< /proc/sys/kernel/osrelease)" == *microsoft* ]]; then
    MODELDIR=${MODELDIR:=C:/models}
    DATADIR=${DATADIR:="/mnt/c/data/inferenceserver/${REPO_VERSION}"}
    BACKEND_DIR=${BACKEND_DIR:=C:/tritonserver/backends}
    SERVER=${SERVER:=/mnt/c/tritonserver/bin/tritonserver.exe}
    export WSLENV=$WSLENV:TRITONSERVER_DELAY_SCHEDULER
else
    MODELDIR=${MODELDIR:=`pwd`}
    DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
    TRITON_DIR=${TRITON_DIR:="/opt/tritonserver"}
    SERVER=${TRITON_DIR}/bin/tritonserver
    BACKEND_DIR=${TRITON_DIR}/backends
fi

SERVER_ARGS_EXTRA="--backend-directory=${BACKEND_DIR} --backend-config=tensorflow,version=${TF_VERSION}"
source ../common/util.sh

RET=0

# If BACKENDS not specified, set to all
BACKENDS=${BACKENDS:="graphdef savedmodel onnx libtorch plan python"}
export BACKENDS

# Basic batcher tests
NO_DELAY_TESTS=${NO_DELAY_TESTS:="test_static_batch_preferred \
                            test_static_batch_lt_any_preferred \
                            test_static_batch_not_preferred \
                            test_static_batch_gt_max_preferred \
                            test_multi_batch_not_preferred \
                            test_multi_batch_gt_max_preferred \
                            test_multi_batch_sum_gt_max_preferred \
                            test_multi_same_output0 \
                            test_multi_same_output1 \
                            test_multi_different_outputs \
                            test_multi_different_output_order"}

# Tests that use scheduler delay
DELAY_TESTS=${DELAY_TESTS:="test_multi_batch_delayed_sum_gt_max_preferred \
                        test_multi_batch_use_biggest_preferred \
                        test_multi_batch_use_best_preferred \
                        test_multi_batch_delayed_use_max_batch"}

# Tests with different shapes
DIFFERENT_SHAPE_TESTS=${DIFFERENT_SHAPE_TESTS:="test_multi_batch_not_preferred_different_shape \
                                        test_multi_batch_preferred_different_shape \
                                        test_multi_batch_different_shape_allow_ragged \
                                        test_multi_batch_different_shape"}

# Test with preferred batch sizes but default max_queue_delay
PREFERRED_BATCH_ONLY_TESTS=${PREFERRED_BATCH_ONLY_TESTS:="test_preferred_batch_only_aligned \
                                                    test_preferred_batch_only_unaligned \
                                                    test_preferred_batch_only_use_biggest_preferred \
                                                    test_preferred_batch_only_use_no_preferred_size"}

# Tests with varying delay for max queue but no preferred batch size
MAX_QUEUE_DELAY_ONLY_TESTS=${MAX_QUEUE_DELAY_ONLY_TESTS:="test_max_queue_delay_only_default \
                                                    test_max_queue_delay_only_non_default"}

# Setup non-variable-size model repository
rm -fr *.log *.serverlog models && mkdir models
for BACKEND in $BACKENDS; do
    TMP_MODEL_DIR="$DATADIR/qa_model_repository/${BACKEND}_float32_float32_float32"
    if [ "$BACKEND" == "python" ]; then
        # We will be using ONNX models config.pbtxt and tweak them to make them
        # appropriate for Python backend
        onnx_model="${DATADIR}/qa_model_repository/onnx_float32_float32_float32"
        python_model=`echo $onnx_model | sed 's/onnx/python/g' | sed 's,'"$DATADIR/qa_model_repository/"',,g'`
        mkdir -p models/$python_model/1/
        cat $onnx_model/config.pbtxt | sed 's/platform:.*/backend:\ "python"/g' | sed 's/onnx/python/g' > models/$python_model/config.pbtxt
        cp $onnx_model/output0_labels.txt models/$python_model
        cp ../python_models/add_sub/model.py models/$python_model/1/
    else
        cp -r $TMP_MODEL_DIR models/. 
    fi
    (cd models/$(basename $TMP_MODEL_DIR) && \
          sed -i "s/^max_batch_size:.*/max_batch_size: 8/" config.pbtxt && \
          sed -i "s/^version_policy:.*/version_policy: { specific { versions: [1] }}/" config.pbtxt && \
          echo "dynamic_batching { preferred_batch_size: [ 2, 6 ], max_queue_delay_microseconds: 10000000 }" >> config.pbtxt)
done

rm -fr preferred_batch_only_models && mkdir preferred_batch_only_models
for BACKEND in $BACKENDS; do
    TMP_MODEL_DIR="$DATADIR/qa_model_repository/${BACKEND}_float32_float32_float32"
    if [ "$BACKEND" == "python" ]; then
        # We will be using ONNX models config.pbtxt and tweak them to make them
        # appropriate for Python backend
        onnx_model="${DATADIR}/qa_model_repository/onnx_float32_float32_float32"
        python_model=`echo $onnx_model | sed 's/onnx/python/g' | sed 's,'"$DATADIR/qa_model_repository/"',,g'`
        mkdir -p preferred_batch_only_models/$python_model/1/
        cat $onnx_model/config.pbtxt | sed 's/platform:.*/backend:\ "python"/g' | sed 's/onnx/python/g' > preferred_batch_only_models/$python_model/config.pbtxt
        cp $onnx_model/output0_labels.txt preferred_batch_only_models/$python_model
        cp ../python_models/add_sub/model.py preferred_batch_only_models/$python_model/1/
    else
        cp -r $TMP_MODEL_DIR preferred_batch_only_models/.
    fi
    (cd preferred_batch_only_models/$(basename $TMP_MODEL_DIR) && \
          sed -i "s/^max_batch_size:.*/max_batch_size: 8/" config.pbtxt && \
          sed -i "s/^version_policy:.*/version_policy: { specific { versions: [1] }}/" config.pbtxt && \
          echo "dynamic_batching { preferred_batch_size: [ 4, 6 ] }" >> config.pbtxt)
done

# Setup variable-size model repository
rm -fr var_models && mkdir var_models
for BACKEND in $BACKENDS; do
    TMP_MODEL_DIR="$DATADIR/qa_variable_model_repository/${BACKEND}_float32_float32_float32"
    if [ "$BACKEND" == "python" ]; then
        # We will be using ONNX models config.pbtxt and tweak them to make them
        # appropriate for Python backend
        onnx_model="${DATADIR}/qa_variable_model_repository/onnx_float32_float32_float32"
        python_model=`echo $onnx_model | sed 's/onnx/python/g' | sed 's,'"$DATADIR/qa_variable_model_repository/"',,g'`
        mkdir -p var_models/$python_model/1/
        cat $onnx_model/config.pbtxt | sed 's/platform:.*/backend:\ "python"/g' | sed 's/onnx/python/g' > var_models/$python_model/config.pbtxt
        cp $onnx_model/output0_labels.txt var_models/$python_model
        cp ../python_models/add_sub/model.py var_models/$python_model/1/
    else
        cp -r $TMP_MODEL_DIR var_models/.
    fi
    (cd var_models/$(basename $TMP_MODEL_DIR) && \
            sed -i "s/^max_batch_size:.*/max_batch_size: 8/" config.pbtxt && \
            sed -i "s/^version_policy:.*/version_policy: { specific { versions: [1] }}/" config.pbtxt && \
            echo "dynamic_batching { preferred_batch_size: [ 2, 6 ], max_queue_delay_microseconds: 10000000 }" >> config.pbtxt)
done

for MC in `ls var_models/*/config.pbtxt`; do
    sed -i "s/16/-1/g" $MC
done

# Create allow-ragged model to variable-size model repository
cp -r ../custom_models/custom_zero_1_float32 var_models/. && \
    (cd var_models/custom_zero_1_float32 && mkdir 1 && \
        echo "instance_group [ { kind: KIND_GPU count: 1 }]" >> config.pbtxt && \
        sed -i "s/^max_batch_size:.*/max_batch_size: 8/" config.pbtxt && \
        sed -i "s/dims:.*\[.*\]/dims: \[ -1 \]/g" config.pbtxt && \
        sed -i "s/name:.*\"INPUT0\"/name: \"INPUT0\"\\nallow_ragged_batch: true/" config.pbtxt && \
        sed -i "s/^version_policy:.*/version_policy: { specific { versions: [1] }}/" config.pbtxt && \
        echo "dynamic_batching { preferred_batch_size: [ 2, 6 ], max_queue_delay_microseconds: 10000000 }" >> config.pbtxt)

if [[ $BACKENDS == *"plan"* ]]; then
    # Use nobatch model to match the ragged test requirement
    cp -r $DATADIR/qa_identity_model_repository/plan_nobatch_zero_1_float32 var_models/plan_zero_1_float32 && \
        (cd var_models/plan_zero_1_float32 && \
            sed -i "s/nobatch_//" config.pbtxt && \
            sed -i "s/^max_batch_size:.*/max_batch_size: 8/" config.pbtxt && \
            sed -i "s/name: \"INPUT0\"/name: \"INPUT0\"\\nallow_ragged_batch: true/" config.pbtxt && \
            echo "batch_output [{target_name: \"OUTPUT0\" \
                                    kind: BATCH_SCATTER_WITH_INPUT_SHAPE \
                                    source_input: \"INPUT0\" }] \
                    dynamic_batching { preferred_batch_size: [ 2, 6 ], max_queue_delay_microseconds: 10000000 }" >> config.pbtxt)
fi

if [[ $BACKENDS == *"onnx"* ]]; then
    # Use nobatch model to match the ragged test requirement
    cp -r $DATADIR/qa_identity_model_repository/onnx_nobatch_zero_1_float32 var_models/onnx_zero_1_float32 && \
        (cd var_models/onnx_zero_1_float32 && \
            sed -i "s/nobatch_//" config.pbtxt && \
            sed -i "s/^max_batch_size:.*/max_batch_size: 8/" config.pbtxt && \
            sed -i "s/name: \"INPUT0\"/name: \"INPUT0\"\\nallow_ragged_batch: true/" config.pbtxt && \
            echo "batch_output [{target_name: \"OUTPUT0\" \
                                    kind: BATCH_SCATTER_WITH_INPUT_SHAPE \
                                    source_input: \"INPUT0\" }] \
                    dynamic_batching { preferred_batch_size: [ 2, 6 ], max_queue_delay_microseconds: 10000000 }" >> config.pbtxt)
fi

# Need to launch the server for each test so that the model status is
# reset (which is used to make sure the correctly batch size was used
# for execution). Test everything with fixed-tensor-size models and
# variable-tensor-size models.

for model_type in FIXED VARIABLE; do
    export BATCHER_TYPE=$model_type
    MODEL_PATH=models && [[ "$model_type" == "VARIABLE" ]] && MODEL_PATH=var_models
    for i in $NO_DELAY_TESTS ; do
        SERVER_ARGS="--model-repository=$MODELDIR/$MODEL_PATH ${SERVER_ARGS_EXTRA}"
        SERVER_LOG="./$i.$model_type.serverlog"

        if [ "$TEST_VALGRIND" -eq 1 ]; then
            LEAKCHECK_LOG="./$i.$model_type.valgrind.log"
            LEAKCHECK_ARGS="$LEAKCHECK_ARGS_BASE --log-file=$LEAKCHECK_LOG"
            run_server_leakcheck
        else
            run_server
        fi

        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        echo "Test: $i, $model_type" >>$CLIENT_LOG

        set +e
        python3 $BATCHER_TEST BatcherTest.$i >>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        else
            check_test_results $TEST_RESULT_FILE 1
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                echo -e "\n***\n*** Test Result Verification Failed\n***"
                RET=1
            fi
        fi
        set -e

        kill_server

        set +e
        if [ "$TEST_VALGRIND" -eq 1 ]; then
            python3 ../common/check_valgrind_log.py -f $LEAKCHECK_LOG
            if [ $? -ne 0 ]; then
                RET=1
            fi
        fi
        set -e
    done

    # Tests that require TRITONSERVER_DELAY_SCHEDULER so that the
    # scheduler is delayed and requests can collect in the queue.
    for i in $DELAY_TESTS ; do
        export TRITONSERVER_DELAY_SCHEDULER=6 &&
            [[ "$i" != "test_multi_batch_use_biggest_preferred" ]] && export TRITONSERVER_DELAY_SCHEDULER=3 &&
            [[ "$i" != "test_multi_batch_use_best_preferred" ]] &&
            [[ "$i" != "test_multi_batch_delayed_use_max_batch" ]] && export TRITONSERVER_DELAY_SCHEDULER=2
        SERVER_ARGS="--model-repository=$MODELDIR/$MODEL_PATH ${SERVER_ARGS_EXTRA}"
        SERVER_LOG="./$i.$model_type.serverlog"

        if [ "$TEST_VALGRIND" -eq 1 ]; then
            LEAKCHECK_LOG="./$i.$model_type.valgrind.log"
            LEAKCHECK_ARGS="$LEAKCHECK_ARGS_BASE --log-file=$LEAKCHECK_LOG"
            run_server_leakcheck
        else
            run_server
        fi

        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        echo "Test: $i" >>$CLIENT_LOG

        set +e
        python3 $BATCHER_TEST BatcherTest.$i >>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        else
            check_test_results $TEST_RESULT_FILE 1
            if [ $? -ne 0 ]; then
                cat $CLIENT_LOG
                echo -e "\n***\n*** Test Result Verification Failed\n***"
                RET=1
            fi
        fi
        set -e

        unset TRITONSERVER_DELAY_SCHEDULER
        kill_server

        set +e
        if [ "$TEST_VALGRIND" -eq 1 ]; then
            python3 ../common/check_valgrind_log.py -f $LEAKCHECK_LOG
            if [ $? -ne 0 ]; then
                RET=1
            fi
        fi
        set -e
    done
done

export BATCHER_TYPE=VARIABLE
for i in $DIFFERENT_SHAPE_TESTS ; do
    SERVER_ARGS="--model-repository=$MODELDIR/var_models ${SERVER_ARGS_EXTRA}"
    SERVER_LOG="./$i.VARIABLE.serverlog"

    if [ "$TEST_VALGRIND" -eq 1 ]; then
        LEAKCHECK_LOG="./$i.VARIABLE.valgrind.log"
        LEAKCHECK_ARGS="$LEAKCHECK_ARGS_BASE --log-file=$LEAKCHECK_LOG"
        run_server_leakcheck
    else
        run_server
    fi

    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    echo "Test: $i" >>$CLIENT_LOG

    set +e
    python3 $BATCHER_TEST BatcherTest.$i >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        check_test_results $TEST_RESULT_FILE 1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi
    set -e

    kill_server

    set +e
    if [ "$TEST_VALGRIND" -eq 1 ]; then
        python3 ../common/check_valgrind_log.py -f $LEAKCHECK_LOG
        if [ $? -ne 0 ]; then
            RET=1
        fi
    fi
    set -e
done

# Tests that run only on the variable-size tensor models and that
# require TRITONSERVER_DELAY_SCHEDULER so that the scheduler is delayed
# and requests can collect in the queue.
export BATCHER_TYPE=VARIABLE
for i in \
        test_multi_batch_delayed_preferred_different_shape ; do
    export TRITONSERVER_DELAY_SCHEDULER=4
    SERVER_ARGS="--model-repository=$MODELDIR/var_models ${SERVER_ARGS_EXTRA}"
    SERVER_LOG="./$i.VARIABLE.serverlog"

    if [ "$TEST_VALGRIND" -eq 1 ]; then
        LEAKCHECK_LOG="./$i.VARIABLE.valgrind.log"
        LEAKCHECK_ARGS="$LEAKCHECK_ARGS_BASE --log-file=$LEAKCHECK_LOG"
        run_server_leakcheck
    else
        run_server
    fi

    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    echo "Test: $i" >>$CLIENT_LOG

    set +e
    python3 $BATCHER_TEST BatcherTest.$i >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        check_test_results $TEST_RESULT_FILE 1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi
    set -e

    unset TRITONSERVER_DELAY_SCHEDULER
    kill_server

    set +e
    if [ "$TEST_VALGRIND" -eq 1 ]; then
        python3 ../common/check_valgrind_log.py -f $LEAKCHECK_LOG
        if [ $? -ne 0 ]; then
            RET=1
        fi
    fi
    set -e
done

export BATCHER_TYPE=FIXED
for i in $PREFERRED_BATCH_ONLY_TESTS ; do
    export TRITONSERVER_DELAY_SCHEDULER=4 &&
            [[ "$i" != "test_preferred_batch_only_aligned" ]] && export TRITONSERVER_DELAY_SCHEDULER=5 &&
            [[ "$i" != "test_preferred_batch_only_unaligned" ]] && export TRITONSERVER_DELAY_SCHEDULER=7 &&
            [[ "$i" != "test_preferred_batch_only_use_biggest_preferred" ]] && export TRITONSERVER_DELAY_SCHEDULER=3
    SERVER_ARGS="--model-repository=$MODELDIR/preferred_batch_only_models ${SERVER_ARGS_EXTRA}"
    SERVER_LOG="./$i.PREFERRED_BATCH_ONLY.serverlog"

    if [ "$TEST_VALGRIND" -eq 1 ]; then
        LEAKCHECK_LOG="./$i.PREFERRED_BATCH_ONLY.valgrind.log"
        LEAKCHECK_ARGS="$LEAKCHECK_ARGS_BASE --log-file=$LEAKCHECK_LOG"
        run_server_leakcheck
    else
        run_server
    fi

    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    echo "Test: $i" >>$CLIENT_LOG

    set +e
    python3 $BATCHER_TEST BatcherTest.$i >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        check_test_results $TEST_RESULT_FILE 1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi
    set -e

    unset TRITONSERVER_DELAY_SCHEDULER
    kill_server

    set +e
    if [ "$TEST_VALGRIND" -eq 1 ]; then
        python3 ../common/check_valgrind_log.py -f $LEAKCHECK_LOG
        if [ $? -ne 0 ]; then
            RET=1
        fi
    fi
    set -e
done

# Test cases that checks the runtime batches created with max_queue_delay
# specification only.
rm -fr ./custom_models && mkdir ./custom_models && \
cp -r ../custom_models/custom_zero_1_float32 ./custom_models/. && \
mkdir -p ./custom_models/custom_zero_1_float32/1

# Provide sufficient delay to allow forming of next batch.
(cd custom_models/custom_zero_1_float32 && \
        sed -i "s/dims:.*\[.*\]/dims: \[ -1 \]/g" config.pbtxt && \
        sed -i "s/max_batch_size:.*/max_batch_size: 100/g" config.pbtxt && \
        echo "dynamic_batching { max_queue_delay_microseconds: 0}" >> config.pbtxt && \
        echo "instance_group [ { kind: KIND_GPU } ]" >> config.pbtxt && \
        echo "parameters [" >> config.pbtxt && \
        echo "{ key: \"execute_delay_ms\"; value: { string_value: \"100\" }}" >> config.pbtxt && \
        echo "]" >> config.pbtxt)

for i in $MAX_QUEUE_DELAY_ONLY_TESTS ; do
    export MAX_QUEUE_DELAY_MICROSECONDS=20000 &&
        [[ "$i" != "test_max_queue_delay_only_non_default" ]] && export MAX_QUEUE_DELAY_MICROSECONDS=0
    (cd custom_models/custom_zero_1_float32 && \
        sed -i "s/max_queue_delay_microseconds:.*\[.*\]/max_queue_delay_microseconds: ${MAX_QUEUE_DELAY_MICROSECONDS}/g" config.pbtxt )

    SERVER_ARGS="--model-repository=$MODELDIR/custom_models ${SERVER_ARGS_EXTRA}"
    SERVER_LOG="./$i.MAX_QUEUE_DELAY_ONLY.serverlog"

    if [ "$TEST_VALGRIND" -eq 1 ]; then
        LEAKCHECK_LOG="./$i.MAX_QUEUE_DELAY_ONLY.valgrind.log"
        LEAKCHECK_ARGS="$LEAKCHECK_ARGS_BASE --log-file=$LEAKCHECK_LOG"
        run_server_leakcheck
    else
        run_server
    fi

    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    echo "Test: $i" >>$CLIENT_LOG

    set +e
    python3 $BATCHER_TEST BatcherTest.$i >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        check_test_results $TEST_RESULT_FILE 1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi
    set -e
    kill_server
    unset MAX_QUEUE_DELAY_MICROSECONDS
    set +e
    if [ "$TEST_VALGRIND" -eq 1 ]; then
        python3 ../common/check_valgrind_log.py -f $LEAKCHECK_LOG
        if [ $? -ne 0 ]; then
            RET=1
        fi
    fi
    set -e
done

# Test that verify the 'preserve_ordering' option in dynamic batcher
# Run the test scheme with and without preserve ordering, verify behavior
# by comparing the "response send" timestamps.
TEST_CASE=test_multi_batch_preserve_ordering

# Skip test for Windows. Trace file concats at 8192 chars on Windows.
if [[ "$(< /proc/sys/kernel/osrelease)" != *microsoft* ]]; then
    rm -fr ./custom_models && mkdir ./custom_models && \
        cp -r ../custom_models/custom_zero_1_float32 ./custom_models/. && \
        mkdir -p ./custom_models/custom_zero_1_float32/1

    # Two instances will be created for the custom model, one delays 100 ms while
    # the other delays 400 ms
    (cd custom_models/custom_zero_1_float32 && \
            sed -i "s/dims:.*\[.*\]/dims: \[ -1 \]/g" config.pbtxt && \
            sed -i "s/max_batch_size:.*/max_batch_size: 4/g" config.pbtxt && \
            echo "dynamic_batching { preferred_batch_size: [ 4 ] }" >> config.pbtxt && \
            echo "instance_group [ { kind: KIND_GPU count: 2 }]" >> config.pbtxt && \
            echo "parameters [" >> config.pbtxt && \
            echo "{ key: \"execute_delay_ms\"; value: { string_value: \"100\" }}," >> config.pbtxt && \
            echo "{ key: \"instance_wise_delay_multiplier\"; value: { string_value: \"4\" }}" >> config.pbtxt && \
            echo "]" >> config.pbtxt)

    # enqueue 3 batches to guarantee that a large delay batch will be followed by
    # a small delay one regardless of the order issued to model instances.
    # i.e. the 3 batches will be queued: [1, 2, 3] and there are two delay instances
    # [small, large], then the distributions can be the following:
    # [1:small 2:large 3:small] or [1:large 2:small 3:*] (* depends on whether order
    # is preserved), and we only interested in the timestamps where the large delay
    # batch is followed by small delay batch
    export TRITONSERVER_DELAY_SCHEDULER=12

    # not preserve
    SERVER_ARGS="--trace-file=not_preserve.log --trace-level=MIN --trace-rate=1 --model-repository=$MODELDIR/custom_models ${SERVER_ARGS_EXTRA}"
    SERVER_LOG="./not_preserve.serverlog"

    if [ "$TEST_VALGRIND" -eq 1 ]; then
        LEAKCHECK_LOG="./not_preserve.valgrind.log"
        LEAKCHECK_ARGS="$LEAKCHECK_ARGS_BASE --log-file=$LEAKCHECK_LOG"
        run_server_leakcheck
    else
        run_server
    fi

    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    echo "Test: not_preserve" >>$CLIENT_LOG

    set +e
    python3 $BATCHER_TEST BatcherTest.$TEST_CASE >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        check_test_results $TEST_RESULT_FILE 1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi
    set -e

    kill_server

    set +e
    if [ "$TEST_VALGRIND" -eq 1 ]; then
        python3 ../common/check_valgrind_log.py -f $LEAKCHECK_LOG
        if [ $? -ne 0 ]; then
            RET=1
        fi
    fi

    python3 $VERIFY_TIMESTAMPS not_preserve.log
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    set -e

    # preserve
    (cd custom_models/custom_zero_1_float32 && \
            sed -i "s/dynamic_batching.*/dynamic_batching { preferred_batch_size: [ 4 ] preserve_ordering: true }/g" config.pbtxt)

    SERVER_ARGS="--trace-file=preserve.log --trace-level=MIN --trace-rate=1 --model-repository=$MODELDIR/custom_models  ${SERVER_ARGS_EXTRA}"
    SERVER_LOG="./preserve.serverlog"

    if [ "$TEST_VALGRIND" -eq 1 ]; then
        LEAKCHECK_LOG="./preserve.valgrind.log"
        LEAKCHECK_ARGS="$LEAKCHECK_ARGS_BASE --log-file=$LEAKCHECK_LOG"
        run_server_leakcheck
    else
        run_server
    fi

    if [ "$SERVER_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $SERVER\n***"
        cat $SERVER_LOG
        exit 1
    fi

    echo "Test: preserve" >>$CLIENT_LOG

    set +e
    python3 $BATCHER_TEST BatcherTest.$TEST_CASE >>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    else
        check_test_results $TEST_RESULT_FILE 1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Result Verification Failed\n***"
            RET=1
        fi
    fi
    set -e

    kill_server

    set +e
    if [ "$TEST_VALGRIND" -eq 1 ]; then
        python3 ../common/check_valgrind_log.py -f $LEAKCHECK_LOG
        if [ $? -ne 0 ]; then
            RET=1
        fi
    fi

    python3 $VERIFY_TIMESTAMPS -p preserve.log
    if [ $? -ne 0 ]; then
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    set -e
    unset TRITONSERVER_DELAY_SCHEDULER
fi

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET

