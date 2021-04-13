#!/bin/bash
# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
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

# Must run on a single device or else the TRITONSERVER_DELAY_SCHEDULER
# can fail when the requests are distributed to multiple devices.
export CUDA_VISIBLE_DEVICES=0

CLIENT_LOG="./client.log"
BATCHER_TEST=batcher_test.py
VERIFY_TIMESTAMPS=verify_timestamps.py

if [ -z "$TEST_VALGRIND" ]; then
    TEST_VALGRIND="0"
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
if [[ "$(< /proc/sys/kernel/osrelease)" == *Microsoft ]]; then
    MODELDIR=${MODELDIR:=C:/models}
    DATADIR=${DATADIR:="/mnt/c/data/inferenceserver/${REPO_VERSION}"}
    BACKEND_DIR=${BACKEND_DIR:=C:/tritonserver/backends}
    SERVER=${SERVER:=/mnt/c/tritonserver/bin/tritonserver.exe}
    export USE_HTTP=0
    export WSLENV=$WSLENV:TRITONSERVER_DELAY_SCHEDULER
else
    MODELDIR=${MODELDIR:=`pwd`}
    DATADIR=${DATADIR:="/data/inferenceserver/${REPO_VERSION}"}
    OPTDIR=${OPTDIR:="/opt"}
    BACKEND_DIR=${OPTDIR}/tritonserver/backends
    SERVER=${OPTDIR}/tritonserver/bin/tritonserver
fi

SERVER_ARGS_EXTRA="--backend-directory=${BACKEND_DIR} --backend-config=tensorflow,version=${TF_VERSION}"
source ../common/util.sh

RET=0

# If BACKENDS not specified, set to all
BACKENDS=${BACKENDS:="graphdef savedmodel onnx libtorch plan custom"}
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

# Setup non-variable-size model repository
rm -fr *.log *.serverlog models && mkdir models
for BACKEND in $BACKENDS; do
    if [ "$BACKEND" != "custom" ]; then
      TMP_MODEL_DIR="$DATADIR/qa_model_repository/${BACKEND}_float32_float32_float32"
    else
      TMP_MODEL_DIR="../custom_models/custom_float32_float32_float32"
    fi

    cp -r $TMP_MODEL_DIR models/. &&
    (cd models/$(basename $TMP_MODEL_DIR) && \
          sed -i "s/^max_batch_size:.*/max_batch_size: 8/" config.pbtxt && \
          sed -i "s/^version_policy:.*/version_policy: { specific { versions: [1] }}/" config.pbtxt && \
          echo "dynamic_batching { preferred_batch_size: [ 2, 6 ], max_queue_delay_microseconds: 10000000 }" >> config.pbtxt)
done

# Setup variable-size model repository
rm -fr var_models && mkdir var_models
for BACKEND in $BACKENDS; do
    if [ "$BACKEND" != "custom" ]; then
      TMP_MODEL_DIR="$DATADIR/qa_variable_model_repository/${BACKEND}_float32_float32_float32"
    else
      TMP_MODEL_DIR="../custom_models/custom_float32_float32_float32 ../custom_models/custom_zero_1_float32"
    fi

    for TMP_DIR in $TMP_MODEL_DIR; do
      cp -r $TMP_DIR var_models/. &&
        (cd var_models/$(basename $TMP_DIR) && \
            sed -i "s/^max_batch_size:.*/max_batch_size: 8/" config.pbtxt && \
            sed -i "s/^version_policy:.*/version_policy: { specific { versions: [1] }}/" config.pbtxt && \
            echo "dynamic_batching { preferred_batch_size: [ 2, 6 ], max_queue_delay_microseconds: 10000000 }" >> config.pbtxt)
    done
done

for MC in `ls var_models/*/config.pbtxt`; do
    sed -i "s/16/-1/g" $MC
done

# Create allow-ragged model to variable-size model repository
if [[ $BACKENDS == *"custom"* ]]; then
    (cd var_models/custom_zero_1_float32 && \
        mkdir -p 1 && cp ../../libtriton_identity.so 1/libcustom.so && \
        echo "instance_group [ { kind: KIND_CPU count: 1 }]" >> config.pbtxt && \
        sed -i "s/dims:.*\[.*\]/dims: \[ -1 \]/g" config.pbtxt && \
        sed -i "s/name:.*\"INPUT0\"/name: \"INPUT0\"\\nallow_ragged_batch: true/" config.pbtxt)
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
        elif [[ "$(< /proc/sys/kernel/osrelease)" == *Microsoft ]]; then
            # We rely on HTTP endpoint in run_server so until HTTP is
            # implemented for win we do this hack...
            run_server_nowait
            sleep 15
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
            check_test_results $CLIENT_LOG 1
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
        elif [[ "$(< /proc/sys/kernel/osrelease)" == *Microsoft ]]; then
            # We rely on HTTP endpoint in run_server so until HTTP is
            # implemented for win we do this hack...
            run_server_nowait
            sleep 15
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
            check_test_results $CLIENT_LOG 1
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
    elif [[ "$(< /proc/sys/kernel/osrelease)" == *Microsoft ]]; then
        # We rely on HTTP endpoint in run_server so until HTTP is
        # implemented for win we do this hack...
        run_server_nowait
        sleep 15
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
        check_test_results $CLIENT_LOG 1
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
    elif [[ "$(< /proc/sys/kernel/osrelease)" == *Microsoft ]]; then
        # We rely on HTTP endpoint in run_server so until HTTP is
        # implemented for win we do this hack...
        run_server_nowait
        sleep 15
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
        check_test_results $CLIENT_LOG 1
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

# Test that verify the 'preserve_ordering' option in dynamic batcher
# Run the test scheme with and without preserve ordering, verify behavior
# by comparing the "response send" timestamps.
TEST_CASE=test_multi_batch_preserve_ordering

if [[ $BACKENDS == *"custom"* ]]; then
    rm -fr ./custom_models && mkdir ./custom_models && \
        cp -r ../custom_models/custom_zero_1_float32 ./custom_models/. && \
        mkdir -p ./custom_models/custom_zero_1_float32/1 && \
        cp ./libtriton_identity.so ./custom_models/custom_zero_1_float32/1/libcustom.so

    # Two instances will be created for the custom model, one delays 100 ms while
    # the other delays 400 ms
    (cd custom_models/custom_zero_1_float32 && \
            sed -i "s/dims:.*\[.*\]/dims: \[ -1 \]/g" config.pbtxt && \
            sed -i "s/max_batch_size:.*/max_batch_size: 4/g" config.pbtxt && \
            echo "dynamic_batching { preferred_batch_size: [ 4 ] }" >> config.pbtxt && \
            echo "instance_group [ { kind: KIND_CPU count: 2 }]" >> config.pbtxt && \
            echo "parameters [" >> config.pbtxt && \
            echo "{ key: \"execute_delay_ms\"; value: { string_value: \"100\" }}," >> config.pbtxt && \
            echo "{ key: \"instance_wise_delay_multiplier\"; value: { string_value: \"4\" }}" >> config.pbtxt && \
            echo "]" >> config.pbtxt)

    # equeue 3 batches to guarantee that a large delay batch will be followed by
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
    elif [[ "$(< /proc/sys/kernel/osrelease)" == *Microsoft ]]; then
        # We rely on HTTP endpoint in run_server so until HTTP is
        # implemented for win we do this hack...
        run_server_nowait
        sleep 15
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
        check_test_results $CLIENT_LOG 1
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
    elif [[ "$(< /proc/sys/kernel/osrelease)" == *Microsoft ]]; then
        # We rely on HTTP endpoint in run_server so until HTTP is
        # implemented for win we do this hack...
        run_server_nowait
        sleep 15
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
        check_test_results $CLIENT_LOG 1
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
