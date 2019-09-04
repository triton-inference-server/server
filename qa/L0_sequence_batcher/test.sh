#!/bin/bash
# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

CLIENT_LOG="./client.log"
BATCHER_TEST=sequence_batcher_test.py

DATADIR=/data/inferenceserver/$1

SERVER=/opt/tensorrtserver/bin/trtserver
source ../common/util.sh

RET=0

# Must run on a single device or else the TRTSERVER_DELAY_SCHEDULER
# can fail when the requests are distributed to multiple devices.
export CUDA_VISIBLE_DEVICES=0

# Setup non-variable-size model repositories. The same models are in each
# repository but they are configured as:
#   models0 - four instance with non-batching model
#   models1 - one instance with batch-size 4
#   models2 - two instances with batch-size 2
#   models4 - four instances with batch-size 1
rm -fr *.log *.serverlog models{0,1,2,4} && mkdir models{0,1,2,4}
for m in \
        $DATADIR/qa_sequence_model_repository/plan_sequence_float32 \
        $DATADIR/qa_sequence_model_repository/netdef_sequence_int32 \
        $DATADIR/qa_sequence_model_repository/graphdef_sequence_object \
        $DATADIR/qa_sequence_model_repository/savedmodel_sequence_float32 \
        $DATADIR/qa_sequence_model_repository/onnx_sequence_int32 \
        $DATADIR/qa_ensemble_model_repository/qa_sequence_model_repository/*_plan_sequence_float32 \
        $DATADIR/qa_ensemble_model_repository/qa_sequence_model_repository/*_netdef_sequence_int32 \
        $DATADIR/qa_ensemble_model_repository/qa_sequence_model_repository/*_graphdef_sequence_object \
        $DATADIR/qa_ensemble_model_repository/qa_sequence_model_repository/*_savedmodel_sequence_float32 \
        ../custom_models/custom_sequence_int32 \
        $DATADIR/qa_sequence_model_repository/libtorch_sequence_int32 ; do
    cp -r $m models1/. && \
        (cd models1/$(basename $m) && \
            sed -i "s/^max_batch_size:.*/max_batch_size: 4/" config.pbtxt && \
            sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 1/" config.pbtxt && \
            sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 1/" config.pbtxt)
    cp -r $m models2/. && \
        (cd models2/$(basename $m) && \
            sed -i "s/^max_batch_size:.*/max_batch_size: 2/" config.pbtxt && \
            sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 2/" config.pbtxt && \
            sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 2/" config.pbtxt)
    cp -r $m models4/. && \
        (cd models4/$(basename $m) && \
            sed -i "s/^max_batch_size:.*/max_batch_size: 1/" config.pbtxt && \
            sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 4/" config.pbtxt && \
            sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 4/" config.pbtxt)
done

for m in \
        $DATADIR/qa_sequence_model_repository/plan_nobatch_sequence_float32 \
        $DATADIR/qa_sequence_model_repository/netdef_nobatch_sequence_int32 \
        $DATADIR/qa_sequence_model_repository/graphdef_nobatch_sequence_object \
        $DATADIR/qa_sequence_model_repository/savedmodel_nobatch_sequence_float32 \
        $DATADIR/qa_sequence_model_repository/onnx_nobatch_sequence_int32 \
        $DATADIR/qa_ensemble_model_repository/qa_sequence_model_repository/*_plan_nobatch_sequence_float32 \
        $DATADIR/qa_ensemble_model_repository/qa_sequence_model_repository/*_netdef_nobatch_sequence_int32 \
        $DATADIR/qa_ensemble_model_repository/qa_sequence_model_repository/*_graphdef_nobatch_sequence_object \
        $DATADIR/qa_ensemble_model_repository/qa_sequence_model_repository/*_savedmodel_nobatch_sequence_float32 \
        $DATADIR/qa_sequence_model_repository/libtorch_nobatch_sequence_int32 ; do
    cp -r $m models0/. && \
        (cd models0/$(basename $m) && \
            sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 4/" config.pbtxt && \
            sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 4/" config.pbtxt)
done

# Setup variable-size model repository.
#   modelsv - one instance with batch-size 4
rm -fr modelsv && mkdir modelsv
for m in \
        $DATADIR/qa_variable_sequence_model_repository/netdef_sequence_int32 \
        $DATADIR/qa_variable_sequence_model_repository/graphdef_sequence_object \
        $DATADIR/qa_variable_sequence_model_repository/savedmodel_sequence_float32 \
        $DATADIR/qa_variable_sequence_model_repository/onnx_sequence_int32 \
        $DATADIR/qa_ensemble_model_repository/qa_variable_sequence_model_repository/*_netdef_sequence_int32 \
        $DATADIR/qa_ensemble_model_repository/qa_variable_sequence_model_repository/*_graphdef_sequence_object \
        $DATADIR/qa_ensemble_model_repository/qa_variable_sequence_model_repository/*_savedmodel_sequence_float32 \
        $DATADIR/qa_variable_sequence_model_repository/libtorch_sequence_int32 ; do
    cp -r $m modelsv/. && \
        (cd modelsv/$(basename $m) && \
            sed -i "s/^max_batch_size:.*/max_batch_size: 4/" config.pbtxt && \
            sed -i "s/kind: KIND_GPU/kind: KIND_GPU\\ncount: 1/" config.pbtxt && \
            sed -i "s/kind: KIND_CPU/kind: KIND_CPU\\ncount: 1/" config.pbtxt)
done

# Same test work on all models since they all have same total number
# of batch slots.
for model_trial in v 0 1 2 4 ; do
    export NO_BATCHING=1 &&
        [[ "$model_trial" != "0" ]] && export NO_BATCHING=0
    export MODEL_INSTANCES=1 &&
        [[ "$model_trial" != "v" ]] && export MODEL_INSTANCES=4 &&
        [[ "$model_trial" != "0" ]] && export MODEL_INSTANCES=$model_trial

    MODEL_DIR=models${model_trial}

    cp -r $DATADIR/qa_ensemble_model_repository/qa_sequence_model_repository/nop_* `pwd`/$MODEL_DIR/.
    create_nop_modelfile `pwd`/libidentity.so `pwd`/$MODEL_DIR

    # Need to launch the server for each test so that the model status is
    # reset (which is used to make sure the correctly batch size was used
    # for execution). Test everything with fixed-tensor-size models and
    # variable-tensor-size models.
    export BATCHER_TYPE="VARIABLE" &&
        [[ "$model_trial" != "v" ]] && export BATCHER_TYPE="FIXED"

    for i in \
            test_simple_sequence \
            test_length1_sequence \
            test_batch_size \
            test_no_sequence_start \
            test_no_sequence_start2 \
            test_no_sequence_end \
            test_no_correlation_id ; do
        SERVER_ARGS="--model-repository=`pwd`/$MODEL_DIR"
        SERVER_LOG="./$i.$MODEL_DIR.serverlog"
        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        echo "Test: $i" >>$CLIENT_LOG

        set +e
        python $BATCHER_TEST SequenceBatcherTest.$i >>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        set -e

        kill $SERVER_PID
        wait $SERVER_PID
    done

    # Tests that require TRTSERVER_DELAY_SCHEDULER so that the
    # scheduler is delayed and requests can collect in the queue.
    for i in \
        test_backlog_fill \
            test_backlog_fill_no_end \
            test_backlog_same_correlation_id \
            test_backlog_same_correlation_id_no_end \
            test_backlog_sequence_timeout \
            test_half_batch \
            test_skip_batch \
            test_full_batch \
            test_backlog ; do
        export TRTSERVER_BACKLOG_DELAY_SCHEDULER=3 &&
            [[ "$i" != "test_backlog_fill_no_end" ]] && export TRTSERVER_BACKLOG_DELAY_SCHEDULER=2 &&
            [[ "$i" != "test_backlog_fill" ]] &&
            [[ "$i" != "test_backlog_same_correlation_id" ]] && export TRTSERVER_BACKLOG_DELAY_SCHEDULER=0
        export TRTSERVER_DELAY_SCHEDULER=10 &&
            [[ "$i" != "test_backlog_fill_no_end" ]] &&
            [[ "$i" != "test_backlog_fill" ]] && export TRTSERVER_DELAY_SCHEDULER=16 &&
            [[ "$i" != "test_backlog_same_correlation_id_no_end" ]] && export TRTSERVER_DELAY_SCHEDULER=8 &&
            [[ "$i" != "test_half_batch" ]] && export TRTSERVER_DELAY_SCHEDULER=4 &&
            [[ "$i" != "test_backlog_sequence_timeout" ]] && export TRTSERVER_DELAY_SCHEDULER=12
        SERVER_ARGS="--model-repository=`pwd`/$MODEL_DIR"
        SERVER_LOG="./$i.$MODEL_DIR.serverlog"
        run_server
        if [ "$SERVER_PID" == "0" ]; then
            echo -e "\n***\n*** Failed to start $SERVER\n***"
            cat $SERVER_LOG
            exit 1
        fi

        echo "Test: $i" >>$CLIENT_LOG

        set +e
        python $BATCHER_TEST SequenceBatcherTest.$i >>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        set -e

        unset TRTSERVER_DELAY_SCHEDULER
        unset TRTSERVER_BACKLOG_DELAY_SCHEDULER
        kill $SERVER_PID
        wait $SERVER_PID
    done
done

# python unittest seems to swallow ImportError and still return 0 exit
# code. So need to explicitly check CLIENT_LOG to make sure we see
# some running tests
grep -c "HTTP/1.1 200 OK" $CLIENT_LOG
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed To Run\n***"
    RET=1
fi

if [ $RET -eq 0 ]; then
    echo -e "\n***\n*** Test Passed\n***"
else
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
