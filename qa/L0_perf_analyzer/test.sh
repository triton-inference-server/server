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

export CUDA_VISIBLE_DEVICES=0

CLIENT_LOG="./perf_analyzer.log"
PERF_ANALYZER=../clients/perf_analyzer

DATADIR=`pwd`/models
TESTDATADIR=`pwd`/test_data

INT_JSONDATAFILE=`pwd`/../common/perf_analyzer_input_data_json/int_data.json
INT_DIFFSHAPE_JSONDATAFILE=`pwd`/../common/perf_analyzer_input_data_json/int_data_diff_shape.json
FLOAT_DIFFSHAPE_JSONDATAFILE=`pwd`/../common/perf_analyzer_input_data_json/float_data_with_shape.json
STRING_JSONDATAFILE=`pwd`/../common/perf_analyzer_input_data_json/string_data.json
STRING_WITHSHAPE_JSONDATAFILE=`pwd`/../common/perf_analyzer_input_data_json/string_data_with_shape.json
SEQ_JSONDATAFILE=`pwd`/../common/perf_analyzer_input_data_json/seq_data.json
SHAPETENSORADTAFILE=`pwd`/../common/perf_analyzer_input_data_json/shape_tensor_data.json
IMAGE_JSONDATAFILE=`pwd`/../common/perf_analyzer_input_data_json/image_data.json

OUTPUT_JSONDATAFILE=`pwd`/../common/perf_analyzer_input_data_json/output.json
NON_ALIGNED_OUTPUT_JSONDATAFILE=`pwd`/../common/perf_analyzer_input_data_json/non_aligned_output.json
WRONG_OUTPUT_JSONDATAFILE=`pwd`/../common/perf_analyzer_input_data_json/wrong_output.json
WRONG_OUTPUT_2_JSONDATAFILE=`pwd`/../common/perf_analyzer_input_data_json/wrong_output_2.json

SEQ_OUTPUT_JSONDATAFILE=`pwd`/../common/perf_analyzer_input_data_json/seq_output.json
SEQ_WRONG_OUTPUT_JSONDATAFILE=`pwd`/../common/perf_analyzer_input_data_json/seq_wrong_output.json

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=${DATADIR}"
SERVER_LOG="./inference_server.log"

ERROR_STRING="error | Request count: 0 | : 0 infer/sec"

source ../common/util.sh

rm -f $SERVER_LOG $CLIENT_LOG
rm -rf $DATADIR $TESTDATADIR $ENSEMBLE_DATADIR

mkdir -p $DATADIR
# Copy fixed-shape models
cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/graphdef_int32_int32_int32 $DATADIR/
cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/graphdef_nobatch_int32_int32_int32 $DATADIR/
cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/graphdef_object_object_object $DATADIR/
cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/graphdef_nobatch_object_object_object $DATADIR/

# Copy a variable-shape models
cp -r /data/inferenceserver/${REPO_VERSION}/qa_variable_model_repository/graphdef_object_int32_int32 $DATADIR/
cp -r /data/inferenceserver/${REPO_VERSION}/qa_variable_model_repository/graphdef_int32_int32_float32 $DATADIR/

# Copy shape tensor models
cp -r /data/inferenceserver/${REPO_VERSION}/qa_shapetensor_model_repository/plan_zero_1_float32 $DATADIR/

# Copying ensemble including a sequential model
cp -r /data/inferenceserver/${REPO_VERSION}/qa_sequence_model_repository/savedmodel_sequence_object $DATADIR
cp -r /data/inferenceserver/${REPO_VERSION}/qa_ensemble_model_repository/qa_sequence_model_repository/simple_savedmodel_sequence_object $DATADIR
cp -r /data/inferenceserver/${REPO_VERSION}/qa_ensemble_model_repository/qa_sequence_model_repository/nop_TYPE_FP32_-1 $DATADIR

# Copying variable sequence model
cp -r /data/inferenceserver/${REPO_VERSION}/qa_variable_sequence_model_repository/graphdef_sequence_float32 $DATADIR

mkdir $DATADIR/nop_TYPE_FP32_-1/1

# Copy inception model to the model repository
cp -r /data/inferenceserver/${REPO_VERSION}/tf_model_store/inception_v1_graphdef $DATADIR

# Copy resnet50v1.5_fp16
cp -r /data/inferenceserver/${REPO_VERSION}/perf_model_store/resnet50v1.5_fp16_savedmodel $DATADIR

# Generating test data
mkdir -p $TESTDATADIR
for INPUT in INPUT0 INPUT1; do
    for i in {1..16}; do
        echo '1' >> $TESTDATADIR/${INPUT}
    done
done

RET=0

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi


# Test whether there was a conflict in sending sequences. This should
# be done before other testing as the server might emit this warning
# in certain test cases that are expected to raise this warning
SERVER_ERROR_STRING="The previous sequence did not end before this sequence start"

set +e
$PERF_ANALYZER -v -i $PROTOCOL -m graphdef_object_object_object -p2000 >$CLIENT_LOG 2>&1
if [ $? -eq 0 ]; then
  cat $CLIENT_LOG
  echo -e "\n***\n*** Test Failed: Expected an error when using dynamic shapes in string inputs\n***"
  RET=1
fi
if [ $(cat $CLIENT_LOG |  grep "input INPUT0 contains dynamic shape, provide shapes to send along with the request" | wc -l) -ne 0 ]; then
  cat $CLIENT_LOG
  echo -e "\n***\n*** Test Failed: \n***"
  RET=1
fi

# Testing with ensemble and sequential model variants
$PERF_ANALYZER -v -i grpc -m  simple_savedmodel_sequence_object -p 2000 -t5 --streaming \
--input-data=$SEQ_JSONDATAFILE  --input-data=$SEQ_JSONDATAFILE >$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed: Sequence conflict when maintaining concurrency\n***"
    RET=1
fi

$PERF_ANALYZER -v -i grpc -m  simple_savedmodel_sequence_object -p 1000 --request-rate-range 100:200:50 --streaming \
--input-data=$SEQ_JSONDATAFILE >$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

if [ $(cat $SERVER_LOG |  grep "${SERVER_ERROR_STRING}" | wc -l) -ne 0 ]; then
    cat $SERVER_LOG |  grep "${SERVER_ERROR_STRING}"
    echo -e "\n***\n*** Test Failed: Sequence conflict\n***"
    RET=1
fi
set -e

for PROTOCOL in grpc http; do

    # Testing simple configurations with different shared memory types
    for SHARED_MEMORY_TYPE in none system cuda; do
        set +e
        $PERF_ANALYZER -v -i $PROTOCOL -m graphdef_int32_int32_int32 -t 1 -p2000 -b 1 \
    --shared-memory=$SHARED_MEMORY_TYPE >$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        if [ $(cat $CLIENT_LOG | grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi

        $PERF_ANALYZER -v -i $PROTOCOL -m graphdef_int32_int32_int32 -t 1 -p2000 -b 1 -a \
    --shared-memory=$SHARED_MEMORY_TYPE>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        if [ $(cat $CLIENT_LOG | grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        set -e
    done

    # TODO Add back testing with preprocess_inception_ensemble model

    # Testing with inception model
    for SHARED_MEMORY_TYPE in none system cuda; do
        set +e
        $PERF_ANALYZER -v -i $PROTOCOL -m inception_v1_graphdef -t 1 -p2000 -b 1 \
    --shared-memory=$SHARED_MEMORY_TYPE >$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        if [ $(cat $CLIENT_LOG | grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi

        $PERF_ANALYZER -v -i $PROTOCOL -m inception_v1_graphdef -t 1 -p2000 -b 1 -a \
    --shared-memory=$SHARED_MEMORY_TYPE>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        if [ $(cat $CLIENT_LOG | grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        set -e
    done

    # Testing with resnet50 models with large batch sizes
    for SHARED_MEMORY_TYPE in none system cuda; do
        set +e
        $PERF_ANALYZER -v -i $PROTOCOL -m inception_v1_graphdef -t 2 -p2000 -b 64 \
    --shared-memory=$SHARED_MEMORY_TYPE >$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        if [ $(cat $CLIENT_LOG | grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi

        $PERF_ANALYZER -v -i $PROTOCOL -m inception_v1_graphdef -t 2 -p2000 -b 64 \
    --shared-memory=$SHARED_MEMORY_TYPE -a >$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        if [ $(cat $CLIENT_LOG | grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        set -e
    done

    # Test perf client behavior on different model with different batch size
    for MODEL in graphdef_nobatch_int32_int32_int32 graphdef_int32_int32_int32; do
        # Valid batch size
        set +e
        $PERF_ANALYZER -v -i $PROTOCOL -m $MODEL -t 1 -p2000 -b 1 >$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        set -e

        # Invalid batch sizes
        for STATIC_BATCH in 0 10; do
            set +e
            $PERF_ANALYZER -v -i $PROTOCOL -m $MODEL -t 1 -p2000 -b $STATIC_BATCH >$CLIENT_LOG 2>&1
            if [ $? -eq 0 ]; then
                cat $CLIENT_LOG
                echo -e "\n***\n*** Test Failed\n***"
                RET=1
            fi
            set -e
        done
    done

    # Testing with the new arguments
    set +e
    $PERF_ANALYZER -v -i $PROTOCOL -m graphdef_int32_int32_int32 >$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    if [ $(cat $CLIENT_LOG | grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi

    $PERF_ANALYZER -v -i $PROTOCOL -m graphdef_int32_int32_int32 --concurrency-range 1:5:2 >$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    if [ $(cat $CLIENT_LOG | grep "error | Request count: 0 | : 0 infer/sec\|: 0 usec|Request concurrency: 2" | wc -l) -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi

    $PERF_ANALYZER -v -i $PROTOCOL -m graphdef_int32_int32_int32 --concurrency-range 1:5:2 \
    --input-data=${INT_JSONDATAFILE} >$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    if [ $(cat $CLIENT_LOG | grep "error | Request count: 0 | : 0 infer/sec\|: 0 usec|Request concurrency: 2" | wc -l) -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi

    $PERF_ANALYZER -v -i $PROTOCOL -m graphdef_int32_int32_int32 --request-rate-range 1000:2000:500 \
    -p1000 -b 1 -a>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi

    $PERF_ANALYZER -v -i $PROTOCOL -m graphdef_int32_int32_int32 --request-rate-range 1000:2000:500 \
    --input-data=${INT_JSONDATAFILE} -p1000 -b 1 -a>$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    if [ $(cat $CLIENT_LOG | grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi

    # Binary search for request rate mode
    $PERF_ANALYZER -v -i $PROTOCOL -m graphdef_int32_int32_int32 --request-rate-range 1000:2000:100 -p1000 -b 1 \
    -a --binary-search --request-distribution "poisson" -l 10 >$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    set -e
    
    # Binary search for concurrency range mode and make sure it doesn't hang
    $PERF_ANALYZER -v -a --request-distribution "poisson" --shared-memory none \
    --percentile 99 --binary-search --concurrency-range 1:8:2 -l 5 \
    -m graphdef_int32_int32_int32 -b 1 >$CLIENT_LOG 2>&1 &
    PA_PID=$!
    if [ "$PA_PID" == "0" ]; then
        echo -e "\n***\n*** Failed to start $PERF_ANALYZER\n***"
        cat $CLIENT_LOG
        RET=1
    fi
    # wait for PA to finish running
    sleep 200
    if ps -p $PA_PID > /dev/null; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** $PERF_ANALYZER is hanging after 200 s\n***"
        kill $PA_PID
        RET=1
    fi
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    set -e

    # Testing with combinations of string input and shared memory types
    for SHARED_MEMORY_TYPE in none system cuda; do
        set +e
        $PERF_ANALYZER -v -i $PROTOCOL -m graphdef_object_object_object --string-data=1 -p2000 \
    --shared-memory=$SHARED_MEMORY_TYPE>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        set -e
    done

    # Testing with combinations of file inputs and shared memory types
    for SHARED_MEMORY_TYPE in none system cuda; do
        set +e
        $PERF_ANALYZER -v -i $PROTOCOL -m graphdef_object_object_object --input-data=$TESTDATADIR -p2000 \
    --shared-memory=$SHARED_MEMORY_TYPE>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        set -e
    done

    for SHARED_MEMORY_TYPE in none system cuda; do
        set +e
        $PERF_ANALYZER -v -i $PROTOCOL -m graphdef_object_object_object --input-data=$STRING_JSONDATAFILE \
    --input-data=$STRING_JSONDATAFILE -p2000 --shared-memory=$SHARED_MEMORY_TYPE>$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        set -e
    done

    # Testing with combinations of variable inputs and shared memory types
    for SHARED_MEMORY_TYPE in none system cuda; do
        set +e
        $PERF_ANALYZER -v -i $PROTOCOL -m graphdef_object_int32_int32 --input-data=$TESTDATADIR \
    --shape INPUT0:2,8 --shape INPUT1:2,8 -p2000 --shared-memory=$SHARED_MEMORY_TYPE \
    >$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        if [ $(cat $CLIENT_LOG | grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        set -e
    done

    for SHARED_MEMORY_TYPE in none system cuda; do
        set +e
        $PERF_ANALYZER -v -i $PROTOCOL -m graphdef_object_int32_int32 --input-data=$STRING_WITHSHAPE_JSONDATAFILE \
    --shape INPUT0:2,8 --shape INPUT1:2,8 -p2000 --shared-memory=$SHARED_MEMORY_TYPE \
    >$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        set -e
    done

    set +e
    $PERF_ANALYZER -v -i $PROTOCOL -m graphdef_int32_int32_float32 --shape INPUT0:2,8,2 \
    --shape INPUT1:2,8,2 -p2000 >$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    set -e

    # Trying to batch tensors with different shape
    for SHARED_MEMORY_TYPE in none system cuda; do
        set +e
        $PERF_ANALYZER -v -i $PROTOCOL -m graphdef_int32_int32_float32 --shape INPUT0:2,8,2 --shape INPUT1:2,8,2 -p2000 -b 4 \
    --shared-memory=$SHARED_MEMORY_TYPE --input-data=$INT_DIFFSHAPE_JSONDATAFILE >$CLIENT_LOG 2>&1
        if [ $? -eq 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        if [ $(cat $CLIENT_LOG | grep "can not batch tensors with different shapes together" | wc -l) -eq 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        set -e
    done

    # Shape tensor I/O model (server needs the shape tensor on the CPU)
    for SHARED_MEMORY_TYPE in none system; do
        set +e
        $PERF_ANALYZER -v -i $PROTOCOL -m plan_zero_1_float32 --input-data=$SHAPETENSORADTAFILE \
    --shape DUMMY_INPUT0:4,4 -p2000 --shared-memory=$SHARED_MEMORY_TYPE -b 8 \
    >$CLIENT_LOG 2>&1
        if [ $? -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        if [ $(cat $CLIENT_LOG | grep ": 0 infer/sec\|: 0 usec" | wc -l) -ne 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        set -e
    done

    set +e
    $PERF_ANALYZER -v -i $PROTOCOL -m  simple_savedmodel_sequence_object -p 2000 -t5 --sync \
    --input-data=$SEQ_JSONDATAFILE >$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi

    $PERF_ANALYZER -v -i $PROTOCOL -m  simple_savedmodel_sequence_object -p 2000 -t5 --sync \
    --input-data=$SEQ_JSONDATAFILE  >$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi

    $PERF_ANALYZER -v -i $PROTOCOL -m  simple_savedmodel_sequence_object -p 1000 --request-rate-range 100:200:50 --sync \
    --input-data=$SEQ_JSONDATAFILE >$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    set -e


    # Testing with variable ensemble model. This unit specifies different shape values
    # for different inferences.
    for SHARED_MEMORY_TYPE in none system cuda; do
        set +e
        # FIXME: Enable HTTP when the server is able to correctly return the complex error messages.
        $PERF_ANALYZER -v -i grpc -m graphdef_sequence_float32 --shape INPUT:2 --input-data=$FLOAT_DIFFSHAPE_JSONDATAFILE \
    --input-data=$FLOAT_DIFFSHAPE_JSONDATAFILE -p2000 --shared-memory=$SHARED_MEMORY_TYPE >$CLIENT_LOG 2>&1
        if [ $? -eq 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        if [ $(cat $CLIENT_LOG |  grep "Inputs to operation Select of type Select must have the same size and shape." | wc -l) -eq 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
        set -e
    done

done

# Test with output validation
set +e
$PERF_ANALYZER -v -m graphdef_int32_int32_int32 --input-data=${NON_ALIGNED_OUTPUT_JSONDATAFILE} >$CLIENT_LOG 2>&1
if [ $? -eq 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ $(cat $CLIENT_LOG |  grep "The 'validation_data' field doesn't align with 'data' field in the json file" | wc -l) -eq 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

$PERF_ANALYZER -v -m graphdef_int32_int32_int32 --input-data=${WRONG_OUTPUT_JSONDATAFILE} >$CLIENT_LOG 2>&1
if [ $? -eq 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ $(cat $CLIENT_LOG |  grep "Output size doesn't match expected size" | wc -l) -eq 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

$PERF_ANALYZER -v -m graphdef_int32_int32_int32 --input-data=${WRONG_OUTPUT_2_JSONDATAFILE} >$CLIENT_LOG 2>&1
if [ $? -eq 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ $(cat $CLIENT_LOG |  grep "Output doesn't match expected output" | wc -l) -eq 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi


$PERF_ANALYZER -v -m graphdef_int32_int32_int32 --input-data=${OUTPUT_JSONDATAFILE} >$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

$PERF_ANALYZER -v -m simple_savedmodel_sequence_object -i grpc --streaming \
--input-data=${SEQ_WRONG_OUTPUT_JSONDATAFILE} >$CLIENT_LOG 2>&1
if [ $? -eq 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ $(cat $CLIENT_LOG |  grep "Output doesn't match expected output" | wc -l) -eq 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

$PERF_ANALYZER -v -m simple_savedmodel_sequence_object -i grpc --streaming \
--input-data=${SEQ_OUTPUT_JSONDATAFILE} >$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

## Testing with very large concurrencies and large dataset
INPUT_DATA_OPTION="--input-data $SEQ_JSONDATAFILE "
for i in {1..9}; do
   INPUT_DATA_OPTION=" ${INPUT_DATA_OPTION} ${INPUT_DATA_OPTION}"
done
set +e
$PERF_ANALYZER -v -m  simple_savedmodel_sequence_object -p 10000 --concurrency-range 1500:2000:250 -i grpc --streaming \
${INPUT_DATA_OPTION} >$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
   cat $CLIENT_LOG
   echo -e "\n***\n*** Test Failed\n***"
   RET=1
fi
if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
   cat $CLIENT_LOG
   echo -e "\n***\n*** Test Failed\n***"
   RET=1
fi
set -e

## Test count_windows mode
set +e

# Send incorrect shape and make sure that perf_analyzer doesn't hang
$PERF_ANALYZER -v -m graphdef_object_int32_int32 --measurement-mode "count_windows" \
    --shape INPUT0:1,8,100 --shape INPUT1:2,8 --string-data=1 >$CLIENT_LOG 2>&1
if [ $? -eq 0 ]; then
   cat $CLIENT_LOG
   echo -e "\n***\n*** Test Failed\n***"
   RET=1
fi
if [ $(cat $CLIENT_LOG |  grep "unexpected shape for input 'INPUT0' for model" | wc -l) -eq 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

$PERF_ANALYZER -v -m graphdef_object_int32_int32 --measurement-mode "count_windows" \
    --shape INPUT0:2,8 --shape INPUT1:2,8 --string-data=1 >$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
   cat $CLIENT_LOG
   echo -e "\n***\n*** Test Failed\n***"
   RET=1
fi
if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
   cat $CLIENT_LOG
   echo -e "\n***\n*** Test Failed\n***"
   RET=1
fi
set -e

## Test perf_analyzer with MPI / multiple models
set +e
mpiexec --allow-run-as-root \
  -n 1 --merge-stderr-to-stdout --output-filename . --tag-output --timestamp-output \
    $PERF_ANALYZER -v -m graphdef_float16_float32_float32 \
      --measurement-mode count_windows -s 50 : \
  -n 1 --merge-stderr-to-stdout --output-filename . --tag-output --timestamp-output \
    $PERF_ANALYZER -v -m libtorch_float32_int32_int32 \
      --measurement-mode count_windows -s 50 : \
  -n 1 --merge-stderr-to-stdout --output-filename . --tag-output --timestamp-output \
    $PERF_ANALYZER -v -m savedmodel_nobatch_float32_int32_int32 \
      --measurement-mode count_windows -s 50
if [ $? -ne 0 ]; then
   cat 1/rank.0/stdout 1/rank.2/stdout 1/rank.2/stdout
   echo -e "\n***\n*** Test Failed\n***"
   RET=1
else
  is_stable() {
    local RANK=$1
    local IS_THROUGHPUT=$2
    if [ $IS_THROUGHPUT ]; then
      local GREP_PATTERN="\[1,$RANK\]<stdout>:  Pass \[[0-9]+\] throughput: \K[0-9]+\.?[0-9]*"
    else
      local GREP_PATTERN="\[1,$RANK\]<stdout>:  Pass \[[0-9]+\] throughput: [0-9]+\.?[0-9]* infer/sec. Avg latency: \K[0-9]+"
    fi
    local LAST_MINUS_0=$(grep -oP "$GREP_PATTERN" 1/rank.$RANK/stdout | tail -3 | sed -n 3p)
    local LAST_MINUS_1=$(grep -oP "$GREP_PATTERN" 1/rank.$RANK/stdout | tail -3 | sed -n 2p)
    local LAST_MINUS_2=$(grep -oP "$GREP_PATTERN" 1/rank.$RANK/stdout | tail -3 | sed -n 1p)
    local MEAN=$(awk "BEGIN {print (($LAST_MINUS_0+$LAST_MINUS_1+$LAST_MINUS_2)/3)}")
    local STABILITY_THRESHOLD=0.5
    local WITHIN_THRESHOLD_0=$(awk "BEGIN {print ($LAST_MINUS_0 >= ((1 - $STABILITY_THRESHOLD) * $MEAN) && $LAST_MINUS_0 <= ((1 + $STABILITY_THRESHOLD) * $MEAN))}")
    local WITHIN_THRESHOLD_1=$(awk "BEGIN {print ($LAST_MINUS_1 >= ((1 - $STABILITY_THRESHOLD) * $MEAN) && $LAST_MINUS_1 <= ((1 + $STABILITY_THRESHOLD) * $MEAN))}")
    local WITHIN_THRESHOLD_2=$(awk "BEGIN {print ($LAST_MINUS_2 >= ((1 - $STABILITY_THRESHOLD) * $MEAN) && $LAST_MINUS_2 <= ((1 + $STABILITY_THRESHOLD) * $MEAN))}")
    echo $(($WITHIN_THRESHOLD_0 && $WITHIN_THRESHOLD_1 && $WITHIN_THRESHOLD_2))
  }

  RANK_0_THROUGHPUT_IS_STABLE=$(is_stable 0 1)
  RANK_0_LATENCY_IS_STABLE=$(is_stable 0 0)
  RANK_1_THROUGHPUT_IS_STABLE=$(is_stable 1 1)
  RANK_1_LATENCY_IS_STABLE=$(is_stable 1 0)
  RANK_2_THROUGHPUT_IS_STABLE=$(is_stable 2 1)
  RANK_2_LATENCY_IS_STABLE=$(is_stable 2 0)

  ALL_STABLE=$(( \
    $RANK_0_THROUGHPUT_IS_STABLE && \
    $RANK_0_LATENCY_IS_STABLE && \
    $RANK_1_THROUGHPUT_IS_STABLE && \
    $RANK_1_LATENCY_IS_STABLE && \
    $RANK_2_THROUGHPUT_IS_STABLE && \
    $RANK_2_LATENCY_IS_STABLE))

  if [ $ALL_STABLE -eq 0 ]; then
    cat 1/rank.0/stdout 1/rank.2/stdout 1/rank.2/stdout
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
  fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

# Generate valid CA
openssl genrsa -passout pass:1234 -des3 -out ca.key 4096
openssl req -passin pass:1234 -new -x509 -days 365 -key ca.key -out ca.crt -subj  "/C=SP/ST=Spain/L=Valdepenias/O=Test/OU=Test/CN=Root CA"

# Generate valid Server Key/Cert
openssl genrsa -passout pass:1234 -des3 -out server.key 4096
openssl req -passin pass:1234 -new -key server.key -out server.csr -subj  "/C=SP/ST=Spain/L=Valdepenias/O=Test/OU=Server/CN=localhost"
openssl x509 -req -passin pass:1234 -days 365 -in server.csr -CA ca.crt -CAkey ca.key -set_serial 01 -out server.crt

# Remove passphrase from the Server Key
openssl rsa -passin pass:1234 -in server.key -out server.key

# Generate valid Client Key/Cert
openssl genrsa -passout pass:1234 -des3 -out client.key 4096
openssl req -passin pass:1234 -new -key client.key -out client.csr -subj  "/C=SP/ST=Spain/L=Valdepenias/O=Test/OU=Client/CN=localhost"
openssl x509 -passin pass:1234 -req -days 365 -in client.csr -CA ca.crt -CAkey ca.key -set_serial 01 -out client.crt

# Remove passphrase from Client Key
openssl rsa -passin pass:1234 -in client.key -out client.key

# Create mutated client key (Make first char of each like capital)
cp client.key client2.key && sed -i "s/\b\(.\)/\u\1/g" client2.key
cp client.crt client2.crt && sed -i "s/\b\(.\)/\u\1/g" client2.crt

SERVER_ARGS="--model-repository=${DATADIR} --grpc-use-ssl=1 --grpc-server-cert=server.crt --grpc-server-key=server.key --grpc-root-cert=ca.crt"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# Test gRPC SSL
set +e

# Test that gRPC protocol with SSL works correctly
$PERF_ANALYZER -v -i grpc -m graphdef_int32_int32_int32 \
  --ssl-grpc-use-ssl \
  --ssl-grpc-root-certifications-file=ca.crt \
  --ssl-grpc-private-key-file=client.key \
  --ssl-grpc-certificate-chain-file=client.crt \
  > ${CLIENT_LOG}.grpc_success 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.grpc_success
    RET=1
fi

# Test that gRPC protocol with SSL fails with incorrect key
$PERF_ANALYZER -v -i grpc -m graphdef_int32_int32_int32 \
    --ssl-grpc-use-ssl \
    --ssl-grpc-root-certifications-file=ca.crt \
    --ssl-grpc-private-key-file=client.key \
    --ssl-grpc-certificate-chain-file=client2.crt \
    > ${CLIENT_LOG}.grpc_failure 2>&1
if [ $? -eq 0 ]; then
    cat ${CLIENT_LOG}.grpc_failure
    echo -e "\n***\n*** Expected test failure\n***"
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

cp server.crt /etc/nginx/cert.crt
cp server.key /etc/nginx/cert.key

SERVER_ARGS="--model-repository=${DATADIR}"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

# Setup the new configuration for the proxy. The HTTPS traffic will be
# redirected to the running instance of server at localhost:8000
cp nginx.conf /etc/nginx/sites-available/default

# Start the proxy server
service nginx restart

# Test HTTP SSL
set +e

# Test that HTTP protocol with SSL works correctly with certificates
$PERF_ANALYZER -v -u https://localhost:443 -i http -m graphdef_int32_int32_int32 \
    --ssl-https-verify-peer 1 \
    --ssl-https-verify-host 2 \
    --ssl-https-ca-certificates-file ca.crt \
    --ssl-https-client-certificate-file client.crt \
    --ssl-https-client-certificate-type PEM \
    --ssl-https-private-key-file client.key \
    --ssl-https-private-key-type PEM \
    > ${CLIENT_LOG}.https_success 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.https_success
    RET=1
fi

# Test that HTTP protocol with SSL works correctly without certificates
$PERF_ANALYZER -v -u https://localhost:443 -i http -m graphdef_int32_int32_int32 \
    --ssl-https-verify-peer 0 \
    --ssl-https-verify-host 0
    > ${CLIENT_LOG}.https_success 2>&1
if [ $? -ne 0 ]; then
    cat ${CLIENT_LOG}.https_success
    RET=1
fi

# Test that HTTP protocol with SSL fails with incorrect key
$PERF_ANALYZER -v -u https://localhost:443 -i http -m graphdef_int32_int32_int32 \
    --ssl-https-verify-peer 1 \
    --ssl-https-verify-host 2 \
    --ssl-https-ca-certificates-file ca.crt \
    --ssl-https-client-certificate-file client.crt \
    --ssl-https-client-certificate-type PEM \
    --ssl-https-private-key-file client2.key \
    --ssl-https-private-key-type PEM \
    > ${CLIENT_LOG}.https_failure 2>&1
if [ $? -eq 0 ]; then
    cat ${CLIENT_LOG}.https_failure
    echo -e "\n***\n*** Expected test failure\n***"
    RET=1
fi

set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
