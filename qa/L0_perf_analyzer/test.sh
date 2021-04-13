#!/bin/bash
# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

export CUDA_VISIBLE_DEVICES=0

CLIENT_LOG="./perf_analyzer.log"
PERF_ANALYZER=../clients/perf_analyzer

DATADIR=`pwd`/models
ENSEMBLE_DATADIR=`pwd`/ensemble_model_repository
TESTDATADIR=`pwd`/test_data

INT_JSONDATAFILE=`pwd`/json_input_data_files/int_data.json
INT_DIFFSHAPE_JSONDATAFILE=`pwd`/json_input_data_files/int_data_diff_shape.json
FLOAT_DIFFSHAPE_JSONDATAFILE=`pwd`/json_input_data_files/float_data_with_shape.json
STRING_JSONDATAFILE=`pwd`/json_input_data_files/string_data.json
STRING_WITHSHAPE_JSONDATAFILE=`pwd`/json_input_data_files/string_data_with_shape.json
SEQ_JSONDATAFILE=`pwd`/json_input_data_files/seq_data.json
SHAPETENSORADTAFILE=`pwd`/json_input_data_files/shape_tensor_data.json
IMAGE_JSONDATAFILE=`pwd`/json_input_data_files/image_data.json

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=${DATADIR} --model-repository=${ENSEMBLE_DATADIR}"
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
cp libtriton_identity.so $DATADIR/nop_TYPE_FP32_-1/1/

# Copy inception model to the model repository
cp -r /data/inferenceserver/${REPO_VERSION}/tf_model_store/inception_v1_graphdef $DATADIR

# Copy resnet50v1.5_fp16
cp -r /data/inferenceserver/${REPO_VERSION}/perf_model_store/resnet50v1.5_fp16_savedmodel $DATADIR

# Set up the ensemble model repository
cp -r ../ensemble_models/image_preprocess_ensemble_example ensemble_model_repository

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

    set +e
    # Testing with preprocess_inception_ensemble model
    $PERF_ANALYZER -v -i $PROTOCOL -m preprocess_inception_ensemble --input-data=$IMAGE_JSONDATAFILE \
    -p2000 >$CLIENT_LOG 2>&1
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

set +e

# Fix me: Uncomment after fixing DLIS-1054
## Testing with very large concurrencies and large dataset
#INPUT_DATA_OPTION="--input-data $SEQ_JSONDATAFILE "
#for i in {1..9}; do
#    INPUT_DATA_OPTION=" ${INPUT_DATA_OPTION} ${INPUT_DATA_OPTION}"
#done
#set +e
#$PERF_ANALYZER -v -m  simple_savedmodel_sequence_object -p 10000 --concurrency-range 1500:2500:500 -i grpc --streaming \
#${INPUT_DATA_OPTION} >$CLIENT_LOG 2>&1
#if [ $? -ne 0 ]; then
#    cat $CLIENT_LOG
#    echo -e "\n***\n*** Test Failed\n***"
#    RET=1
#fi
#if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
#    cat $CLIENT_LOG
#    echo -e "\n***\n*** Test Failed\n***"
#    RET=1
#fi
#set -e

kill $SERVER_PID
wait $SERVER_PID

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
