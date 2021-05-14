#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#PERF_ANALYZER=../clients/perf_analyzer
PERF_ANALYZER=/workspace/install/bin/perf_client

DATADIR=`pwd`/models
TESTDATADIR=`pwd`/test_data

SERVER_LIBRARY_PATH=/opt/tritonserver
SERVER=$SERVER_LIBRARY_PATH/bin/tritonserver
SERVER_ARGS="--model-repository=${DATADIR}"
SERVER_LOG="./inference_server.log"

INT_JSONDATAFILE=`pwd`/json_input_data_files/int_data.json
INT_DIFFSHAPE_JSONDATAFILE=`pwd`/json_input_data_files/int_data_diff_shape.json
FLOAT_DIFFSHAPE_JSONDATAFILE=`pwd`/json_input_data_files/float_data_with_shape.json
STRING_JSONDATAFILE=`pwd`/json_input_data_files/string_data.json
STRING_WITHSHAPE_JSONDATAFILE=`pwd`/json_input_data_files/string_data_with_shape.json
SEQ_JSONDATAFILE=`pwd`/json_input_data_files/seq_data.json
SHAPETENSORADTAFILE=`pwd`/json_input_data_files/shape_tensor_data.json
IMAGE_JSONDATAFILE=`pwd`/json_input_data_files/image_data.json


ERROR_STRING="error | Request count: 0 | : 0 infer/sec"

source ../common/util.sh

DATASRC=/tmp/host/data
rm -f $SERVER_LOG $CLIENT_LOG
rm -rf $DATADIR $TESTDATADIR $ENSEMBLE_DATADIR

mkdir -p $DATADIR
# Copy fixed-shape models
cp -r ${DATASRC}/qa_model_repository/graphdef_int32_int32_int32 $DATADIR/
cp -r ${DATASRC}/qa_model_repository/graphdef_nobatch_int32_int32_int32 $DATADIR/
cp -r ${DATASRC}/qa_model_repository/graphdef_object_object_object $DATADIR/
cp -r ${DATASRC}/qa_model_repository/graphdef_nobatch_object_object_object $DATADIR/

# Copy a variable-shape models
cp -r ${DATASRC}/qa_variable_model_repository/graphdef_object_int32_int32 $DATADIR/
cp -r ${DATASRC}/qa_variable_model_repository/graphdef_int32_int32_float32 $DATADIR/

# Copy shape tensor models
cp -r ${DATASRC}/qa_shapetensor_model_repository/plan_zero_1_float32 $DATADIR/

# Copying ensemble including a sequential model
cp -r ${DATASRC}/qa_sequence_model_repository/savedmodel_sequence_object $DATADIR
cp -r ${DATASRC}/qa_ensemble_model_repository/qa_sequence_model_repository/simple_savedmodel_sequence_object $DATADIR
cp -r ${DATASRC}/qa_ensemble_model_repository/qa_sequence_model_repository/nop_TYPE_FP32_-1 $DATADIR

# Copying variable sequence model
cp -r ${DATASRC}/qa_variable_sequence_model_repository/graphdef_sequence_float32 $DATADIR

mkdir $DATADIR/nop_TYPE_FP32_-1/1

# Copy inception model to the model repository
cp -r ${DATASRC}/tf_model_store/inception_v1_graphdef $DATADIR

# Copy resnet50v1.5_fp16
cp -r ${DATASRC}/perf_model_store/resnet50v1.5_fp16_savedmodel $DATADIR

# Generating test data
mkdir -p $TESTDATADIR
for INPUT in INPUT0 INPUT1; do
    for i in {1..16}; do
        echo '1' >> $TESTDATADIR/${INPUT}
    done
done
set -x #echo on
RET=0
########## Test C API #############
# C API tests
# Make sure tritonserver is not running first
SERVER_PID=$(pidof tritonserver)
if [ $? -ne 1 ]; then
echo -e "\n There was a previous instance of tritonserver, killing \n"
  kill $SERVER_PID
  wait $SERVER_PID
fi

# #Testing simple configurations

$PERF_ANALYZER -v -m graphdef_int32_int32_int32 \
--service-kind=triton_local --model-repo=$DATADIR --library-name=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
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


$PERF_ANALYZER -v -m graphdef_int32_int32_int32 -t 1 -p2000 -b 1 \
--service-kind=triton_local --model-repo=$DATADIR --library-name=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
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


#Testing that async does NOT work

$PERF_ANALYZER -v -m graphdef_int32_int32_int32 -t 1 -p2000 -b 1 -a \
--service-kind=triton_local --model-repo=$DATADIR --library-name=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
if [ $? -eq 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi


#Testing that shared memory does not work
for SHARED_MEMORY_TYPE in system cuda; do
    set +e
    $PERF_ANALYZER -v -m graphdef_int32_int32_int32 -t 1 -p2000 -b 1 --shared-memory=$SHARED_MEMORY_TYPE \
    --service-kind=triton_local --model-repo=$DATADIR --library-name=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
    if [ $? -ne 1 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    
done

# Testing --request-rate-range does NOT work

$PERF_ANALYZER -v -m graphdef_int32_int32_int32 --request-rate-range 1000:2000:500 -p1000 -b 1 \
--service-kind=triton_local --model-repo=$DATADIR --library-name=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
if [ $? -eq 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi


# Testing with inception model

$PERF_ANALYZER -v -m inception_v1_graphdef -t 1 -p2000 -b 1 \
--service-kind=triton_local --model-repo=$DATADIR --library-name=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
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

# Testing with resnet50 models with large batch sizes

$PERF_ANALYZER -v -m inception_v1_graphdef -t 2 -p2000 -b 64 \
--service-kind=triton_local --model-repo=$DATADIR --library-name=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
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

# Test perf client behavior on different model with different batch size
for MODEL in graphdef_nobatch_int32_int32_int32 graphdef_int32_int32_int32; do
    # Valid batch size
    $PERF_ANALYZER -v  -m $MODEL -t 1 -p2000 -b 1 \
    --service-kind=triton_local --model-repo=$DATADIR --library-name=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
    if [ $? -ne 0 ]; then
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Failed\n***"
        RET=1
    fi
    # Invalid batch sizes
    for STATIC_BATCH in 0 10; do
        $PERF_ANALYZER -v -m $MODEL -t 1 -p2000 -b $STATIC_BATCH \
        --service-kind=triton_local --model-repo=$DATADIR --library-name=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
        if [ $? -eq 0 ]; then
            cat $CLIENT_LOG
            echo -e "\n***\n*** Test Failed\n***"
            RET=1
        fi
    done
done

# Test concurrency
$PERF_ANALYZER -v -m graphdef_int32_int32_int32 --concurrency-range 1:5:2 \
--service-kind=triton_local --model-repo=$DATADIR --library-name=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
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

#Testing with string input
set +e
$PERF_ANALYZER -v -m graphdef_object_object_object --string-data=1 -p2000 \
--service-kind=triton_local --model-repo=$DATADIR --library-name=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
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

# Testing with different file inputs
set +e
$PERF_ANALYZER -v -m graphdef_object_object_object --input-data=$TESTDATADIR -p2000 \
--service-kind=triton_local --model-repo=$DATADIR --library-name=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
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
$PERF_ANALYZER -v -m graphdef_object_object_object --input-data=$STRING_JSONDATAFILE -p2000 \
--service-kind=triton_local --model-repo=$DATADIR --library-name=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
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


# Testing with variable inputs
 
$PERF_ANALYZER -v -m graphdef_object_int32_int32 --input-data=$TESTDATADIR \
--shape INPUT0:2,8 --shape INPUT1:2,8 \
--service-kind=triton_local --model-repo=$DATADIR --library-name=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi


$PERF_ANALYZER -v -m graphdef_object_int32_int32 --input-data=$STRING_WITHSHAPE_JSONDATAFILE \
--shape INPUT0:2,8 --shape INPUT1:2,8 -p2000 \
--service-kind=triton_local --model-repo=$DATADIR --library-name=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
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


$PERF_ANALYZER -v -m graphdef_int32_int32_float32 --shape INPUT0:2,8,2 \
--shape INPUT1:2,8,2 -p2000 \
--service-kind=triton_local --model-repo=$DATADIR --library-name=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
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


# Shape tensor I/O model (server needs the shape tensor on the CPU)

$PERF_ANALYZER -v -m plan_zero_1_float32 --input-data=$SHAPETENSORADTAFILE \
--shape DUMMY_INPUT0:4,4 -p2000 -b 8 \
--service-kind=triton_local --model-repo=$DATADIR --library-name=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
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


# # FIXME: simple_savedmodel_sequence_object segfaults with local CAPI
# 
# $PERF_ANALYZER -v -m  simple_savedmodel_sequence_object -p 2000 -t5 --sync \
# --input-data=$SEQ_JSONDATAFILE \
# --service-kind=triton_local --model-repo=$DATADIR --library-name=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
# if [ $? -ne 0 ]; then
#     cat $CLIENT_LOG
#     echo -e "\n***\n*** Test Failed\n***"
#     RET=1
# fi
# if [ $(cat $CLIENT_LOG |  grep "${ERROR_STRING}" | wc -l) -ne 0 ]; then
#     cat $CLIENT_LOG
#     echo -e "\n***\n*** Test Failed\n***"
#     RET=1
# fi
# 
# # FIXME: Testing with variable ensemble model doesn't work
# $PERF_ANALYZER -v -m graphdef_sequence_float32 --shape INPUT:2 --input-data=$FLOAT_DIFFSHAPE_JSONDATAFILE \
# --input-data=$FLOAT_DIFFSHAPE_JSONDATAFILE -p2000 \
# --service-kind=triton_local --model-repo=$DATADIR --library-name=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
# 
# if [ $? -eq 0 ]; then
#     cat $CLIENT_LOG
#     echo -e "\n***\n*** Test Failed\n***"
#     RET=1
# fi
# if [ $(cat $CLIENT_LOG |  grep "Inputs to operation Select of type Select must have the same size and shape." | wc -l) -eq 0 ]; then
#     cat $CLIENT_LOG
#     echo -e "\n***\n*** Test Failed\n***"
#     RET=1
# fi
# 

# Make sure server is not still running
SERVER_PID=$(pidof tritonserver)
if [ $? -ne 1 ]; then
  echo "\n Tritonserver did not exit properly, killing \n"
  kill $SERVER_PID
  wait $SERVER_PID
  RET=1
fi

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
