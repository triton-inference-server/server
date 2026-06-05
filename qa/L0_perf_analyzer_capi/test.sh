#!/bin/bash
# Copyright 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# TESTS COPIED FROM L0_perf_analyzer/test.sh
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

SERVER_LIBRARY_PATH=/opt/tritonserver

FLOAT_DIFFSHAPE_JSONDATAFILE=`pwd`/../common/perf_analyzer_input_data_json/float_data_with_shape.json
STRING_WITHSHAPE_JSONDATAFILE=`pwd`/../common/perf_analyzer_input_data_json/string_data_with_shape.json
SEQ_JSONDATAFILE=`pwd`/../common/perf_analyzer_input_data_json/seq_data.json
SHAPETENSORADTAFILE=`pwd`/../common/perf_analyzer_input_data_json/shape_tensor_data.json

ERROR_STRING="error | Request count: 0 | : 0 infer/sec"

STABILITY_THRESHOLD="9999"

source ../common/util.sh

rm -f $CLIENT_LOG
rm -rf $DATADIR $TESTDATADIR $ENSEMBLE_DATADIR

mkdir -p $DATADIR
# Copy fixed-shape models
cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/graphdef_int32_int32_int32 $DATADIR/
cp -r /data/inferenceserver/${REPO_VERSION}/qa_model_repository/graphdef_object_object_object $DATADIR/

# Copy a variable-shape models
cp -r /data/inferenceserver/${REPO_VERSION}/qa_variable_model_repository/graphdef_object_int32_int32 $DATADIR/
cp -r /data/inferenceserver/${REPO_VERSION}/qa_variable_model_repository/graphdef_int32_int32_float32 $DATADIR/

# Copy shape tensor models
cp -r /data/inferenceserver/${REPO_VERSION}/qa_shapetensor_model_repository/plan_zero_1_float32_int32 $DATADIR/

# Copying ensemble including a sequential model
cp -r /data/inferenceserver/${REPO_VERSION}/qa_sequence_model_repository/savedmodel_sequence_object $DATADIR
cp -r /data/inferenceserver/${REPO_VERSION}/qa_ensemble_model_repository/qa_sequence_model_repository/simple_savedmodel_sequence_object $DATADIR

# Copying variable sequence model
cp -r /data/inferenceserver/${REPO_VERSION}/qa_variable_sequence_model_repository/graphdef_sequence_float32 $DATADIR

# Copying bls model with undefined variable
mkdir -p $DATADIR/bls_undefined/1 && \
    cp ../python_models/bls_undefined/model.py $DATADIR/bls_undefined/1/. && \
    cp ../python_models/bls_undefined/config.pbtxt $DATADIR/bls_undefined/.

# Generating test data
mkdir -p $TESTDATADIR
for INPUT in INPUT0 INPUT1; do
    for i in {1..16}; do
        echo '1' >> $TESTDATADIR/${INPUT}
    done
done

RET=0

########## Test C API #############
# Make sure tritonserver is not running first
set +e
SERVER_PID=$(pidof tritonserver)
if [ $? -ne 1 ]; then
echo -e "\n There was a previous instance of tritonserver, killing \n"
  kill $SERVER_PID
  wait $SERVER_PID
fi
set -e

# Testing simple configuration
$PERF_ANALYZER -v -m graphdef_int32_int32_int32 \
--service-kind=triton_c_api \
--model-repository=$DATADIR --triton-server-directory=$SERVER_LIBRARY_PATH \
-s ${STABILITY_THRESHOLD} >$CLIENT_LOG 2>&1
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
--service-kind=triton_c_api --model-repository=$DATADIR \
--triton-server-directory=$SERVER_LIBRARY_PATH -s ${STABILITY_THRESHOLD} \
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

#Testing with string input
$PERF_ANALYZER -v -m graphdef_object_object_object --string-data=1 -p2000 \
--service-kind=triton_c_api --model-repository=$DATADIR \
--triton-server-directory=$SERVER_LIBRARY_PATH -s ${STABILITY_THRESHOLD} \
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

# Testing with variable inputs
$PERF_ANALYZER -v -m graphdef_object_int32_int32 --input-data=$TESTDATADIR \
--shape INPUT0:2,8 --shape INPUT1:2,8 \
--service-kind=triton_c_api --model-repository=$DATADIR \
--triton-server-directory=$SERVER_LIBRARY_PATH -s ${STABILITY_THRESHOLD} \
>$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi

$PERF_ANALYZER -v -m graphdef_object_int32_int32 \
--input-data=$STRING_WITHSHAPE_JSONDATAFILE \
--shape INPUT0:2,8 --shape INPUT1:2,8 -p2000 \
--service-kind=triton_c_api --model-repository=$DATADIR \
--triton-server-directory=$SERVER_LIBRARY_PATH -s ${STABILITY_THRESHOLD} \
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
--service-kind=triton_c_api --model-repository=$DATADIR \
--triton-server-directory=$SERVER_LIBRARY_PATH -s ${STABILITY_THRESHOLD} \
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

# Shape tensor I/O model (server needs the shape tensor on the CPU)
$PERF_ANALYZER -v -m plan_zero_1_float32_int32 --input-data=$SHAPETENSORADTAFILE \
--shape DUMMY_INPUT0:4,4 -p2000 -b 8 \
--service-kind=triton_c_api --model-repository=$DATADIR \
--triton-server-directory=$SERVER_LIBRARY_PATH -s ${STABILITY_THRESHOLD} \
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

$PERF_ANALYZER -v -m  simple_savedmodel_sequence_object -p 2000 -t5 --sync \
-s ${STABILITY_THRESHOLD} \
--input-data=$SEQ_JSONDATAFILE \
--service-kind=triton_c_api --model-repository=$DATADIR \
--triton-server-directory=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
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

set +e
$PERF_ANALYZER -v -m graphdef_sequence_float32 --shape INPUT:2 \
-s ${STABILITY_THRESHOLD} \
--input-data=$FLOAT_DIFFSHAPE_JSONDATAFILE \
--input-data=$FLOAT_DIFFSHAPE_JSONDATAFILE -p2000 \
--service-kind=triton_c_api --model-repository=$DATADIR \
--triton-server-directory=$SERVER_LIBRARY_PATH --sync >$CLIENT_LOG 2>&1
if [ $? -eq 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
if [ $(cat $CLIENT_LOG |  grep -P "The supplied shape .+ is incompatible with the model's input shape" | wc -l) -eq 0 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

for SHARED_MEMORY_TYPE in system cuda; do
    $PERF_ANALYZER -v -m graphdef_int32_int32_int32 -t 1 -p2000 -b 1 \
    -s ${STABILITY_THRESHOLD} \
    --shared-memory=$SHARED_MEMORY_TYPE \
    --service-kind=triton_c_api --model-repository=$DATADIR \
    --triton-server-directory=$SERVER_LIBRARY_PATH >$CLIENT_LOG 2>&1
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
done


$PERF_ANALYZER -v -m graphdef_int32_int32_int32 --request-rate-range 1000:2000:500 -p1000 -b 1 \
--service-kind=triton_c_api --model-repository=$DATADIR \
--triton-server-directory=$SERVER_LIBRARY_PATH -s ${STABILITY_THRESHOLD} \
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

set +e
# Testing erroneous configuration
# This model is expected to fail
$PERF_ANALYZER -v -m bls_undefined --shape INPUT0:1048576 -t 64\
--service-kind=triton_c_api \
--model-repository=$DATADIR --triton-server-directory=$SERVER_LIBRARY_PATH \
-s ${STABILITY_THRESHOLD} >$CLIENT_LOG 2>&1
if [ $? -ne 99 ]; then
    cat $CLIENT_LOG
    echo -e "\n***\n*** Test Failed\n***"
    RET=1
fi
set -e

# Make sure server is not still running
set +e
SERVER_PID=$(pidof tritonserver)
if [ $? -eq 0 ]; then
  echo -e "\n Tritonserver did not exit properly, killing \n"
  kill $SERVER_PID
  wait $SERVER_PID
  RET=1
fi
set -e

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test FAILED\n***"
fi
exit $RET
