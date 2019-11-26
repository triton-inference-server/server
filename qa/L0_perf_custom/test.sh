#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

REPO_VERSION=${NVIDIA_TENSORRT_SERVER_VERSION}
if [ "$#" -ge 1 ]; then
    REPO_VERSION=$1
fi
if [ -z "$REPO_VERSION" ]; then
    echo -e "Repository version must be specified"
    echo -e "\n***\n*** Test Failed\n***"
    exit 1
fi

PERF_CLIENT=../clients/perf_client
REPORTER=../common/reporter.py

PROTOCOLS="grpc http"
APIS="async sync"
CONCURRENCIES="1 8"
TENSOR_SIZES="16384 16777216"

SERVER=/opt/tensorrtserver/bin/trtserver
source ../common/util.sh

# Select the single GPU that will be available to the inference server
export CUDA_VISIBLE_DEVICES=0

rm -f *.json *.log *.tjson *.metrics *.metrics *.csv
RET=0

MODEL_NAME="custom_zero_1_float32"
REPO_DIR=`pwd`/models

# Prepare the model 
rm -fr $REPO_DIR && mkdir $REPO_DIR && \
    cp -r ../custom_models/$MODEL_NAME $REPO_DIR/. && \
    mkdir -p $REPO_DIR/$MODEL_NAME/1 && \
    cp ./libidentity.so $REPO_DIR/$MODEL_NAME/1/libcustom.so

(cd $REPO_DIR/$MODEL_NAME
sed -i "s/dims:.*\[.*\]/dims: \[ -1 \]/g" config.pbtxt && \
                echo "instance_group [ { kind: KIND_CPU }]" >> config.pbtxt)

SERVER_ARGS=--model-repository=$REPO_DIR
SERVER_LOG="inferenceserver.log"

run_server
if (( $SERVER_PID == 0 )); then
  echo -e "\n***\n*** Failed to start $SERVER\n***"
  cat $SERVER_LOG
  exit 1
fi

for PROTOCOL in $PROTOCOLS; do
  for API in $APIS; do
    for CONCURRENCY in $CONCURRENCIES; do
      for TENSOR_SIZE in $TENSOR_SIZES; do
        if [[ $TENSOR_SIZE = "16384" ]]; then
          NAME=${MODEL_NAME}_t${CONCURRENCY}_${API}_${PROTOCOL}_small
        else
          NAME=${MODEL_NAME}_t${CONCURRENCY}_${API}_${PROTOCOL}_large
        fi
        CLIENT_ARGS="-v -m${MODEL_NAME} -t${CONCURRENCY} --shape INPUT0:${TENSOR_SIZE} -f ${NAME}.csv"
        EXTRA_ARGS=""
        if [[ $PROTOCOL = "grpc" ]]; then
          EXTRA_ARGS="-i grpc"
        fi
        if [[ $API = "async" ]]; then
          # Using a single context for maintaining the concurrency in async case, similar to
          # simple_perf client.
          EXTRA_ARGS="${EXTRA_ARGS} -a --max-threads=1"
        fi

        set +e
        $PERF_CLIENT $CLIENT_ARGS ${EXTRA_ARGS}>> ${NAME}.log 2>&1
        if (( $? != 0 )); then
          RET=1
        fi

        curl localhost:8002/metrics -o ${NAME}.metrics >> ${NAME}.log 2>&1
        if (( $? != 0 )); then
          RET=1
        fi

        set -e

        echo -e "[{\"s_benchmark_kind\":\"perf_client\"," >> ${NAME}.tjson
        echo -e "\"s_benchmark_name\":\"${API}\"," >> ${NAME}.tjson
        echo -e "\"s_protocol\":\"${PROTOCOL}\"," >> ${NAME}.tjson
        echo -e "\"s_framework\":\"custom\"," >> ${NAME}.tjson
        echo -e "\"s_model\":\"${MODEL_NAME}\"," >> ${NAME}.tjson
        echo -e "\"l_concurrency\":${CONCURRENCY}," >> ${NAME}.tjson
        echo -e "\"l_dynamic_batch_size\":1," >> ${NAME}.tjson
        echo -e "\"l_batch_size\":1," >> ${NAME}.tjson
        echo -e "\"l_size\":${TENSOR_SIZE}," >> ${NAME}.tjson
        echo -e "\"l_instance_count\":1}]" >> ${NAME}.tjson

        if [ -f $REPORTER ]; then
          set +e
          URL_FLAG=
          if [ ! -z ${BENCHMARK_REPORTER_URL} ]; then
            URL_FLAG="-u ${BENCHMARK_REPORTER_URL}"
          fi

          $REPORTER -v -o ${NAME}.json --csv ${NAME}.csv ${URL_FLAG} ${NAME}.tjson
          if (( $? != 0 )); then
            RET=1
          fi
          set -e
        fi
      done
    done
  done
done

kill $SERVER_PID
wait $SERVER_PID


if (( $RET == 0 )); then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
