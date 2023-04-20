#!/bin/bash
# Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

STATIC_BATCH_SIZES=${STATIC_BATCH_SIZES:=1}
DYNAMIC_BATCH_SIZES=${DYNAMIC_BATCH_SIZES:=1}
INSTANCE_COUNTS=${INSTANCE_COUNTS:=1}
TF_VERSION=${TF_VERSION:=2}

PERF_CLIENT=../clients/perf_client
#REPORTER=../common/reporter.py
#REPORTER=./reporter.py
ls
REPORTER=./tmp/server/qa/L0_perf_deeprecommender/reporter.py
echo ${REPORTER}

SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=`pwd`/models --backend-config=tensorflow,version=${TF_VERSION}"
source ../common/util.sh

# Select the single GPU that will be available to the inference
# server. Or use "export CUDA_VISIBLE_DEVICE=" to run on CPU.
export CUDA_VISIBLE_DEVICES=0

RET=0

for STATIC_BATCH in $STATIC_BATCH_SIZES; do
    for DYNAMIC_BATCH in $DYNAMIC_BATCH_SIZES; do
        for INSTANCE_CNT in $INSTANCE_COUNTS; do
            if (( ($DYNAMIC_BATCH > 1) && ($STATIC_BATCH >= $DYNAMIC_BATCH) )); then
                continue
            fi

            MAX_BATCH=${STATIC_BATCH} && \
                (( $DYNAMIC_BATCH > $STATIC_BATCH )) && \
                MAX_BATCH=${DYNAMIC_BATCH}

            if (( $DYNAMIC_BATCH > 1 )); then
                NAME=${MODEL_NAME}_sbatch${STATIC_BATCH}_dbatch${DYNAMIC_BATCH}_instance${INSTANCE_CNT}_${PERF_CLIENT_PROTOCOL}
            else
                NAME=${MODEL_NAME}_sbatch${STATIC_BATCH}_instance${INSTANCE_CNT}_${PERF_CLIENT_PROTOCOL}
            fi

            rm -fr models && mkdir -p models && \
                cp -r $MODEL_PATH models/. && \
                (cd models/$MODEL_NAME && \
                        sed -i "s/^max_batch_size:.*/max_batch_size: ${MAX_BATCH}/" config.pbtxt && \
                        echo "instance_group [ { count: ${INSTANCE_CNT} }]" >> config.pbtxt)
            if (( $DYNAMIC_BATCH > 1 )); then
                (cd models/$MODEL_NAME && \
                        echo "dynamic_batching { preferred_batch_size: [ ${DYNAMIC_BATCH} ] }" >> config.pbtxt)
            fi

            SERVER_LOG="${NAME}.serverlog"
            run_server
            if (( $SERVER_PID == 0 )); then
                echo -e "\n***\n*** Failed to start $SERVER\n***"
                cat $SERVER_LOG
                exit 1
            fi

            set +e

            # Run test until it passes, or hits max tries. 
            # Expose environment variable to optionally set max tries from CI
            MAX_TRIES=${TRITON_PERF_MAX_TRIES:-"5"}
            for i in $(seq 1 ${MAX_TRIES}); do 
                echo -e "\n***\n*** Perf Analyzer attempt ${i}/${MAX_TRIES}.\n***"
                $PERF_CLIENT -v -i ${PERF_CLIENT_PROTOCOL} -m $MODEL_NAME -p5000 \
                             -b${STATIC_BATCH} --concurrency-range ${CONCURRENCY} \
                             -f ${NAME}.csv || RV=$?
                if [ ${RV} -ne 0 ]; then
                    # If all runs fail, RET will remain 1 at the end and fail test
                    if [ "${i}" -eq "${MAX_TRIES}" ]; then
                      RET=1
                    fi
                # If any runs succeed, move on
                else
                    break
                fi
            done

            curl localhost:8002/metrics -o ${NAME}.metrics >> ${NAME}.log 1>&2
            if (( $? != 0 )); then
                echo -e "\n***\n*** Failed to get metrics\n***"
                RET=1
            fi

            set -e

            echo -e "[{\"s_benchmark_kind\":\"benchmark_perf\"," >> ${NAME}.tjson
            echo -e "\"s_benchmark_name\":\"deeprecommender\"," >> ${NAME}.tjson
            echo -e "\"s_server\":\"triton\"," >> ${NAME}.tjson
            echo -e "\"s_protocol\":\"${PERF_CLIENT_PROTOCOL}\"," >> ${NAME}.tjson
            echo -e "\"s_framework\":\"${MODEL_FRAMEWORK}\"," >> ${NAME}.tjson
            echo -e "\"s_model\":\"${MODEL_NAME}\"," >> ${NAME}.tjson
            echo -e "\"l_concurrency\":${CONCURRENCY}," >> ${NAME}.tjson
            echo -e "\"l_dynamic_batch_size\":${DYNAMIC_BATCH}," >> ${NAME}.tjson
            echo -e "\"l_batch_size\":${STATIC_BATCH}," >> ${NAME}.tjson
            echo -e "\"l_instance_count\":${INSTANCE_CNT}}]" >> ${NAME}.tjson

            kill $SERVER_PID
            wait $SERVER_PID

            if [ -f $REPORTER ]; then
                set +e

                URL_FLAG=
                if [ ! -z ${BENCHMARK_REPORTER_URL} ]; then
                    URL_FLAG="-u ${BENCHMARK_REPORTER_URL}"
                fi

                $REPORTER -v -o ${NAME}.json --csv ${NAME}.csv ${URL_FLAG} ${NAME}.tjson
                if (( $? != 0 )); then
                    echo -e "\n***\n*** Failed to report results\n***"
                    RET=1
                fi

                set -e
            fi
        done
    done
done

if (( $RET == 0 )); then
    echo -e "\n***\n*** Test Passed\n***"
else
    echo -e "\n***\n*** Test FAILED\n***"
fi

exit $RET
