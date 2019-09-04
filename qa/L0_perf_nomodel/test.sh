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

# Descriptive name for the current results
UNDERTEST_NAME=${NVIDIA_TENSORRT_SERVER_VERSION}

# Subdirectory containing results to compare against.
BASELINE_DIR=baseline

# Confidence percentile to use when stabilizing and reporting
# results. A value of 0 indicates that average value should be used
# for stabilizing results.
PERF_CLIENT_PERCENTILE=${PERF_CLIENT_PERCENTILE:=95}

# Threshold, as a percentage, to mark any performance change as a
# speedup or a slowdown.
PERF_CLIENT_SPEEDUP_THRESHOLD=5.0
PERF_CLIENT_SLOWDOWN_THRESHOLD=5.0

# Length of window, in milliseconds, to use when stabilizing latency
# and infer/sec results.
PERF_CLIENT_STABILIZE_WINDOW=5000

# Threshold, as a percentage, to use when stabilizing latency and
# infer/sec results. Values must vary by less than this percent over 3
# measurement windows to be considered value.
PERF_CLIENT_STABILIZE_THRESHOLD=5.0

RUNTEST=./runtest.sh
ANALYZE=./perf_analysis.py
ANALYZE_LOG_EXT=analysis

# The model used for data collection has a single input and a single
# output. The model does minimal work (just copy input to
# output). TENSOR_SIZE is the number of elements in the model input
# and the model output. The tensor element type is float so to get the
# number of elements in each tensor need to divide the test I/O size
# by 4.
TENSOR_SIZE_16MB=$((4*1024*1024))

LATENCY_NAMES=(
    "${UNDERTEST_NAME} Minimum Latency GRPC"
    "${UNDERTEST_NAME} Minimum Latency HTTP"
    "${UNDERTEST_NAME} 16MB I/O Latency GRPC"
    "${UNDERTEST_NAME} 16MB I/O Latency HTTP")
LATENCY_DIRS=(
    min_latency_grpc
    min_latency_http
    16mb_latency_grpc
    16mb_latency_http)
LATENCY_PROTOCOLS=(
    grpc
    http
    grpc
    http)
LATENCY_TENSOR_SIZES=(
    1
    1
    ${TENSOR_SIZE_16MB}
    ${TENSOR_SIZE_16MB})
# If TensorRT adds support for variable-size tensors can fix identity
# model to allow TENSOR_SIZE > 1. For libtorch we need to create an
# identity model with variable-size input.
LATENCY_BACKENDS=(
    "plan custom graphdef savedmodel onnx libtorch netdef"
    "plan custom graphdef savedmodel onnx libtorch netdef"
    "custom graphdef savedmodel onnx netdef"
    "custom graphdef savedmodel onnx netdef"
)

# Remove any exiting data collection and analysis
rm -f *.${ANALYZE_LOG_EXT}
for idx in "${!LATENCY_NAMES[@]}"; do
    LATENCY_DIR=${LATENCY_DIRS[$idx]}
    rm -fr ${LATENCY_DIR}
done

#
# Data Collection
#

RET=0
set +e

for idx in "${!LATENCY_NAMES[@]}"; do
    LATENCY_NAME=${LATENCY_NAMES[$idx]}
    LATENCY_DIR=${LATENCY_DIRS[$idx]}
    LATENCY_PROTOCOL=${LATENCY_PROTOCOLS[$idx]}
    LATENCY_TENSOR_SIZE=${LATENCY_TENSOR_SIZES[$idx]}
    LATENCY_BACKEND=${LATENCY_BACKENDS[$idx]}

    RESULTNAME=${LATENCY_NAME} \
              RESULTDIR=${LATENCY_DIR} \
              PERF_CLIENT_PERCENTILE=${PERF_CLIENT_PERCENTILE} \
              PERF_CLIENT_STABILIZE_WINDOW=${PERF_CLIENT_STABILIZE_WINDOW} \
              PERF_CLIENT_STABILIZE_THRESHOLD=${PERF_CLIENT_STABILIZE_THRESHOLD} \
              PERF_CLIENT_PROTOCOL=${LATENCY_PROTOCOL} \
              TENSOR_SIZE=${LATENCY_TENSOR_SIZE} \
              BACKENDS=${LATENCY_BACKEND} \
              STATIC_BATCH_SIZES=1 \
              DYNAMIC_BATCH_SIZES=1 \
              INSTANCE_COUNTS=1 \
              REQUIRED_MAX_CONCURRENCY=1 \
              bash -x ${RUNTEST} $1
    if (( $? != 0 )); then
        RET=1
    fi
done

set -e

if (( $RET == 0 )); then
    echo -e "\n***\n*** Data Collection Passed\n***"
else
    echo -e "\n***\n*** Data Collection FAILED\n***"
    exit $RET
fi

#
# Analyze
#

RET=0
set +e

for BASELINE_NAME in $(ls ${BASELINE_DIR}); do
    ANALYZE_LOG=${BASELINE_NAME}.${ANALYZE_LOG_EXT}

    for idx in "${!LATENCY_NAMES[@]}"; do
        LATENCY_NAME=${LATENCY_NAMES[$idx]}
        LATENCY_DIR=${LATENCY_DIRS[$idx]}

        $ANALYZE --name="${LATENCY_NAME}" \
                 --latency \
                 --slowdown-threshold=${PERF_CLIENT_SLOWDOWN_THRESHOLD} \
                 --speedup-threshold=${PERF_CLIENT_SPEEDUP_THRESHOLD} \
                 --baseline-name=${BASELINE_NAME} \
                 --baseline=${BASELINE_DIR}/${BASELINE_NAME}/${LATENCY_DIR} \
                 --undertest-name=${UNDERTEST_NAME} \
                 --undertest=${LATENCY_DIR} >> ${ANALYZE_LOG} 2>&1
        if (( $? != 0 )); then
            echo -e "** ${LATENCY_NAME} Analysis Failed"
            RET=1
        else
            cat ${ANALYZE_LOG}
        fi
    done
done

set -e

if (( $RET == 0 )); then
    echo -e "\n***\n*** Analysis Passed\n***"
else
    echo -e "\n***\n*** Analysis FAILED\n***"
fi

exit $RET
