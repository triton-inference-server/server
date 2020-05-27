#!/bin/bash
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

rm -f *.log *.serverlog *.csv *.metrics *.tjson *.json

# Descriptive name for the current results
UNDERTEST_NAME=${NVIDIA_TRITON_SERVER_VERSION}

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

RUNTEST=./run_test.sh

# The model used for data collection has a single input and a single
# output. The model does minimal work (just copy input to
# output). TENSOR_SIZE is the number of elements in the model input
# and the model output. The tensor element type is float so to get the
# number of elements in each tensor need to divide the test I/O size
# by 4.
TENSOR_SIZE_16MB=$((4*1024*1024))

TEST_NAMES=(
    "${UNDERTEST_NAME} Minimum Latency GRPC"
    "${UNDERTEST_NAME} Minimum Latency HTTP"
    "${UNDERTEST_NAME} 16MB I/O Latency GRPC"
    "${UNDERTEST_NAME} 16MB I/O Latency HTTP"
    "${UNDERTEST_NAME} Maximum Throughput GRPC"
    "${UNDERTEST_NAME} Maximum Throughput HTTP"
    "${UNDERTEST_NAME} 16MB I/O Throughput GRPC"
    "${UNDERTEST_NAME} 16MB I/O Throughput HTTP"
    "${UNDERTEST_NAME} 16MB I/O Latency GRPC System Shared Memory"
    "${UNDERTEST_NAME} 16MB I/O Latency HTTP System Shared Memory"
    "${UNDERTEST_NAME} 16MB I/O Latency GRPC CUDA Shared Memory"
    "${UNDERTEST_NAME} 16MB I/O Latency HTTP CUDA Shared Memory"
    "${UNDERTEST_NAME} 16MB I/O Throughput GRPC System Shared Memory"
    "${UNDERTEST_NAME} 16MB I/O Throughput HTTP System Shared Memory"
    "${UNDERTEST_NAME} 16MB I/O Throughput GRPC CUDA Shared Memory"
    "${UNDERTEST_NAME} 16MB I/O Throughput HTTP CUDA Shared Memory")
TEST_DIRS=(
    min_latency_grpc
    min_latency_http
    16mb_latency_grpc
    16mb_latency_http
    max_throughput_grpc
    max_throughput_http
    16mb_throughput_grpc
    16mb_throughput_http
    16mb_latency_grpc_shm
    16mb_latency_http_shm
    16mb_latency_grpc_cudashm
    16mb_latency_http_cudashm
    16mb_throughput_grpc_shm
    16mb_throughput_http_shm
    16mb_throughput_grpc_cudashm
    16mb_throughput_http_cudashm)
TEST_PROTOCOLS=(
    grpc
    http
    grpc
    http
    grpc
    http
    grpc
    http
    grpc
    http
    grpc
    http
    grpc
    http
    grpc
    http)
TEST_TENSOR_SIZES=(
    1
    1
    ${TENSOR_SIZE_16MB}
    ${TENSOR_SIZE_16MB}
    1
    1
    ${TENSOR_SIZE_16MB}
    ${TENSOR_SIZE_16MB}
    ${TENSOR_SIZE_16MB}
    ${TENSOR_SIZE_16MB}
    ${TENSOR_SIZE_16MB}
    ${TENSOR_SIZE_16MB}
    ${TENSOR_SIZE_16MB}
    ${TENSOR_SIZE_16MB}
    ${TENSOR_SIZE_16MB})
TEST_INSTANCE_COUNTS=(
    1
    1
    1
    1
    2
    2
    2
    2
    1
    1
    1
    1
    2
    2
    2
    2)
TEST_CONCURRENCY=(
    1
    1
    1
    1
    16
    16
    16
    16
    1
    1
    1
    1
    16
    16
    16
    16)
# If TensorRT adds support for variable-size tensors can fix identity
# model to allow TENSOR_SIZE > 1. For libtorch we need to create an
# identity model with variable-size input.
TEST_BACKENDS=(
    "plan custom graphdef savedmodel onnx libtorch netdef"
    "plan custom graphdef savedmodel onnx libtorch netdef"
    "custom graphdef savedmodel onnx netdef"
    "custom graphdef savedmodel onnx netdef"
    "plan custom graphdef savedmodel onnx libtorch netdef"
    "plan custom graphdef savedmodel onnx libtorch netdef"
    "custom graphdef savedmodel onnx netdef"
    "custom graphdef savedmodel onnx netdef"
    "custom graphdef savedmodel onnx netdef"
    "custom graphdef savedmodel onnx netdef"
    "custom graphdef savedmodel onnx netdef"
    "custom graphdef savedmodel onnx netdef"
    "custom graphdef savedmodel onnx netdef"
    "custom graphdef savedmodel onnx netdef"
    "custom graphdef savedmodel onnx netdef"
    "custom graphdef savedmodel onnx netdef"
    "custom graphdef savedmodel onnx netdef")
TEST_SHARED_MEMORIES=(
    "none"
    "none"
    "none"
    "none"
    "none"
    "none"
    "none"
    "none"
    "system"
    "system"
    "cuda"
    "cuda"
    "system"
    "system"
    "cuda"
    "cuda")

mkdir -p ${REPO_VERSION}

#
# Run Performance tests
#

RET=0
set +e

for idx in "${!TEST_NAMES[@]}"; do
    TEST_NAME=${TEST_NAMES[$idx]}
    TEST_DIR=${TEST_DIRS[$idx]}
    TEST_PROTOCOL=${TEST_PROTOCOLS[$idx]}
    TEST_TENSOR_SIZE=${TEST_TENSOR_SIZES[$idx]}
    TEST_BACKEND=${TEST_BACKENDS[$idx]}
    TEST_SHARED_MEMORY=${TEST_SHARED_MEMORIES[$idx]}
    TEST_INSTANCE_COUNT=${TEST_INSTANCE_COUNTS[$idx]}
    TEST_CONCURRENCY=${TEST_CONCURRENCY[$idx]}

    RESULTNAME=${TEST_NAME} \
                RESULTDIR=${REPO_VERSION}/${TEST_DIR} \
                PERF_CLIENT_PERCENTILE=${PERF_CLIENT_PERCENTILE} \
                PERF_CLIENT_STABILIZE_WINDOW=${PERF_CLIENT_STABILIZE_WINDOW} \
                PERF_CLIENT_STABILIZE_THRESHOLD=${PERF_CLIENT_STABILIZE_THRESHOLD} \
                PERF_CLIENT_PROTOCOL=${TEST_PROTOCOL} \
                TENSOR_SIZE=${TEST_TENSOR_SIZE} \
                BACKENDS=${TEST_BACKEND} \
                SHARED_MEMORY=${TEST_SHARED_MEMORY} \
                STATIC_BATCH_SIZES=1 \
                DYNAMIC_BATCH_SIZES=1 \
                INSTANCE_COUNTS=${TEST_INSTANCE_COUNT} \
                CONCURRENCY=${TEST_CONCURRENCY} \
                bash -x ${RUNTEST} ${REPO_VERSION}
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

exit $RET
