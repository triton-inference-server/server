#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

# Test with 20.05 because kaldi image for 20.06 is not yet available
TRITON_VERSION="20.05"

cd /workspace
git clone --single-branch --depth=1 -b r${TRITON_VERSION} \
    https://github.com/NVIDIA/triton-inference-server.git

echo "add_subdirectory(kaldi-asr-client)" >> triton-inference-server/src/clients/c++/CMakeLists.txt

cp -r asr_kaldi/kaldi-asr-client triton-inference-server/src/clients/c++
cp -r asr_kaldi/model-repo/kaldi_online/config.pbtxt model-repo/kaldi_online/

# Client dependencies
(apt-get update && \
    apt-get install -y --no-install-recommends \
        libssl-dev \
        libb64-dev \
        rapidjson-dev)

pip3 install --upgrade wheel setuptools grpcio-tools

# Build client library and kaldi perf client
(cd triton-inference-server/build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX:PATH=/workspace/install && \
    make -j16 trtis-clients)

RET=0
rm -rf *.log

# Run server
/opt/tritonserver/bin/trtserver --model-repo=/workspace/model-repo > server.log 2>&1 &
SERVER_PID=$!
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start server\n***"
    cat server.log
    exit 1
fi

KALDI_CLIENT=install/bin/kaldi_asr_parallel_client

# Run client
RESULTS_DIR="/data/results"
mkdir -p $RESULTS_DIR

CONCURRENCY=2000

# Client only supports GRPC (5 iterations on the dataset)
$KALDI_CLIENT -i 5 -c ${CONCURRENCY} >> client_1.log 2>&1
if (( $? != 0 )); then
    RET=1
fi

# Capture Throughput
THROUGHPUT=`cat client_1.log | grep 'Throughput:' | cut -f 2 | cut -f 1 -d ' '`

# '-o' Flag is needed to run online and capture latency
$KALDI_CLIENT -i 5 -c ${CONCURRENCY} -o >> client_2.log 2>&1
if (( $? != 0 )); then
    RET=1
fi

# Capture Latency 95 percentile
LATENCY_95=`cat client_2.log | grep -A1 "Latencies:" | sed -n '2 p' | cut -f 5`

REPORTER=triton-inference-server/qa/common/reporter.py

echo -e "[{\"s_benchmark_kind\":\"benchmark_perf\"," >> results.tjson
echo -e "\"s_benchmark_name\":\"kaldi\"," >> results.tjson
echo -e "\"s_server\":\"triton\"," >> results.tjson
echo -e "\"s_protocol\":\"grpc\"," >> results.tjson
echo -e "\"s_model\":\"asr_kaldi\"," >> results.tjson
echo -e "\"l_concurrency\":${CONCURRENCY}," >> results.tjson
echo -e "\"d_infer_per_sec\":${THROUGHPUT}," >> results.tjson
echo -e "\"d_latency_p95_ms\":${LATENCY_95}," >> results.tjson
echo -e "\"l_instance_count\":1}]" >> results.tjson

if [ -f $REPORTER ]; then
    set +e

    URL_FLAG=
    if [ ! -z ${BENCHMARK_REPORTER_URL} ]; then
        URL_FLAG="-u ${BENCHMARK_REPORTER_URL}"
    fi

    $REPORTER -v -o results.json ${URL_FLAG} results.tjson
    if (( $? != 0 )); then
        RET=1
    fi

    set -e
fi

if (( $RET == 0 )); then
    echo -e "\n***\n*** ASR Kaldi Benchmark Passed\n***"
else
    echo -e "\n***\n*** ASR Kaldi Benchmark FAILED\n***"
fi

exit $RET
