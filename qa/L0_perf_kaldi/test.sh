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

# Test with 20.03 for now because model is compatible with that version
REPO_VERSION="20.03"

git clone ssh://git@gitlab-master.nvidia.com:12051/dl/JoC/asr_kaldi.git && \
    cd asr_kaldi && \
    ./scripts/docker/build.sh && \
    ./scripts/docker/launch_download.sh && \
    ./scripts/docker/launch_server.sh

CONCURRENCY=2000

# Client only supports GRPC (5 iterations on the dataset)
./scripts/docker/launch_client.sh -i 5 -c ${CONCURRENCY} >> client_1.log 2>&1

# Capture Throughput
THROUGHPUT=cat client_1.log | grep 'Throughput:' | cut -f 2 | cut -f 1 -d ' '

# '-o' Flag is needed to run online and capture latency
./scripts/docker/launch_client.sh -i 5 -c ${CONCURRENCY} -o >> client_2.log 2>&1

# Capture Latency 95 percentile
LATENCY_95=cat client_2.log | grep -A1 "Latencies:" | sed -n '2 p' | cut -f 5

REPORTER=../common/reporter.py

RET=0

echo -e "[{\"s_benchmark_kind\":\"benchmark_perf\"," >> results.tjson
echo -e "\"s_benchmark_name\":\"kaldi\"," >> results.tjson
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
    echo -e "\n***\n*** Data Collection Passed\n***"
else
    echo -e "\n***\n*** Data Collection FAILED\n***"
fi

# Cleanup folder and docker images
rm -rf asr_kaldi && \
    sudo docker image rm trtis_kaldi_client:latest trtis_kaldi_server:latest

exit $RET
