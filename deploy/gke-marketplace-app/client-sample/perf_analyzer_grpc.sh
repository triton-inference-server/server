#!/usr/bin/env bash
# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

SERVER_HOST=${1:-"${INGRESS_HOST}:${INGRESS_PORT}"} # need update public IP
MODEL_VERSION=${2:-1}
precision=${3:-"int8"}
BATCH_SIZE=${4:-1}
MAX_LATENCY=${5:-500}
MAX_CLIENT_THREADS=${6:-6}
MAX_CONCURRENCY=${7:-20}
MODEL_NAME=${8:-"bert_large"}
SEQ_LENGTH=${9:-"128"}
PERFCLIENT_PERCENTILE=${10:-90}
STABILITY_PERCENTAGE=${11:-0.01}
MAX_TRIALS=${12:-1000000}

ARGS="\
   --max-threads ${MAX_CLIENT_THREADS} \
   -m ${MODEL_NAME} \
   -x ${MODEL_VERSION} \
   -p 1000 \
   -t ${MAX_CONCURRENCY} \
   -s ${STABILITY_PERCENTAGE} \
   -r ${MAX_TRIALS} \
   -v \
   -i gRPC \
   -u ${SERVER_HOST} \
   -b ${BATCH_SIZE} \
   -l ${MAX_LATENCY} \
   -z \
   --shape=input_ids:${SEQ_LENGTH} \
   --shape=segment_ids:${SEQ_LENGTH} \
   --shape=input_mask:${SEQ_LENGTH} \
   --percentile=${PERFCLIENT_PERCENTILE}"

echo "Using args:  $(echo "$ARGS" | sed -e 's/   -/\n-/g')"

/workspace/install/bin/perf_client $ARGS
