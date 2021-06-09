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
MODEL_NAME=${2:-"${MODEL_NAME}"}
SEQ_LENGTH=${3:-"${SEQ_LEN}"}
BATCH_SIZE=${4:-2}
MAX_LATENCY=${5:-5000}
MAX_CLIENT_THREADS=${6:-20}
MAX_CONCURRENCY=${7:-24}
MODEL_VERSION=${8:-1}
precision=${9:-"fp32"}
PERFCLIENT_PERCENTILE=${10:-90}
MAX_TRIALS=${12:-40}

ARGS="\
   --max-threads ${MAX_CLIENT_THREADS} \
   -m ${MODEL_NAME} \
   -x ${MODEL_VERSION} \
   -p 3000 \
   --async \
   --concurrency-range 4:${MAX_CONCURRENCY}:2 \
   -r ${MAX_TRIALS} \
   -v \
   -i HTTP \
   -u ${SERVER_HOST} \
   -b ${BATCH_SIZE} \
   -l ${MAX_LATENCY} \
   -z \
   --percentile=${PERFCLIENT_PERCENTILE}"

echo "Using args:  $(echo "$ARGS" | sed -e 's/   -/\n-/g')"

/workspace/install/bin/perf_client $ARGS -f perf.csv