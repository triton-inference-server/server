#!/usr/bin/env bash

SERVER_HOST=${1:-"${INGRESS_HOST}:${INGRESS_PORT}"} # need update public IP
MODEL_VERSION=${2:-1}
precision=${3:-"int8"}
BATCH_SIZE=${4:-16}
MAX_LATENCY=${5:-500}
MAX_CLIENT_THREADS=${6:-6}
MAX_CONCURRENCY=${7:-6}
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
