#!/bin/bash
MODEL=${MODEL:-"llama-3.1-8b-instruct"}
TOKENIZER=${TOKENIZER:-"meta-llama/Meta-Llama-3-8B-Instruct"}
CONCURRENCY=${CONCURRENCY:-128}
REQUESTS_PER_THREAD=${REQUESTS_PER_THREAD:-16}
genai-perf profile \
  --model ${MODEL} \
  --tokenizer ${TOKENIZER} \
  --service-kind openai \
  --endpoint-type chat \
  --synthetic-input-tokens-mean 256 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 256 \
  --output-tokens-stddev 0 \
  --concurrency ${CONCURRENCY} \
  --streaming \
  -- --request-count $((CONCURRENCY*REQUESTS_PER_THREAD))
