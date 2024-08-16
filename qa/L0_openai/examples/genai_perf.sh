#!/bin/bash
genai-perf \
  --model tensorrt_llm_bls \
  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
  --service-kind openai \
  --endpoint-type chat \
  --synthetic-input-tokens-mean 256 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 256 \
  --output-tokens-stddev 0 \
  # --streaming
  # --extra-inputs stream:true
