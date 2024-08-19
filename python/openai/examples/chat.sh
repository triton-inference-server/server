#!/bin/bash
# or "tensorrt_llm_bls" for TRT-LLM
MODEL=${1:-"llama-3.1-8b-instruct"}
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "'${MODEL}'",
     "messages": [{"role": "user", "content": "Say this is a test!"}]
   }' | jq
