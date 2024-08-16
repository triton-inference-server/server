#!/bin/bash
MODEL=${1:-"tensorrt_llm_bls"}
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "'${MODEL}'",
     "messages": [{"role": "user", "content": "Say this is a test!"}]
   }' | jq
