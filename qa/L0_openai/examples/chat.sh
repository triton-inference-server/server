#!/bin/bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "tensorrt_llm_bls",
     "messages": [{"role": "user", "content": "Say this is a test!"}]
   }' | jq
