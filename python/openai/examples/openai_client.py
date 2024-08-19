#!/usr/bin/env python3
import sys

from openai import OpenAI

# or "tensorrt_llm_bls" for TRT-LLM
model = "llama-3.1-8b-instruct"
if len(sys.argv) > 1:
    model = sys.argv[1]

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

completion = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {"role": "user", "content": "What are LLMs?"},
    ],
    max_tokens=256,
)

print(completion.choices[0].message.content)
