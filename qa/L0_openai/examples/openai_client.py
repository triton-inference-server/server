#!/usr/bin/env python3
import sys

from openai import OpenAI

model = "tensorrt_llm_bls"
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
