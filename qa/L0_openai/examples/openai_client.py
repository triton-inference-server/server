#!/usr/bin/env python3
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

completion = client.chat.completions.create(
    model="tensorrt_llm_bls",
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
