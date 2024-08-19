# OpenAI-Compatible Frontend for Triton Inference Server

## Pre-requisites

1. Docker + NVIDIA Container Runtime
2. A correctly configured `HF_TOKEN` for access to HuggingFace models.
    - The current examples and testing primarily use the
      [`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
      model, but you can manually bring your own models and adjust accordingly.

## VLLM

1. Build and launch the container:
```bash
docker build -t tritonserver-openai-vllm -f Dockerfile.vllm .
# NOTE: The volume mount is flexible as long as you can access
# all the source files within the container.
docker run -it --net=host --gpus all --rm \
  -v ${PWD}:/workspace \
  -w /workspace \
  tritonserver-openai-vllm
```

2. Launch the OpenAI server:
```bash
# NOTE: Adjust the --tokenizer based on the model being used
python3 main.py --model-repository src/tests/vllm_models/ --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct
```

3. Send a `/chat/completions` request:
```bash
MODEL="llama-3.1-8b-instruct"
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "'${MODEL}'",
  "messages": [{"role": "user", "content": "Say this is a test!"}]
}'
```

4. Send a `/completions` request:
```bash
MODEL="llama-3.1-8b-instruct"
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "'${MODEL}'",
  "prompt": "Machine learning is"
}'
```

5. Benchmark with `genai-perf`:
```bash
MODEL="llama-3.1-8b-instruct"
TOKENIZER="meta-llama/Meta-Llama-3-8B-Instruct"
genai-perf \
  --model ${MODEL} \
  --tokenizer ${TOKENIZER} \
  --service-kind openai \
  --endpoint-type chat \
  --synthetic-input-tokens-mean 256 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 256 \
  --output-tokens-stddev 0 \
  --streaming
```

6. Use an OpenAI client:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

model = "llama-3.1-8b-instruct"
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
```

7. Run tests (NOTE: The server should not be running, the tests will handle starting/stopping the server as necessary):
```
cd src/tests
pytest -v
```

8. For other examples, see the `examples/` folder.

## TensorRT-LLM

0. `[TODO]` Prepare your model repository for a TensorRT-LLM model, build the engine, etc.

1. Build and launch the container:
```
docker build -t tritonserver-openai-tensorrtllm -f Dockerfile.tensorrtllm .
# NOTE: The volume mount is flexible as long as you can access
# all the source files within the container.
docker run -it --net=host --gpus all --rm \
  -v ${PWD}:/workspace \
  -w /workspace \
  tritonserver-openai-tensorrtllm
```

2. Launch the OpenAI server:
```
# NOTE: Adjust the --tokenizer based on the model being used
python3 main.py --model-repository src/tests/tensorrt_llm_models/ --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct
```

3. Send requests:
```
MODEL="tensorrt_llm_bls"
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "'${MODEL}'",
  "messages": [{"role": "user", "content": "Say this is a test!"}]
}'
```

The other examples should be the same as vLLM, except that you should set `MODEL="tensorrt_llm_bls"`,
everywhere applicable as seen in the example request above.
