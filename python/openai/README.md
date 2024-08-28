# OpenAI-Compatible Frontend for Triton Inference Server

## Pre-requisites

1. Docker + NVIDIA Container Runtime
2. A correctly configured `HF_TOKEN` for access to HuggingFace models.
    - The current examples and testing primarily use the
      [`meta-llama/Meta-Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
      model, but you can manually bring your own models and adjust accordingly.

## VLLM

1. Build and launch the container:
  - Mounts the `~/.huggingface/cache` for re-use of downloaded models across runs, containers, etc.
  - Sets the [`HF_TOKEN`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hftoken) environment variable to
    access gated models, make sure this is set in your local environment if needed.

```bash
docker build -t tritonserver-openai-vllm -f docker/Dockerfile.vllm .

docker run -it --net=host --gpus all --rm \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN \
  tritonserver-openai-vllm
```

2. Launch the OpenAI-compatible Triton Inference Server:
```bash
# NOTE: Adjust the --tokenizer based on the model being used
python3 openai_frontend/main.py --model-repository tests/vllm_models/ --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct
```

3. Send a `/v1/chat/completions` request:
  - Note the use of `jq` is optional, but provides a nicely formatted output for JSON responses.
```bash
MODEL="llama-3.1-8b-instruct"
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "'${MODEL}'",
  "messages": [{"role": "user", "content": "Say this is a test!"}]
}' | jq
```

4. Send a `/v1/completions` request:
  - Note the use of `jq` is optional, but provides a nicely formatted output for JSON responses.
```bash
MODEL="llama-3.1-8b-instruct"
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "'${MODEL}'",
  "prompt": "Machine learning is"
}' | jq
```

5. Benchmark with `genai-perf`:
```bash
MODEL="llama-3.1-8b-instruct"
TOKENIZER="meta-llama/Meta-Llama-3.1-8B-Instruct"
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

6. Use the OpenAI python client directly:
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
```bash
pytest -v tests/
```

8. For a list of examples, see the `examples/` folder.

## TensorRT-LLM

**NOTE**: The workflow for preparing TRT-LLM engines, model repository, etc. in order to
load and test is not fleshed out in the README here yet. You can try using the Triton CLI
or follow existing TRT-LLM backend examples to prepare a model repository, and point
at the model repository accordingly when following the examples.

0. Prepare your model repository for a TensorRT-LLM model, build the engine, etc.

1. Build and launch the container:
  - Mounts the openai source files to `/workspace` for simplicity, later on these will be shipped in the container.
  - Mounts the `~/.huggingface/cache` for re-use of downloaded models across runs, containers, etc.
  - Sets the [`HF_TOKEN`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hftoken) environment variable to
    access gated models, make sure this is set in your local environment if needed.

```bash
docker build -t tritonserver-openai-tensorrtllm -f docker/Dockerfile.tensorrtllm ./docker

docker run -it --net=host --gpus all --rm \
  -v ${PWD}:/workspace \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN \
  -w /workspace \
  tritonserver-openai-tensorrtllm
```

2. Launch the OpenAI server:
```bash
# NOTE: Adjust the --tokenizer based on the model being used
python3 openai_frontend/main.py --model-repository tests/tensorrtllm_models/ --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct
```

3. Send a `/v1/chat/completions` request:
  - Note the use of `jq` is optional, but provides a nicely formatted output for JSON responses.
```bash
MODEL="tensorrt_llm_bls"
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "'${MODEL}'",
  "messages": [{"role": "user", "content": "Say this is a test!"}]
}' | jq
```

The other examples should be the same as vLLM, except that you should set `MODEL="tensorrt_llm_bls"`,
everywhere applicable as seen in the example request above.
