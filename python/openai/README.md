<!--
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->
# OpenAI-Compatible Frontend for Triton Inference Server (Beta)

> [!NOTE]
> The OpenAI-Compatible API is currently in BETA. Its features and functionality
> are subject to change as we collect feedback. We're excited to hear any thoughts
> you have and what features you'd like to see!

## Pre-requisites

1. Docker + NVIDIA Container Runtime
2. A correctly configured `HF_TOKEN` for access to HuggingFace models.
    - The current examples and testing primarily use the
      [`meta-llama/Meta-Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
      model, but you can manually bring your own models and adjust accordingly.

## VLLM

1. Launch the container and install dependencies:
  - Mounts the `~/.huggingface/cache` for re-use of downloaded models across runs, containers, etc.
  - Sets the [`HF_TOKEN`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hftoken) environment variable to
    access gated models, make sure this is set in your local environment if needed.

```bash
docker run -it --net=host --gpus all --rm \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN \
  nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3
```

2. Install dependencies inside the container:
```bash
# Install python bindings for tritonserver and tritonfrontend
pip install /opt/tritonserver/python/triton*.whl

# Install application requirements
git clone https://github.com/triton-inference-server/server.git
cd server/python/openai/
pip install -r requirements.txt
```

3. Launch the OpenAI-compatible Triton Inference Server:
```bash
# NOTE: Adjust the --tokenizer based on the model being used
python3 openai_frontend/main.py --model-repository tests/vllm_models --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct
```

4. Send a `/v1/chat/completions` request:
  - Note the use of `jq` is optional, but provides a nicely formatted output for JSON responses.
```bash
MODEL="llama-3.1-8b-instruct"
curl -s http://localhost:9000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "'${MODEL}'",
  "messages": [{"role": "user", "content": "Say this is a test!"}]
}' | jq
```

5. Send a `/v1/completions` request:
  - Note the use of `jq` is optional, but provides a nicely formatted output for JSON responses.
```bash
MODEL="llama-3.1-8b-instruct"
curl -s http://localhost:9000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "'${MODEL}'",
  "prompt": "Machine learning is"
}' | jq
```

6. Benchmark with `genai-perf`:
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

7. Use the OpenAI python client directly:
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9000/v1",
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

8. Run tests (NOTE: The server should not be running, the tests will handle starting/stopping the server as necessary):
```bash
cd server/python/openai/
pip install -r requirements-test.txt

pytest -v tests/
```

## TensorRT-LLM

0. Prepare your model repository for serving a TensorRT-LLM model:
   https://github.com/triton-inference-server/tensorrtllm_backend?tab=readme-ov-file#quick-start

1. Launch the container:
  - Mounts the `~/.huggingface/cache` for re-use of downloaded models across runs, containers, etc.
  - Sets the [`HF_TOKEN`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hftoken) environment variable to
    access gated models, make sure this is set in your local environment if needed.

```bash
docker run -it --net=host --gpus all --rm \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN \
  nvcr.io/nvidia/tritonserver:24.08-trtllm-python-py3
```

2. Install dependencies inside the container:
```bash
# Install python bindings for tritonserver and tritonfrontend
pip install /opt/tritonserver/python/triton*.whl

# Install application requirements
git clone https://github.com/triton-inference-server/server.git
cd server/python/openai/
pip install -r requirements.txt
```

2. Launch the OpenAI server:
```bash
# NOTE: Adjust the --tokenizer based on the model being used
python3 openai_frontend/main.py --model-repository tests/tensorrtllm_models --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct
```

3. Send a `/v1/chat/completions` request:
  - Note the use of `jq` is optional, but provides a nicely formatted output for JSON responses.
```bash
MODEL="tensorrt_llm_bls"
curl -s http://localhost:9000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "'${MODEL}'",
  "messages": [{"role": "user", "content": "Say this is a test!"}]
}' | jq
```

The other examples should be the same as vLLM, except that you should set `MODEL="tensorrt_llm_bls"`,
everywhere applicable as seen in the example request above.

## KServe Frontends

To support serving requests through both the OpenAI-Compatible and
KServe Predict v2 frontends to the same running Triton Inference Server,
the `tritonfrontend` python bindings are included for optional use in this
application as well.

You can opt-in to including these additional frontends, assuming `tritonfrontend`
is installed, with `--enable-kserve-frontends` like below:

```
python3 openai_frontend/main.py \
  --model-repository tests/vllm_models \
  --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct \
  --enable-kserve-frontends
```

See `python3 openai_frontend/main.py --help` for more information on the
available arguments and default values.

For more information on the `tritonfrontend` python bindings, see the docs
[here](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/tritonfrontend.md).
