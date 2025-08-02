<!--
# Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
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
  nvcr.io/nvidia/tritonserver:25.07-vllm-python-py3
```

2. Launch the OpenAI-compatible Triton Inference Server:
```bash
cd /opt/tritonserver/python/openai

# NOTE: Adjust the --tokenizer based on the model being used
python3 openai_frontend/main.py --model-repository tests/vllm_models --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct
```

<details>
<summary>Example output</summary>

```
...
+-----------------------+---------+--------+
| Model                 | Version | Status |
+-----------------------+---------+--------+
| llama-3.1-8b-instruct | 1       | READY  | <- Correct Model Loaded in Triton
+-----------------------+---------+--------+
...
Found model: name='llama-3.1-8b-instruct', backend='vllm'
[WARNING] Adding CORS for the following origins: ['http://localhost']
INFO:     Started server process [126]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:9000 (Press CTRL+C to quit) <- OpenAI Frontend Started Successfully
```

</details>

3. Send a `/v1/chat/completions` request:
  - Note the use of `jq` is optional, but provides a nicely formatted output for JSON responses.
```bash
MODEL="llama-3.1-8b-instruct"
curl -s http://localhost:9000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "'${MODEL}'",
  "messages": [{"role": "user", "content": "Say this is a test!"}]
}' | jq
```

<details>
<summary>Example output</summary>

```json
{
  "id": "cmpl-0242093d-51ae-11f0-b339-e7480668bfbe",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message":
      {
        "content": "This is only a test.",
        "tool_calls": null,
        "role": "assistant",
        "function_call": null
      },
      "logprobs": null
    }
  ],
  "created": 1750846825,
  "model": "llama-3.1-8b-instruct",
  "system_fingerprint": null,
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 7,
    "prompt_tokens": 42,
    "total_tokens": 49
  }
}
```

</details>

4. Send a `/v1/completions` request:
  - Note the use of `jq` is optional, but provides a nicely formatted output for JSON responses.
```bash
MODEL="llama-3.1-8b-instruct"
curl -s http://localhost:9000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "'${MODEL}'",
  "prompt": "Machine learning is"
}' | jq
```

<details>
<summary>Example output</summary>

```json
{
  "id": "cmpl-58fba3a0-51ae-11f0-859d-e7480668bfbe",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "text": " an amazing field that can truly understand the hidden patterns that exist in the data,"
    }
  ],
  "created": 1750846970,
  "model": "llama-3.1-8b-instruct",
  "system_fingerprint": null,
  "object": "text_completion",
  "usage": {
    "completion_tokens": 16,
    "prompt_tokens": 4,
    "total_tokens": 20
  }
}
```

</details>

5. Benchmark with `genai-perf`:
- To install genai-perf in this container, see the instructions [here](https://github.com/triton-inference-server/perf_analyzer/tree/main/genai-perf#install-perf-analyzer-ubuntu-python-38)
- Or try using genai-perf from the [SDK container](https://github.com/triton-inference-server/perf_analyzer/tree/main/genai-perf#install-perf-analyzer-ubuntu-python-38)

```bash
MODEL="llama-3.1-8b-instruct"
TOKENIZER="meta-llama/Meta-Llama-3.1-8B-Instruct"
genai-perf profile \
  --model ${MODEL} \
  --tokenizer ${TOKENIZER} \
  --service-kind openai \
  --endpoint-type chat \
  --url localhost:9000 \
  --streaming
```

<details>
<summary>Example output</summary>

```
2024-10-14 22:43 [INFO] genai_perf.parser:82 - Profiling these models: llama-3.1-8b-instruct
2024-10-14 22:43 [INFO] genai_perf.wrapper:163 - Running Perf Analyzer : 'perf_analyzer -m llama-3.1-8b-instruct --async --input-data artifacts/llama-3.1-8b-instruct-openai-chat-concurrency1/inputs.json -i http --concurrency-range 1 --endpoint v1/chat/completions --service-kind openai -u localhost:9000 --measurement-interval 10000 --stability-percentage 999 --profile-export-file artifacts/llama-3.1-8b-instruct-openai-chat-concurrency1/profile_export.json'
                              NVIDIA GenAI-Perf | LLM Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃                         Statistic ┃    avg ┃    min ┃    max ┃    p99 ┃    p90 ┃    p75 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│          Time to first token (ms) │  71.66 │  64.32 │  86.52 │  76.13 │  74.92 │  73.26 │
│          Inter token latency (ms) │  18.47 │  18.25 │  18.72 │  18.67 │  18.61 │  18.53 │
│              Request latency (ms) │ 348.00 │ 274.60 │ 362.27 │ 355.41 │ 352.29 │ 350.66 │
│            Output sequence length │  15.96 │  12.00 │  16.00 │  16.00 │  16.00 │  16.00 │
│             Input sequence length │ 549.66 │ 548.00 │ 551.00 │ 550.00 │ 550.00 │ 550.00 │
│ Output token throughput (per sec) │  45.84 │    N/A │    N/A │    N/A │    N/A │    N/A │
│      Request throughput (per sec) │   2.87 │    N/A │    N/A │    N/A │    N/A │    N/A │
└───────────────────────────────────┴────────┴────────┴────────┴────────┴────────┴────────┘
2024-10-14 22:44 [INFO] genai_perf.export_data.json_exporter:62 - Generating artifacts/llama-3.1-8b-instruct-openai-chat-concurrency1/profile_export_genai_perf.json
2024-10-14 22:44 [INFO] genai_perf.export_data.csv_exporter:71 - Generating artifacts/llama-3.1-8b-instruct-openai-chat-concurrency1/profile_export_genai_perf.csv
```

</details>

6. Use the OpenAI python client directly:
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
    max_completion_tokens=256,
)

print(completion.choices[0].message.content)
```

7. Run tests (NOTE: The server should not be running, the tests will handle starting/stopping the server as necessary):
```bash
cd /opt/tritonserver/python/openai/
pip install -r requirements-test.txt

pytest -v tests/
```

### LoRA Adapters

If the command line argument `--lora-separator=<separator_string>` is provided
when starting the OpenAI Frontend, a vLLM LoRA adaptor listed on the
`multi_lora.json` may be selected by appending the LoRA name to the model name,
separated by the LoRA separator, on the inference request in
`<model_name><separator_string><lora_name>` format.

<details>
<summary>For example</summary>

```bash
# start server with model named gemma-2b
python3 openai_frontend/main.py --lora-separator=_lora_ ...

# inference without LoRA
curl -s http://localhost:9000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "gemma-2b",
  "temperature": 0,
  "prompt": "When was the wheel invented?"
}'
{
  ...
  "choices":[{..."text":"\n\nThe wheel was invented by the Sumerians in Mesopotamia around 350"}],
  ...
}

# inference with LoRA named doll
curl -s http://localhost:9000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "gemma-2b_lora_doll",
  "temperature": 0,
  "prompt": "When was the wheel invented?"
}'
{
  ...
  "choices":[{..."text":"\n\nThe wheel was invented in Mesopotamia around 3500 BC.\n\n"}],
  ...
}

# inference with LoRA named sheep
curl -s http://localhost:9000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "gemma-2b_lora_sheep",
  "temperature": 0,
  "prompt": "When was the wheel invented?"
}'
{
  ...
  "choices":[{..."text":"\n\nThe wheel was invented around 3000 BC in Mesopotamia.\n\n"}],
  ...
}
```

</details>

When listing or retrieving model(s), the model id will include the LoRA name in
the same `<model_name><separator_string><lora_name>` format for each LoRA
adapter listed on the `multi_lora.json`. Note: The LoRA name inclusion is
limited to locally stored models, inference requests are not limited though.

See the
[vLLM documentation](https://github.com/triton-inference-server/vllm_backend/blob/main/docs/llama_multi_lora_tutorial.md)
on how to serve a model with LoRA adapters.

## TensorRT-LLM

0. Prepare your model repository for a TensorRT-LLM model, build the engine, etc. You can try any of the following options:
  - [Triton CLI](https://github.com/triton-inference-server/triton_cli/)
  - [TRT-LLM Backend Quickstart](https://github.com/triton-inference-server/tensorrtllm_backend?tab=readme-ov-file#quick-start)

1. Launch the container:
  - Mounts the `~/.huggingface/cache` for re-use of downloaded models across runs, containers, etc.
  - Sets the [`HF_TOKEN`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hftoken) environment variable to
    access gated models, make sure this is set in your local environment if needed.

```bash
docker run -it --net=host --gpus all --rm \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN \
  -e TRTLLM_ORCHESTRATOR=1 \
  nvcr.io/nvidia/tritonserver:24.11-trtllm-python-py3
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
python3 openai_frontend/main.py --model-repository path/to/models --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct
```

3. Send a `/v1/chat/completions` request:
  - Note the use of `jq` is optional, but provides a nicely formatted output for JSON responses.
```bash
# MODEL should be the client-facing model name in your model repository for a pipeline like TRT-LLM.
# For example, this could also be "ensemble", or something like "gpt2" if generated from Triton CLI
MODEL="tensorrt_llm_bls"
curl -s http://localhost:9000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "'${MODEL}'",
  "messages": [{"role": "user", "content": "Say this is a test!"}]
}' | jq
```

<details>
<summary>Example output</summary>

```json
{
  "id": "cmpl-704c758c-8a84-11ef-b106-107c6149ca79",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "It looks like you're testing the system!",
        "tool_calls": null,
        "role": "assistant",
        "function_call": null
      },
      "logprobs": null
    }
  ],
  "created": 1728948689,
  "model": "llama-3-8b-instruct",
  "system_fingerprint": null,
  "object": "chat.completion",
  "usage": null
}
```

</details>

The other examples should be the same as vLLM, except that you should set `MODEL="tensorrt_llm_bls"` or `MODEL="ensemble"`,
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

## Model Parallelism Support

- [x] vLLM ([EngineArgs](https://github.com/triton-inference-server/vllm_backend/blob/main/README.md#using-the-vllm-backend))
    - ex: Configure `tensor_parallel_size: 2` in the
      [model.json](https://github.com/triton-inference-server/vllm_backend/blob/main/samples/model_repository/vllm_model/1/model.json)
- [x] TensorRT-LLM ([Orchestrator Mode](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/README.md#orchestrator-mode))
    - Set the following environment variable: `export TRTLLM_ORCHESTRATOR=1`
- [ ] TensorRT-LLM ([Leader Mode](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/README.md#leader-mode))
    - Not currently supported

## Tool Calling

The OpenAI frontend supports `tools` and `tool_choice` in the `v1/chat/completions` API. Please refer to the OpenAI API reference for more details about these parameters:
  [tools](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools),
  [tool_choice](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice)

To enable the tool-calling feature, add the `--tool-call-parser {parser_name}` flag when starting the server. The two available parsers are `llama3` and `mistral`.
The `llama3` parser supports tool-calling features for LLaMA 3.1, 3.2, and 3.3 models, while the `mistral` parser supports tool-calling features for the Mistral Instruct model.

Example for launching the OpenAI frontend with a tool call parser:
```
python3 openai_frontend/main.py \
  --model-repository tests/vllm_models \
  --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct \
  --tool-call-parser llama3
```

Example for making a tool calling request:

```python
import json
from openai import OpenAI


def get_current_weather(city: str, state: str, unit: "str"):
    return (
        "The weather in Dallas, Texas is 85 degrees fahrenheit. It is "
        "partly cloudly, with highs in the 90's."
    )

available_tools = {"get_current_weather": get_current_weather}

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:9000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model = "llama-3.1-8b-instruct" # change this to the model in the repository

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. 'San Francisco'",
                    },
                    "state": {
                        "type": "string",
                        "description": "the two-letter abbreviation for the state that the city is"
                        " in, e.g. 'CA' which would mean 'California'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    }
]

messages = [
    {
        "role": "system",
        "content": "You're a helpful assistant! Answer the users question best you can.",
    },
    {"role": "user", "content": "What is the weather in Dallas, Texas in Fahrenheit?"},
]

tool_calls = client.chat.completions.create(
    messages=messages, model=model, tools=tools, max_completion_tokens=128
)
function_name = tool_calls.choices[0].message.tool_calls[0].function.name
function_arguments = tool_calls.choices[0].message.tool_calls[0].function.arguments

print(f"function name: " f"{function_name}")
print(f"function arguments: {function_arguments}")
print(f"tool calling result: {available_tools[function_name](**json.loads(function_arguments))}")
```

Example output:
```
function name: get_current_weather
function arguments: {"city": "Dallas", "state": "TX", "unit": "fahrenheit"}
tool calling result: The weather in Dallas, Texas is 85 degrees fahrenheit. It is partly cloudly, with highs in the 90's.
```

#### Named Tool Calling

The OpenAI frontend supports named function calling, utilizing guided decoding in the vLLM and TensorRT-LLM backends. Users can specify one of the tools in `tool_choice` to force the model to select a specific tool for function calling.

> [!NOTE]
> The latest release of TensorRT-LLM (v0.18.0) does not yet support guided decoding. To enable this feature, use a build from the main branch of TensorRT-LLM.
> For instructions on enabling guided decoding in the TensorRT-LLM backend, please refer to [this guide](https://github.com/triton-inference-server/tensorrtllm_backend/blob/main/docs/guided_decoding.md)

Example for making a named tool calling request:

```python
import json
from openai import OpenAI


def get_current_weather(city: str, state: str, unit: "str"):
    return (
        "The weather in Dallas, Texas is 85 degrees fahrenheit. It is "
        "partly cloudly, with highs in the 90's."
    )

def get_n_day_weather_forecast(city: str, state: str, unit: str, num_days: int):
    return (
        f"The weather in Dallas, Texas is 85 degrees fahrenheit in next {num_days} days."
    )

available_tools = {"get_current_weather": get_current_weather,
                  "get_n_day_weather_forecast": get_n_day_weather_forecast}

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:9000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
model = "llama-3.1-8b-instruct" # change this to the model in the repository
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. 'San Francisco'",
                    },
                    "state": {
                        "type": "string",
                        "description": "the two-letter abbreviation for the state that the city is"
                        " in, e.g. 'CA' which would mean 'California'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, "
                        "e.g. 'San Francisco'",
                    },
                    "state": {
                        "type": "string",
                        "description": "must the two-letter abbreviation for the state "
                        "that the city is in, e.g. 'CA' which would "
                        "mean 'California'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    },
                },
                "required": ["city", "state", "unit", "num_days"],
            },
        },
     }
]

tool_choice = {"function": {"name": "get_n_day_weather_forecast"}, "type": "function"}

messages = [
    {
        "role": "system",
        "content": "You're a helpful assistant! Answer the users question best you can.",
    },
    {"role": "user", "content": "What is the weather in Dallas, Texas in Fahrenheit?"},
]

tool_calls = client.chat.completions.create(
    messages=messages, model=model, tools=tools, tool_choice=tool_choice, max_completion_tokens=128
)
function_name = tool_calls.choices[0].message.tool_calls[0].function.name
function_arguments = tool_calls.choices[0].message.tool_calls[0].function.arguments

print(f"function name: {function_name}")
print(f"function arguments: {function_arguments}")
print(f"tool calling result: {available_tools[function_name](**json.loads(function_arguments))}")
```

Example output:
```
function name: get_n_day_weather_forecast
function arguments: {"city": "Dallas", "state": "TX", "unit": "fahrenheit", num_days: 1}
tool calling result: The weather in Dallas, Texas is 85 degrees fahrenheit in next 1 days.
```