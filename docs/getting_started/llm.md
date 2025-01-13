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

# Deploying Phi-3 Model with Triton and TRT-LLM

This guide captures the steps to build Phi-3 with TRT-LLM and deploy with Triton Inference Server. It also shows a shows how to use GenAI-Perf to run benchmarks to measure model performance in terms of throughput and latency.

This guide is tested on A100 80GB SXM4 and H100 80GB PCIe. It is confirmed to work with Phi-3-mini-128k-instruct and Phi-3-mini-4k-instruct (see [Support Matrix](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/phi) for full list) using TRT-LLM v0.11 and Triton Inference Server 24.07.

- [Build and test TRT-LLM engine](#build-and-test-trt-llm-engine)
- [Deploy with Triton Inference Server](#deploy-with-triton-inference-server)
- [Benchmark with GenAI-Perf](#benchmark-with-genai-perf)
- [Reference Configurations](#reference-configurations)


## Build and test TRT-LLM engine

Reference: <https://nvidia.github.io/TensorRT-LLM/installation/linux.html>

1. ## Retrieve and launch the Docker container (optional)

<!---->

    # Pre-install the environment using the NVIDIA Container Toolkit to avoid manual environment configuration
    docker run --rm --ipc=host --runtime=nvidia --gpus '"device=0"' --entrypoint /bin/bash -it nvidia/cuda:12.4.1-devel-ubuntu22.04

2. ## Install TensorRT-LLM

<!---->

    # Install dependencies, TensorRT-LLM requires Python 3.10
    apt-get update && apt-get -y install python3.10 python3-pip openmpi-bin libopenmpi-dev git git-lfs

    # Install TensorRT-LLM (v0.11.0)
    pip3 install tensorrt_llm==0.11.0 --extra-index-url https://pypi.nvidia.com

    # Check installation
    python3 -c "import tensorrt_llm"

3. ## Clone the TRT-LLM repo with the Phi-3 conversion script

<!---->

    git clone -b v0.11.0 https://github.com/NVIDIA/TensorRT-LLM.git
    cd TensorRT-LLM/examples/phi/

    # only need to install requirements.txt if you want to test the summarize.py example
    # if so, modify requirements.txt such that tensorrt_llm==0.11.0
    # pip install -r requirements.txt


## Build the TRT-LLM Engine

Reference: <https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/phi>

4. ## Download Phi-3-mini-4k-instruct

<!---->

    git lfs install
    git clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

5. ## Convert weights from HF Transformers to TensorRT-LLM format

<!---->

    python3 ./convert_checkpoint.py \
                        --model_dir ./Phi-3-mini-4k-instruct \
                        --output_dir ./phi-checkpoint \
                        --dtype float16

6. ## Build TensorRT engine(s)

<!---->

    # Build a float16 engine using a single GPU and HF weights.
    # Enable several TensorRT-LLM plugins to increase runtime performance. It also helps with build time.
    # --tp_size and --pp_size are the model shard size
    trtllm-build \
        --checkpoint_dir ./phi-checkpoint \
        --output_dir ./phi-engine \
        --gemm_plugin float16 \
        --max_batch_size 8 \
        --max_input_len 1024 \
        --max_seq_len 2048 \
        --tp_size 1 \
        --pp_size 1

7. ## Run the model

<!---->

    python3 ../run.py --engine_dir ./phi-engine \
         --max_output_len 500 \
         --tokenizer_dir ./Phi-3-mini-4k-instruct \
         --input_text "How do I count to nine in French?"

8. ## Summarization test using the Phi model

The TensorRT-LLM Phi model can be tested to summarize the articles from the [cnn\_dailymail](https://huggingface.co/datasets/cnn_dailymail) dataset. For each summary, the script can compute the [ROUGE](https://en.wikipedia.org/wiki/ROUGE_\(metric\)) scores and use the ROUGE-1 score to validate the implementation. The script can also perform the same summarization using the HF Phi model.

    # Run the summarization task using a TensorRT-LLM model and a single GPU.
    python3 ../summarize.py --engine_dir ./phi-engine \
                            --hf_model_dir ./Phi-3-mini-4k-instruct \
                            --batch_size 1 \
                            --test_trt_llm \
                            --test_hf \
                            --data_type fp16 \
                            --check_accuracy \
                            --tensorrt_llm_rouge1_threshold=20


## Deploy with Triton Inference Server

9. ## Copy engine files from the Docker container to the host

<!---->

    # In another terminal instance, before exiting the current container
    docker cp <container_id>:<path_in_container> <path_on_host>

    # For example
    docker cp 452ee1c1d8a1:/TensorRT-LLM/examples/phi/phi-engine /home/user/phi-engine

10. ## Copy the compiled model to the skeleton repository with TRT-LLM backend

<!---->

    # After exiting the TensorRT-LLM Docker container
    git clone https://github.com/triton-inference-server/tensorrtllm_backend.git
    cd tensorrtllm_backend
    cp ../phi-engine/*   all_models/inflight_batcher_llm/tensorrt_llm/1/

11. ## Modify the configuration files from the model repository

The following configuration files need to be updated:

- ensemble/config.pbtxt

- postprocessing/config.pbtxt

- preprocessing/config.pbtxt

- tensorrt\_llm/config.pbxt

- tensorrt\_llm/1/config.json


### Update ensemble/config.pbtxt

    python3 tools/fill_template.py --in_place \
        all_models/inflight_batcher_llm/ensemble/config.pbtxt \
    triton_max_batch_size:128


### Update preprocessing/config.pbtxt

    python3 tools/fill_template.py --in_place \
        all_models/inflight_batcher_llm/postprocessing/config.pbtxt \
    tokenizer_type:auto,\
    tokenizer_dir:../Phi-3-mini-4k-instruct,\
    triton_max_batch_size:128,\
    postprocessing_instance_count:2


### Update postprocessing/config.pbtxt

    python3 tools/fill_template.py --in_place \
        all_models/inflight_batcher_llm/preprocessing/config.pbtxt \
    tokenizer_type:auto,\
    tokenizer_dir:../Phi-3-mini-4k-instruct,\
    triton_max_batch_size:128,\
    preprocessing_instance_count:2


### Update tensorrt\_llm/config.pbxt

    python3 tools/fill_template.py --in_place \
        all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt \
    decoupled_mode:true,\
    engine_dir:/all_models/inflight_batcher_llm/tensorrt_llm/1,\
    max_tokens_in_paged_kv_cache:,\
    batch_scheduler_policy:guaranteed_completion,\
    kv_cache_free_gpu_mem_fraction:0.2,\
    max_num_sequences:4,\
    triton_backend:tensorrtllm,\
    triton_max_batch_size:128,\
    max_queue_delay_microseconds:10,\
    max_beam_width:1,\
    batching_strategy:inflight_fused_batching,\
    engine_dir:/opt/all_models/inflight_batcher_llm/tensorrt_llm/1,\
    max_tokens_in_paged_kv_cache:1,\
    batch_scheduler_policy:guaranteed_completion,\
    kv_cache_free_gpu_mem_fraction:0.2


    # manually access tensort_llm/config.pbtxt and change the CPU instances to > 1
    # unfortunately this was hard-coded and cannot be update with the above script

    # instance_group [
    #   {
    #     count: 2
    #     kind : KIND_CPU
    #   }
    # ]


#### Max Tokens in Paged KV Cache

This is only required for Phi-3-mini-128k-instruct, and it is not necessary to modify this parameter for Phi-3-mini-4k-instruct.

To accommodate for the 128k context, remove the following from tensorrt\_llm/config.pbxt - which will allow the max tokens to be determined by the KV cache manager. If you don’t want to remove it, you can also set maxTokensInPagedKvCache such that it is large enough (e.g. 4096) to process at least 1 sequence to completion (i.e. must be larger than beam\_width \* tokensPerBlock \* maxBlocksPerSeq)

    parameters: {
      key: "max_tokens_in_paged_kv_cache"
      value: {
        string_value: "4096"
      }
    }


### Update tensorrt\_llm/1/config.json

In the engine config (tensorrtllm\_backend/all\_models/inflight\_batcher\_llm/tensorrt\_llm/1/config.json), add the following under plugin\_config

    "Use_context_fmha_for_generation": false

    # for example:
            "plugin_config": {
                "dtype": "float16",
                "bert_attention_plugin": "auto",
                "streamingllm": false,
                "Use_context_fmha_for_generation": false

The above needs to be done manually with your favorite editor. Once finished, please be sure your working directory is \~/tensorrtllm\_backend

12. ## Delete tensorrt\_llm\_bls

<!---->

    # Recommended to remove the BLS directory if not needed
    rm -rf all_models/inflight_batcher_llm/tensorrt_llm_bls/

13. ## Download model repository

<!---->

    # for tokenizer
    git lfs install
    git clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

14. ## Launch Triton Inference Server (trtllm-python3-py3)

<!---->

    docker run -it --rm --gpus all --network host --shm-size=1g \
    -v $(pwd)/all_models:/opt/all_models \
    -v $(pwd)/scripts:/opt/scripts \
    -v $(pwd)/Phi-3-mini-4k-instruct:/opt/Phi-3-mini-4k-instruct \
    nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3

    # Launch Server
    python3 ../scripts/launch_triton_server.py --model_repo ../all_models/inflight_batcher_llm --world_size 1

15. ## Send Requests

<!---->

    curl -X POST localhost:8000/v2/models/ensemble/generate -d \
    '{
    "text_input": "A farmer with a wolf, a goat, and a cabbage must cross a river by boat. The boat can carry only the farmer and a single item. If left unattended together, the wolf would eat the goat, or the goat would eat the cabbage. How can they cross the river without anything being eaten?",
    "parameters": {
    "max_tokens": 256,
    "bad_words":[""],
    "stop_words":[""]
    }
    }' | jq


## Benchmark with GenAI-Perf

16. ## Launch Triton Inference Server (py3-sdk)

<!---->

    export RELEASE="24.07"
    docker run -it --net=host --gpus '"device=0"'  nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

17. ## Download the Phi-3 tokenizer

Login to Hugging Face (with User Access Tokens) to get the Phi-3 tokenizer. This step is not necessary but helps with interpreting token metrics from prompts and responses. If you skip this step, be sure to remove the --tokenizer flag from the GenAI-Perf script in Step 18.

    git lfs install
    git clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

    pip install huggingface_hub
    huggingface-cli login --token hf_***

18. ## Run GenAI-Perf

<!---->

    export INPUT_SEQUENCE_LENGTH=128
    export OUTPUT_SEQUENCE_LENGTH=128
    export CONCURRENCY=25

    genai-perf \
      -m ensemble \
      --service-kind triton \
      --backend tensorrtllm \
      --random-seed 123 \
      --synthetic-input-tokens-mean $INPUT_SEQUENCE_LENGTH \
      --synthetic-input-tokens-stddev 0 \
      --streaming \
      --output-tokens-mean $OUTPUT_SEQUENCE_LENGTH \
      --output-tokens-stddev 0 \
      --output-tokens-mean-deterministic \
      --concurrency $CONCURRENCY \
      --tokenizer microsoft/Phi-3-mini-4k-instruct \
      --measurement-interval 4000 \
      --url localhost:8001

More details on performance benchmarking with GenAI-Perf can be found [here](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md).

## Reference Configurations

All config files inside /tensorrtllm\_backend/all\_models/inflight\_batcher\_llm are shown below.

<details>
<summary><b> ensemble/config.pbtxt</b></summary>

    # Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    name: "ensemble"
    platform: "ensemble"
    max_batch_size: 128
    input [
      {
        name: "text_input"
        data_type: TYPE_STRING
        dims: [ 1 ]
      },
      {
        name: "decoder_text_input"
        data_type: TYPE_STRING
        dims: [ 1 ]
        optional: true
      },
      {
        name: "image_input"
        data_type: TYPE_FP16
        dims: [ 3, 224, 224 ]
        optional: true
      },
      {
        name: "max_tokens"
        data_type: TYPE_INT32
        dims: [ 1 ]
      },
      {
       name: "bad_words"
       data_type: TYPE_STRING
       dims: [ -1 ]
       optional: true
      },
      {
       name: "stop_words"
       data_type: TYPE_STRING
       dims: [ -1 ]
       optional: true
      },
      {
        name: "end_id"
        data_type: TYPE_INT32
        dims: [ 1 ]
        optional: true
      },
      {
        name: "pad_id"
        data_type: TYPE_INT32
        dims: [ 1 ]
        optional: true
      },
      {
        name: "top_k"
        data_type: TYPE_INT32
        dims: [ 1 ]
        optional: true
      },
      {
        name: "top_p"
        data_type: TYPE_FP32
        dims: [ 1 ]
        optional: true
      },
      {
        name: "temperature"
        data_type: TYPE_FP32
        dims: [ 1 ]
        optional: true
      },
      {
        name: "length_penalty"
        data_type: TYPE_FP32
        dims: [ 1 ]
        optional: true
      },
      {
        name: "repetition_penalty"
        data_type: TYPE_FP32
        dims: [ 1 ]
        optional: true
      },
      {
        name: "min_length"
        data_type: TYPE_INT32
        dims: [ 1 ]
        optional: true
      },
      {
        name: "presence_penalty"
        data_type: TYPE_FP32
        dims: [ 1 ]
        optional: true
      },
      {
        name: "frequency_penalty"
        data_type: TYPE_FP32
        dims: [ 1 ]
        optional: true
      },
      {
        name: "random_seed"
        data_type: TYPE_UINT64
        dims: [ 1 ]
        optional: true
      },
      {
        name: "return_log_probs"
        data_type: TYPE_BOOL
        dims: [ 1 ]
        optional: true
      },
      {
        name: "return_context_logits"
        data_type: TYPE_BOOL
        dims: [ 1 ]
        optional: true
      },
      {
        name: "return_generation_logits"
        data_type: TYPE_BOOL
        dims: [ 1 ]
        optional: true
      },
      {
        name: "beam_width"
        data_type: TYPE_INT32
        dims: [ 1 ]
        optional: true
      },
      {
        name: "stream"
        data_type: TYPE_BOOL
        dims: [ 1 ]
        optional: true
      },
      {
        name: "prompt_embedding_table"
        data_type: TYPE_FP16
        dims: [ -1, -1 ]
        optional: true
      },
      {
        name: "prompt_vocab_size"
        data_type: TYPE_INT32
        dims: [ 1 ]
        optional: true
      },
      {
        name: "embedding_bias_words"
        data_type: TYPE_STRING
        dims: [ -1 ]
        optional: true
      },
      {
        name: "embedding_bias_weights"
        data_type: TYPE_FP32
        dims: [ -1 ]
        optional: true
      }
    ]
    output [
      {
        name: "text_output"
        data_type: TYPE_STRING
        dims: [ -1 ]
      },
      {
        name: "cum_log_probs"
        data_type: TYPE_FP32
        dims: [ -1 ]
      },
      {
        name: "output_log_probs"
        data_type: TYPE_FP32
        dims: [ -1, -1 ]
      },
      {
        name: "context_logits"
        data_type: TYPE_FP32
        dims: [ -1, -1 ]
      },
      {
        name: "generation_logits"
        data_type: TYPE_FP32
        dims: [ -1, -1, -1 ]
      },
      {
        name: "batch_index"
        data_type: TYPE_INT32
        dims: [ 1 ]
      }
    ]
    ensemble_scheduling {
      step [
        {
          model_name: "preprocessing"
          model_version: -1
          input_map {
            key: "QUERY"
            value: "text_input"
          }
          input_map {
            key: "DECODER_QUERY"
            value: "decoder_text_input"
          }
          input_map {
            key: "IMAGE"
            value: "image_input"
          }
          input_map {
            key: "REQUEST_OUTPUT_LEN"
            value: "max_tokens"
          }
          input_map {
            key: "BAD_WORDS_DICT"
            value: "bad_words"
          }
          input_map {
            key: "STOP_WORDS_DICT"
            value: "stop_words"
          }
          input_map {
            key: "EMBEDDING_BIAS_WORDS"
            value: "embedding_bias_words"
          }
          input_map {
            key: "EMBEDDING_BIAS_WEIGHTS"
            value: "embedding_bias_weights"
          }
          input_map {
            key: "END_ID"
            value: "end_id"
          }
          input_map {
            key: "PAD_ID"
            value: "pad_id"
          }
          input_map {
            key: "PROMPT_EMBEDDING_TABLE"
            value: "prompt_embedding_table"
          }
          output_map {
            key: "REQUEST_INPUT_LEN"
            value: "_REQUEST_INPUT_LEN"
          }
          output_map {
            key: "INPUT_ID"
            value: "_INPUT_ID"
          }
          output_map {
            key: "REQUEST_DECODER_INPUT_LEN"
            value: "_REQUEST_DECODER_INPUT_LEN"
          }
          output_map {
            key: "DECODER_INPUT_ID"
            value: "_DECODER_INPUT_ID"
          }
          output_map {
            key: "REQUEST_OUTPUT_LEN"
            value: "_REQUEST_OUTPUT_LEN"
          }
          output_map {
            key: "STOP_WORDS_IDS"
            value: "_STOP_WORDS_IDS"
          }
          output_map {
            key: "BAD_WORDS_IDS"
            value: "_BAD_WORDS_IDS"
          }
          output_map {
            key: "EMBEDDING_BIAS"
            value: "_EMBEDDING_BIAS"
          }
          output_map {
            key: "OUT_END_ID"
            value: "_PREPROCESSOR_END_ID"
          }
          output_map {
            key: "OUT_PAD_ID"
            value: "_PREPROCESSOR_PAD_ID"
          }
          output_map {
            key: "OUT_PROMPT_EMBEDDING_TABLE"
            value: "out_prompt_embedding_table"
          }
        },
        {
          model_name: "tensorrt_llm"
          model_version: -1
          input_map {
            key: "input_ids"
            value: "_INPUT_ID"
          }
          input_map {
            key: "decoder_input_ids"
            value: "_DECODER_INPUT_ID"
          }
          input_map {
            key: "input_lengths"
            value: "_REQUEST_INPUT_LEN"
          }
          input_map {
            key: "decoder_input_lengths"
            value: "_REQUEST_DECODER_INPUT_LEN"
          }
          input_map {
            key: "request_output_len"
            value: "_REQUEST_OUTPUT_LEN"
          }
          input_map {
              key: "end_id"
              value: "_PREPROCESSOR_END_ID"
          }
          input_map {
              key: "pad_id"
              value: "_PREPROCESSOR_PAD_ID"
          }
          input_map {
              key: "embedding_bias"
              value: "_EMBEDDING_BIAS"
          }
          input_map {
              key: "runtime_top_k"
              value: "top_k"
          }
          input_map {
              key: "runtime_top_p"
              value: "top_p"
          }
          input_map {
              key: "temperature"
              value: "temperature"
          }
          input_map {
              key: "len_penalty"
              value: "length_penalty"
          }
          input_map {
              key: "repetition_penalty"
              value: "repetition_penalty"
          }
          input_map {
              key: "min_length"
              value: "min_length"
          }
          input_map {
              key: "presence_penalty"
              value: "presence_penalty"
          }
          input_map {
              key: "frequency_penalty"
              value: "frequency_penalty"
          }
          input_map {
              key: "random_seed"
              value: "random_seed"
          }
          input_map {
              key: "return_log_probs"
              value: "return_log_probs"
          }
          input_map {
              key: "return_context_logits"
              value: "return_context_logits"
          }
          input_map {
              key: "return_generation_logits"
              value: "return_generation_logits"
          }
          input_map {
              key: "beam_width"
              value: "beam_width"
          }
          input_map {
              key: "streaming"
              value: "stream"
          }
          input_map {
            key: "prompt_embedding_table"
            value: "out_prompt_embedding_table"
          }
          input_map {
            key: "prompt_vocab_size"
            value: "prompt_vocab_size"
          }
          input_map {
            key: "stop_words_list"
            value: "_STOP_WORDS_IDS"
          }
          input_map {
            key: "bad_words_list"
            value: "_BAD_WORDS_IDS"
          }
          output_map {
            key: "output_ids"
            value: "_TOKENS_BATCH"
          }
          output_map {
            key: "sequence_length"
            value: "_SEQUENCE_LENGTH"
          },
          output_map {
            key: "cum_log_probs"
            value: "_CUM_LOG_PROBS"
          }
          output_map {
            key: "output_log_probs"
            value: "_OUTPUT_LOG_PROBS"
          },
          output_map {
            key: "context_logits"
            value: "_CONTEXT_LOGITS"
          },
          output_map {
            key: "generation_logits"
            value: "_GENERATION_LOGITS"
          },
          output_map {
            key: "batch_index"
            value: "_BATCH_INDEX"
          }
        },
        {
          model_name: "postprocessing"
          model_version: -1
          input_map {
            key: "TOKENS_BATCH"
            value: "_TOKENS_BATCH"
          }
          input_map {
            key: "CUM_LOG_PROBS"
            value: "_CUM_LOG_PROBS"
          }
          input_map {
            key: "OUTPUT_LOG_PROBS"
            value: "_OUTPUT_LOG_PROBS"
          }
          input_map {
            key: "CONTEXT_LOGITS"
            value: "_CONTEXT_LOGITS"
          }
          input_map {
            key: "GENERATION_LOGITS"
            value: "_GENERATION_LOGITS"
          }
          input_map {
            key: "SEQUENCE_LENGTH"
            value: "_SEQUENCE_LENGTH"
          }
          input_map {
            key: "BATCH_INDEX"
            value: "_BATCH_INDEX"
          }
          output_map {
            key: "OUTPUT"
            value: "text_output"
          }
          output_map {
            key: "OUT_OUTPUT_LOG_PROBS"
            value: "output_log_probs"
          }
          output_map {
            key: "OUT_CUM_LOG_PROBS"
            value: "cum_log_probs"
          }
          output_map {
            key: "OUT_CONTEXT_LOGITS"
            value: "context_logits"
          }
          output_map {
            key: "OUT_GENERATION_LOGITS"
            value: "generation_logits"
          }
          output_map {
            key: "OUT_BATCH_INDEX"
            value: "batch_index"
          }
        }
      ]
    }
</details>

<details>
<summary><b>postprocessing/config.pbtxt</b></summary>

    # Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    name: "postprocessing"
    backend: "python"
    max_batch_size: 128
    input [
      {
        name: "TOKENS_BATCH"
        data_type: TYPE_INT32
        dims: [ -1, -1 ]
      },
      {
        name: "SEQUENCE_LENGTH"
        data_type: TYPE_INT32
        dims: [ -1 ]
      },
      {
        name: "CUM_LOG_PROBS"
        data_type: TYPE_FP32
        dims: [ -1 ]
        optional: true
      },
      {
        name: "OUTPUT_LOG_PROBS"
        data_type: TYPE_FP32
        dims: [ -1, -1 ]
        optional: true
      },
      {
        name: "CONTEXT_LOGITS"
        data_type: TYPE_FP32
        dims: [ -1, -1 ]
        optional: true
      },
      {
        name: "GENERATION_LOGITS"
        data_type: TYPE_FP32
        dims: [ -1, -1, -1 ]
        optional: true
      },
      {
        name: "BATCH_INDEX"
        data_type: TYPE_INT32
        dims: [ 1 ]
        optional: true
      }
    ]
    output [
      {
        name: "OUTPUT"
        data_type: TYPE_STRING
        dims: [ -1 ]
      },
      {
        name: "OUT_CUM_LOG_PROBS"
        data_type: TYPE_FP32
        dims: [ -1 ]
      },
      {
        name: "OUT_OUTPUT_LOG_PROBS"
        data_type: TYPE_FP32
        dims: [ -1, -1 ]
      },
      {
        name: "OUT_CONTEXT_LOGITS"
        data_type: TYPE_FP32
        dims: [ -1, -1 ]
      },
      {
        name: "OUT_GENERATION_LOGITS"
        data_type: TYPE_FP32
        dims: [ -1, -1, -1 ]
      },
      {
        name: "OUT_BATCH_INDEX"
        data_type: TYPE_INT32
        dims: [ 1 ]
      }
    ]

    parameters {
      key: "tokenizer_dir"
      value: {
        string_value: "../Phi-3-mini-4k-instruct"
      }
    }

    parameters {
      key: "skip_special_tokens"
      value: {
        string_value: "${skip_special_tokens}"
      }
    }

    instance_group [
        {
            count: 4
            kind: KIND_CPU
        }
    ]
</details>

<details>
<summary><b> preprocessing/config.pbtxt</b> </summary>

    # Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    name: "preprocessing"
    backend: "python"
    max_batch_size: 128
    input [
        {
            name: "QUERY"
            data_type: TYPE_STRING
            dims: [ 1 ]
        },
        {
            name: "DECODER_QUERY"
            data_type: TYPE_STRING
            dims: [ 1 ]
            optional: true
        },
        {
            name: "IMAGE"
            data_type: TYPE_FP16
            dims: [ 3, 224, 224 ]
            optional: true
        },
        {
            name: "REQUEST_OUTPUT_LEN"
            data_type: TYPE_INT32
            dims: [ 1 ]
        },
        {
            name: "BAD_WORDS_DICT"
            data_type: TYPE_STRING
            dims: [ -1 ]
            optional: true
        },
        {
            name: "STOP_WORDS_DICT"
            data_type: TYPE_STRING
            dims: [ -1 ]
            optional: true
        },
        {
            name: "EMBEDDING_BIAS_WORDS"
            data_type: TYPE_STRING
            dims: [ -1 ]
            optional: true
        },
        {
            name: "EMBEDDING_BIAS_WEIGHTS"
            data_type: TYPE_FP32
            dims: [ -1 ]
            optional: true
        },
        {
            name: "END_ID"
            data_type: TYPE_INT32
            dims: [ 1 ]
            optional: true
        },
        {
            name: "PAD_ID"
            data_type: TYPE_INT32
            dims: [ 1 ]
            optional: true
        },
        {
            name: "PROMPT_EMBEDDING_TABLE"
            data_type: TYPE_FP16
            dims: [ -1, -1 ]
            optional: true
            allow_ragged_batch: true
        }
    ]
    output [
        {
            name: "INPUT_ID"
            data_type: TYPE_INT32
            dims: [ -1 ]
        },
        {
            name: "REQUEST_INPUT_LEN"
            data_type: TYPE_INT32
            dims: [ 1 ]
        },
        {
            name: "DECODER_INPUT_ID"
            data_type: TYPE_INT32
            dims: [ -1 ]
        },
        {
            name: "REQUEST_DECODER_INPUT_LEN"
            data_type: TYPE_INT32
            dims: [ 1 ]
        },
        {
            name: "BAD_WORDS_IDS"
            data_type: TYPE_INT32
            dims: [ 2, -1 ]
        },
        {
            name: "STOP_WORDS_IDS"
            data_type: TYPE_INT32
            dims: [ 2, -1 ]
        },
        {
            name: "EMBEDDING_BIAS"
            data_type: TYPE_FP32
            dims: [ -1 ]
        },
        {
            name: "REQUEST_OUTPUT_LEN"
            data_type: TYPE_INT32
            dims: [ -1 ]
        },
        {
            name: "OUT_END_ID"
            data_type: TYPE_INT32
            dims: [ 1 ]
        },
        {
            name: "OUT_PAD_ID"
            data_type: TYPE_INT32
            dims: [ 1 ]
        },
        {
            name: "OUT_PROMPT_EMBEDDING_TABLE"
            data_type: TYPE_FP16
            dims: [ -1, -1 ]
        }
    ]

    parameters {
      key: "tokenizer_dir"
      value: {
        string_value: "../Phi-3-mini-4k-instruct"
      }
    }

    parameters {
      key: "add_special_tokens"
      value: {
        string_value: "${add_special_tokens}"
      }
    }

    parameters {
      key: "visual_model_path"
      value: {
        string_value: "${visual_model_path}"
      }
    }

    parameters: {
      key: "gpt_model_path"
      value: {
        string_value: "${engine_dir}"
      }
    }

    instance_group [
        {
            count: 4
            kind: KIND_CPU
        }
    ]

</details>

<details>
<summary> <b> tensorrt_llm/config.pbtxt </b></summary>


    # Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    name: "tensorrt_llm"
    backend: "tensorrtllm"
    max_batch_size: 128

    model_transaction_policy {
      decoupled: true
    }

    dynamic_batching {
        preferred_batch_size: [ 128 ]
        max_queue_delay_microseconds: 10
    }

    input [
      {
        name: "input_ids"
        data_type: TYPE_INT32
        dims: [ -1 ]
        allow_ragged_batch: true
      },
      {
        name: "input_lengths"
        data_type: TYPE_INT32
        dims: [ 1 ]
        reshape: { shape: [ ] }
      },
      {
        name: "request_output_len"
        data_type: TYPE_INT32
        dims: [ 1 ]
        reshape: { shape: [ ] }
      },
      {
        name: "draft_input_ids"
        data_type: TYPE_INT32
        dims: [ -1 ]
        optional: true
        allow_ragged_batch: true
      },
      {
        name: "decoder_input_ids"
        data_type: TYPE_INT32
        dims: [ -1 ]
        optional: true
        allow_ragged_batch: true
      },
      {
        name: "decoder_input_lengths"
        data_type: TYPE_INT32
        dims: [ 1 ]
        optional: true
        reshape: { shape: [ ] }
      },
      {
        name: "draft_logits"
        data_type: TYPE_FP32
        dims: [ -1, -1 ]
        optional: true
        allow_ragged_batch: true
      },
      {
        name: "draft_acceptance_threshold"
        data_type: TYPE_FP32
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "end_id"
        data_type: TYPE_INT32
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "pad_id"
        data_type: TYPE_INT32
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "stop_words_list"
        data_type: TYPE_INT32
        dims: [ 2, -1 ]
        optional: true
        allow_ragged_batch: true
      },
      {
        name: "bad_words_list"
        data_type: TYPE_INT32
        dims: [ 2, -1 ]
        optional: true
        allow_ragged_batch: true
      },
      {
        name: "embedding_bias"
        data_type: TYPE_FP32
        dims: [ -1 ]
        optional: true
        allow_ragged_batch: true
      },
      {
        name: "beam_width"
        data_type: TYPE_INT32
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "temperature"
        data_type: TYPE_FP32
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "runtime_top_k"
        data_type: TYPE_INT32
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "runtime_top_p"
        data_type: TYPE_FP32
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "runtime_top_p_min"
        data_type: TYPE_FP32
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "runtime_top_p_decay"
        data_type: TYPE_FP32
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "runtime_top_p_reset_ids"
        data_type: TYPE_INT32
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "len_penalty"
        data_type: TYPE_FP32
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "early_stopping"
        data_type: TYPE_BOOL
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "repetition_penalty"
        data_type: TYPE_FP32
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "min_length"
        data_type: TYPE_INT32
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "beam_search_diversity_rate"
        data_type: TYPE_FP32
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "presence_penalty"
        data_type: TYPE_FP32
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "frequency_penalty"
        data_type: TYPE_FP32
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "random_seed"
        data_type: TYPE_UINT64
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "return_log_probs"
        data_type: TYPE_BOOL
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "return_context_logits"
        data_type: TYPE_BOOL
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "return_generation_logits"
        data_type: TYPE_BOOL
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "stop"
        data_type: TYPE_BOOL
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "streaming"
        data_type: TYPE_BOOL
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      {
        name: "prompt_embedding_table"
        data_type: TYPE_FP16
        dims: [ -1, -1 ]
        optional: true
        allow_ragged_batch: true
      },
      {
        name: "prompt_vocab_size"
        data_type: TYPE_INT32
        dims: [ 1 ]
        reshape: { shape: [ ] }
        optional: true
      },
      # the unique task ID for the given LoRA.
      # To perform inference with a specific LoRA for the first time `lora_task_id` `lora_weights` and `lora_config` must all be given.
      # The LoRA will be cached, so that subsequent requests for the same task only require `lora_task_id`.
      # If the cache is full the oldest LoRA will be evicted to make space for new ones.  An error is returned if `lora_task_id` is not cached.
      {
        name: "lora_task_id"
    	data_type: TYPE_UINT64
    	dims: [ 1 ]
        reshape: { shape: [ ] }
    	optional: true
      },
      # weights for a lora adapter shape [ num_lora_modules_layers, D x Hi + Ho x D ]
      # where the last dimension holds the in / out adapter weights for the associated module (e.g. attn_qkv) and model layer
      # each of the in / out tensors are first flattened and then concatenated together in the format above.
      # D=adapter_size (R value), Hi=hidden_size_in, Ho=hidden_size_out.
      {
        name: "lora_weights"
    	data_type: TYPE_FP16
    	dims: [ -1, -1 ]
    	optional: true
    	allow_ragged_batch: true
      },
      # module identifier (same size a first dimension of lora_weights)
      # See LoraModule::ModuleType for model id mapping
      #
      # "attn_qkv": 0     # compbined qkv adapter
      # "attn_q": 1       # q adapter
      # "attn_k": 2       # k adapter
      # "attn_v": 3       # v adapter
      # "attn_dense": 4   # adapter for the dense layer in attention
      # "mlp_h_to_4h": 5  # for llama2 adapter for gated mlp layer after attention / RMSNorm: up projection
      # "mlp_4h_to_h": 6  # for llama2 adapter for gated mlp layer after attention / RMSNorm: down projection
      # "mlp_gate": 7     # for llama2 adapter for gated mlp later after attention / RMSNorm: gate
      #
      # last dim holds [ module_id, layer_idx, adapter_size (D aka R value) ]
      {
        name: "lora_config"
    	data_type: TYPE_INT32
    	dims: [ -1, 3 ]
    	optional: true
    	allow_ragged_batch: true
      }
    ]
    output [
      {
        name: "output_ids"
        data_type: TYPE_INT32
        dims: [ -1, -1 ]
      },
      {
        name: "sequence_length"
        data_type: TYPE_INT32
        dims: [ -1 ]
      },
      {
        name: "cum_log_probs"
        data_type: TYPE_FP32
        dims: [ -1 ]
      },
      {
        name: "output_log_probs"
        data_type: TYPE_FP32
        dims: [ -1, -1 ]
      },
      {
        name: "context_logits"
        data_type: TYPE_FP32
        dims: [ -1, -1 ]
      },
      {
        name: "generation_logits"
        data_type: TYPE_FP32
        dims: [ -1, -1, -1 ]
      },
      {
        name: "batch_index"
        data_type: TYPE_INT32
        dims: [ 1 ]
      }
    ]
    instance_group [
      {
        count: 4
        kind : KIND_CPU
      }
    ]
    parameters: {
      key: "max_beam_width"
      value: {
        string_value: "1"
      }
    }
    parameters: {
      key: "FORCE_CPU_ONLY_INPUT_TENSORS"
      value: {
        string_value: "no"
      }
    }
    parameters: {
      key: "gpt_model_type"
      value: {
        string_value: "inflight_fused_batching"
      }
    }
    parameters: {
      key: "gpt_model_path"
      value: {
        string_value: "/opt/all_models/inflight_batcher_llm/tensorrt_llm/1"
      }
    }
    parameters: {
      key: "encoder_model_path"
      value: {
        string_value: "${encoder_engine_dir}"
      }
    }

    </details>
    parameters: {
      key: "max_tokens_in_paged_kv_cache"
      value: {
        string_value: ""
      }
    }
    parameters: {
      key: "max_attention_window_size"
      value: {
        string_value: "${max_attention_window_size}"
      }
    }
    parameters: {
      key: "sink_token_length"
      value: {
        string_value: "${sink_token_length}"
      }
    }
    parameters: {
      key: "batch_scheduler_policy"
      value: {
        string_value: "guaranteed_completion"
      }
    }
    parameters: {
      key: "kv_cache_free_gpu_mem_fraction"
      value: {
        string_value: "0.2"
      }
    }
    parameters: {
      key: "kv_cache_host_memory_bytes"
      value: {
        string_value: "${kv_cache_host_memory_bytes}"
      }
    }
    parameters: {
      key: "kv_cache_onboard_blocks"
      value: {
        string_value: "${kv_cache_onboard_blocks}"
      }
    }
    # enable_trt_overlap is deprecated and doesn't have any effect on the runtime
    # parameters: {
    #   key: "enable_trt_overlap"
    #   value: {
    #     string_value: "${enable_trt_overlap}"
    #   }
    # }
    parameters: {
      key: "exclude_input_in_output"
      value: {
        string_value: "${exclude_input_in_output}"
      }
    }
    parameters: {
      key: "cancellation_check_period_ms"
      value: {
        string_value: "${cancellation_check_period_ms}"
      }
    }
    parameters: {
      key: "stats_check_period_ms"
      value: {
        string_value: "${stats_check_period_ms}"
      }
    }
    parameters: {
      key: "iter_stats_max_iterations"
      value: {
        string_value: "${iter_stats_max_iterations}"
      }
    }
    parameters: {
      key: "request_stats_max_iterations"
      value: {
        string_value: "${request_stats_max_iterations}"
      }
    }
    parameters: {
      key: "enable_kv_cache_reuse"
      value: {
        string_value: "${enable_kv_cache_reuse}"
      }
    }
    parameters: {
      key: "normalize_log_probs"
      value: {
        string_value: "${normalize_log_probs}"
      }
    }
    parameters: {
      key: "enable_chunked_context"
      value: {
        string_value: "${enable_chunked_context}"
      }
    }
    parameters: {
      key: "gpu_device_ids"
      value: {
        string_value: "${gpu_device_ids}"
      }
    }
    parameters: {
      key: "lora_cache_optimal_adapter_size"
      value: {
        string_value: "${lora_cache_optimal_adapter_size}"
      }
    }
    parameters: {
      key: "lora_cache_max_adapter_size"
      value: {
        string_value: "${lora_cache_max_adapter_size}"
      }
    }
    parameters: {
      key: "lora_cache_gpu_memory_fraction"
      value: {
        string_value: "${lora_cache_gpu_memory_fraction}"
      }
    }
    parameters: {
      key: "lora_cache_host_memory_bytes"
      value: {
        string_value: "${lora_cache_host_memory_bytes}"
      }
    }
    parameters: {
      key: "decoding_mode"
      value: {
        string_value: "${decoding_mode}"
      }
    }
    parameters: {
      key: "executor_worker_path"
      value: {
        string_value: "/opt/tritonserver/backends/tensorrtllm/trtllmExecutorWorker"
      }
    }
    parameters: {
      key: "medusa_choices"
        value: {
          string_value: "${medusa_choices}"
      }
    }
    parameters: {
      key: "gpu_weights_percent"
        value: {
          string_value: "${gpu_weights_percent}"
      }
    }