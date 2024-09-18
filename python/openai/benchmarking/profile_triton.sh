#!/bin/bash
python3 -m cProfile -o cprofile_triton_chat_async.prof /mnt/triton/jira/7088-openai/server/python/openai/openai_frontend/main.py --model-repository /workspace/openai/tests/vllm_models/ --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct
