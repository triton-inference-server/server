#!/bin/bash
python3 -m cProfile -o cprofile_triton_chat_async.prof ../openai_frontend/main.py --model-repository ../tests/vllm_models/ --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct
