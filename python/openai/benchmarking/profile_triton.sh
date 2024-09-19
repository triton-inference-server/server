#!/bin/bash
OUTPUT=${OUTPUT:-"cprofile_triton_chat_async.prof"}
DIR="cprofile"
mkdir -p ${DIR}
python3 -m cProfile -o ${DIR}/${OUTPUT} ../openai_frontend/main.py --model-repository ../tests/vllm_models/ --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct
