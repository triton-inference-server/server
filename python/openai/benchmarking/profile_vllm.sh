#!/bin/bash
OUTPUT=${OUTPUT:-"cprofile_vllm_serve.prof"}
DIR="cprofile"
mkdir -p ${DIR}
python3 -m cProfile -o ${DIR}/${OUTPUT} -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --served-model-name llama-3.1-8b-instruct --disable-log-requests
