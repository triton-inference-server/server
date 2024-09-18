#!/bin/bash
python3 -m cProfile -o cprofile_vllm_serve.prof -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --served-model-name llama-3.1-8b-instruct --disable-log-requests
