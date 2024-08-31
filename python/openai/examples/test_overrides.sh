#!/bin/bash
export TEST_BACKEND="vllm"
export TEST_MODEL=llama-3.1-8b-instruct
export TEST_MODEL_REPOSITORY="${PWD}/tests/vllm_models/"
export TEST_TOKENIZER="meta-llama/Meta-Llama-3.1-8B-Instruct"
python3 -m pytest -s -v tests/
