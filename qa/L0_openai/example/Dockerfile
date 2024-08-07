ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:24.07-vllm-python-py3
FROM ${BASE_IMAGE}
RUN pip install /opt/tritonserver/python/*.whl
RUN pip install "fastapi==0.111.1" "pytest==8.1.1"
