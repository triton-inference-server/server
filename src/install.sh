pip install vllm &&
mkdir -p /opt/tritonserver/backends/vllm &&
wget -P /opt/tritonserver/backends/vllm https://raw.githubusercontent.com/triton-inference-server/vllm_backend/main/src/model.py
pip install git+https://github.com/triton-inference-server/triton_cli.git 