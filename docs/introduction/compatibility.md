<!--
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

[Please visit Deep Learning Framework (DLFW) website for the complete compatibility matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

# Release Compatibility Matrix
 1. [Container Name: trtllm-python-py3](#container-name-trtllm-python-py3)
 2. [Container Name: vllm-python-py3](#container-name-vllm-python-py3)
 3. [ONNX Versions](#onnx-versions)

## Container Name: trtllm-python-py3

| Triton release version	 | NGC Tag	 | Python version	 | Torch version | TensorRT version | TensorRT-LLM version | CUDA version | CUDA Driver version | Size |
| --- | ---  | --- | --- | --- | --- | --- | --- | --- |
| 24.10 | nvcr.io/nvidia/tritonserver:24.10-trtllm-python-py3 | Python 3.10.12  | 2.4.0a0%2B3bcc3cddb5.nv24.7 | 10.4.0 | 0.14.0 | 12.5.1.007 | 555.42.06 | 21G |
| 24.09 | nvcr.io/nvidia/tritonserver:24.09-trtllm-python-py3 | Python 3.10.12  | 2.4.0a0%2B3bcc3cddb5.nv24.7 | 10.4.0 | 0.13.0 | 12.5.1.007 | 555.42.06 | 21G |
| 24.08 | nvcr.io/nvidia/tritonserver:24.08-trtllm-python-py3 | Python 3.10.12 | 2.4.0a0%2B3bcc3cddb5.nv24.7 | 10.3.0 | 0.12.0 | 12.5.1.007 | 555.42.06 | 21G |
| 24.07 | nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3 | Python 3.10.12 | 2.4.0a0%2B07cecf4168.nv24.5 | 10.1.0 | 0.11.0 | 12.4.1.003 | 550.54.15 | 23G |
| 24.06 | nvcr.io/nvidia/tritonserver:24.06-trtllm-python-py3 | Python 3.10.12  | 2.3.0a0%2B40ec155e58.nv24.3 | 10.0.1 | 0.10.0 | 12.4.0.041 | 550.54.14 | 31G |
| 24.05 | nvcr.io/nvidia/tritonserver:24.05-trtllm-python-py3 | Python 3.10.12  | 2.3.0a0%2Bebedce2 | 10.0.1.6  | 0.9.0 |  12.3.2.001 | 545.23.08 | 34G |
| 24.04 | nvcr.io/nvidia/tritonserver:24.04-trtllm-python-py3 | Python 3.10.12  | 2.3.0a0%2Bebedce2 | 9.3.0.post12.dev1 | 0.9.0  | 12.3.2.001 | 545.23.08 | 34G |

## Container Name: vllm-python-py3

| Triton release version	 | NGC Tag	 | Python version	 | vLLM version | CUDA version | CUDA Driver version | Size |
| --- | --- | --- | --- | --- | --- | --- |
| 24.10 | nvcr.io/nvidia/tritonserver:24.10-vllm-python-py3 | Python 3.10.12 | 0.5.5 | 12.6.2.004 | 560.35.03 | 19G |
| 24.09 | nvcr.io/nvidia/tritonserver:24.09-vllm-python-py3 | Python 3.10.12 | 0.5.3.post1 | 12.6.1.006 | 560.35.03 | 19G |
| 24.08 | nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3 | Python 3.10.12  | 0.5.0 post1 | 12.6.0.022 | 560.35.03 | 19G |
| 24.07 | nvcr.io/nvidia/tritonserver:24.07-vllm-python-py3 | Python 3.10.12  | 0.5.0 post1 | 12.5.1 | 555.42.06 | 19G |
| 24.06 | nvcr.io/nvidia/tritonserver:24.06-vllm-python-py3 | Python 3.10.12  | 0.4.3 | 12.5.0.23 | 555.42.02 | 18G |
| 24.05 | nvcr.io/nvidia/tritonserver:24.05-vllm-python-py3 | Python 3.10.12  | 0.4.0 post1 | 12.4.1 | 550.54.15 | 18G |
| 24.04 | nvcr.io/nvidia/tritonserver:24.04-vllm-python-py3 | Python 3.10.12  | 0.4.0 post1 | 12.4.1 | 550.54.15 | 17G |

## ONNX Versions

| Triton release version	 | ONNX Runtime	 |
| --- | --- |
| 24.10 | 1.19.2 |
| 24.09 | 1.19.2 |
| 24.08 | 1.18.1 |
| 24.07 | 1.18.1 |
| 24.06 | 1.18.0 |
| 24.05 | 1.18.0 |
| 24.04 | 1.17.3 |
