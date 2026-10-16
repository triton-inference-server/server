# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# One image for every L0_trtllm_llmapi scenario, on top of the tritonserver
# image built as in docs/customization_guide/build.md, "Building With the
# TensorRT-LLM Backend" (BASE_IMAGE); the ensemble scenario needs its
# '--backend=ensemble'. Every scenario serves the LLM API model files that build
# installed under /opt/tritonserver/tensorrtllm, so nothing else is needed
# from a QA image. Run one scenario per container:
#
#   docker build -t trtllm_llmapi_qa --build-arg BASE_IMAGE=tritonserver \
#       --build-context common=qa/common qa/L0_trtllm_llmapi
#   docker run --gpus all --net host --ipc host \
#       -v ~/.cache/huggingface:/root/.cache/huggingface \
#       trtllm_llmapi_qa <scenario>
#
# common/ is a separate build context so the image gets the qa/common of the
# commit under test (action_log). tritonclient and requests are installed
# explicitly: the runtime image ships neither. pytest comes with the
# TensorRT-LLM base image.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY --from=common . /opt/tritonserver/qa/common/
COPY . /opt/tritonserver/qa/L0_trtllm_llmapi/
WORKDIR /opt/tritonserver/qa/L0_trtllm_llmapi

RUN pip3 install --no-cache-dir tritonclient[all] requests

ENTRYPOINT ["python3", "pytest.py"]
