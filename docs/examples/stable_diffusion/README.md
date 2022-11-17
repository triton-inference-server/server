<!--
# Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Stable diffusion

*Note*: This tutorial aims at demonstrating the ease of deployment and doesn't incorporate all possible optimizations using the NVIDIA ecosystem.

This example focuses on showcasing two of Triton Inference Server's features:
* Using multiple frameworks in the same inference pipeline. Refer [this for more information](https://github.com/triton-inference-server/backend#where-can-i-find-all-the-backends-that-are-available-for-triton) about supported frameworks.
* Using the Python Backend's [Business Logic Scripting](https://github.com/triton-inference-server/python_backend#business-logic-scripting) API to build complex non linear pipelines.

It is recommended to watch [this explainer video](https://youtu.be/JgP2WgNIq_w) with discusses the pipeline, before proceeding with the example. 

## How to run?

Before starting, clone this repository and navigate to the root folder. Use three different terminals for an easier user experience.

### Step 1: Prepare the Server Environment
* First, run the Triton Inference Server Container.
```
# Replace yy.mm with year and month of release. Eg. 22.08
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:yy.mm-py3 bash
```
* Next, install all the dependencies required by the models running in the python backend and login with your [huggingface token](https://huggingface.co/settings/tokens)(Account on [HuggingFace](https://huggingface.co/) is required).

```
# PyTorch & Transformers Lib
pip install torch torchvision torchaudio
pip install transformers ftfy scipy accelerate diffusers
pip install transformers[onnxruntime]
huggingface-cli login
```

### Step 2: Exporting and converting the models
Use the NGC PyTorch container, to export and convert the models.

```
docker run -it --gpus all -p 8888:8888 -v ${PWD}:/mount nvcr.io/nvidia/pytorch:yy.mm-py3

pip install transformers ftfy scipy
pip install transformers[onnxruntime]
pip install diffusers
huggingface-cli login
cd /mount
python export.py

# Accelerating VAE with TensorRT
trtexec --onnx=vae.onnx --saveEngine=vae.plan --minShapes=latent_sample:1x4x64x64 --optShapes=latent_sample:4x4x64x64 --maxShapes=latent_sample:8x4x64x64 --fp16

# Place the models in the model repository
mkdir model_repository/vae/1
mkdir model_repository/text_encoder/1
mv vae.plan model_repository/vae/1/model.plan
mv encoder.onnx model_repository/text_encoder/1/model.onnx
```

### Step 3: Launch the Server
From the server container, launch the Triton Inference Server.
```
tritonserver --model-repository=/models
```

### Step 4: Run the client
Use the client container and run the client.
```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:yy.mm-py3-sdk bash

# Client with no GUI
python3 client.py

# Client with GUI
pip install gradio packaging
python3 gui/client.py --triton_url="localhost:8001"
```
Note: First Inference query may take more time than successive queries
