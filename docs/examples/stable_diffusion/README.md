# Stable diffusion

This example focuses on showcasing two of Triton Inference Server's features:
* Using multiple frameworks in the same inference pipeline
* Using the Python Backend API to build complex non linear pipelines.

## How to run?

Before starting, clone this repository and navigate to the root folder. Use three different terminals for an easier user experience.

### Step 1: Prepare the Server Environment
* First, run the Triton Inference Server Container.
```
# Replace yy.mm with year and month of release. Eg. 22.08
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:yy.mm-py3 bash
```
* Next, install all the dependencies required by the models running in the python backend.

```
# PyTorch & Transformers Lib
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install transformers ftfy scipy
pip install transformers[onnxruntime]
```

As of the creation of this example, there are some unmerged optimizations for the diffusers repository, which are being used to accelerate the UNet Model. If you DO NOT wish to use the optimizations:
```
pip install diffusers
```
If you wish to use the optimizations:
```
# this pip install take ~ 20 minutes
pip install git+https://github.com/facebookresearch/xformers@51dd119#egg=xformers
git clone https://github.com/MatthieuTPHR/diffusers.git
cd diffusers
git checkout memory_efficient_attention
pip install -e .
export USE_MEMORY_EFFICIENT_ATTENTION=1
```

### Step 2: Exporting and converting the models
Use the NGC PyTorch container, to export and convert the models.

```
# Replace yy.mm with year and month of release. Eg. 22.08
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
# Replace yy.mm with year and month of release. Eg. 22.08
docker run -it --net=host -v /home/tvarshney:/mount -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:yy.mm-py3-sdk bash

# Client with no GUI
python3 client.py

# Client with GUI
pip install gradio
python3 gui/client.py --triton_url="localhost:8000"
```
Note: First Inference query may take more time than successive queries
