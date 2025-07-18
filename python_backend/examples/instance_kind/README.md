<!--
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Model Instance Kind Example

Triton model configuration allows users to provide kind to [instance group
settings.](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups)
A python backend model can be written to respect the kind setting to control
the execution of a model instance either on CPU or GPU.

In this example, we demonstrate how this can be achieved for your python model.
We will use a `ResNet50` model as our base model for this example.

## Create a ResNet50 model repository

We will use the files that come with this example to create the model
repository.

First, download the [client.py](client.py), [config.pbtxt](config.pbtxt),
[resnet50_labels.txt](resnet50_labels.txt), and [model.py](model.py)
to your local machine.

Next, in the same directory with the four aforementioned files, create the model
repository with the following commands:
```
mkdir -p models/resnet50/1 &&
mv model.py models/resnet50/1/ &&
mv config.pbtxt models/resnet50/
```

## Pull the Triton Docker images

We need to install Docker and NVIDIA Container Toolkit before proceeding, refer
to the
[installation steps](https://github.com/triton-inference-server/server/tree/main/docs#installation).

To pull the latest containers, run the following commands:
```
docker pull nvcr.io/nvidia/tritonserver:<yy.mm>-py3
docker pull nvcr.io/nvidia/tritonserver:<yy.mm>-py3-sdk
```
See the installation steps above for the `<yy.mm>` version.

For example, if the latest version is `23.01`, the above commands translate
to the following:
```
docker pull nvcr.io/nvidia/tritonserver:23.01-py3
docker pull nvcr.io/nvidia/tritonserver:23.01-py3-sdk
```

Be sure to replace the `<yy.mm>` with the version pulled for all the remaining
parts of this example.

## Start the Triton Server

At the directory where we copied our resnet50 model (at where the "models"
folder is located), run the following command:
```
docker run --gpus all --shm-size 1G -it --rm -p 8000:8000 -v `pwd`:/instance_kind nvcr.io/nvidia/tritonserver:<yy.mm>-py3 /bin/bash
```

Inside the container, we need to install `torch`, `torchvision` and `pillow` to run
this example. We recommend to use `pip` method for the installation:

```
pip3 install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html torchvision==0.14.0+cu117 pillow
```

Finally, we need to start the Triton Server:
```
tritonserver --model-repository /instance_kind/models
```

To leave the container for the next step, press: `CTRL + P + Q`.

## Start the Triton SDK Container and Test Inference

To start the sdk container, run the following command:
```
docker run --gpus all --network=host --pid=host --ipc=host -v `pwd`:/instance_kind -ti nvcr.io/nvidia/tritonserver:<yy.mm>-py3-sdk /bin/bash
```

The `client.py` requires the following packages to be installed: `torch`,
`torchvision`, `pillow` and `validators`.  Similarly, we recommend to use `pip`
method for the installation:

```
pip3 install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html torchvision==0.14.0+cu117 pillow validators
```

Finally, let's test an inference call with the following command:
```
python client.py
```
On a first run, a successful inference will print the following at the end:
```
Downloading: "https://github.com/NVIDIA/DeepLearningExamples/zipball/torchhub" to /root/.cache/torch/hub/torchhub.zip
Results is class: TABBY
PASS: ResNet50
```
It may take some time due to `torchhub` downloads, but any future calls
will be quicker, since the client will use already downloaded artifacts.

## Test Instance Kind

Provided `config.pbtxt` sets the instance group setting to `KIND_CPU`,
which enables the execution of a model on the CPU.
To test that your model is actually loaded onto CPU, run the following:
```
python client.py -v
```
The `-v` argument asks the client to request model's confiuration from
the server and prints it in your console:
```
{
    ...,
    "instance_group": [
        {
            "name": "resnet50_0",
            "kind": "KIND_CPU",
            "count": 1,
            "gpus": [],
            "secondary_devices": [],
            "profile": [],
            "passive": false,
            "host_policy": ""
        }
    ],
    ...
}
Results is class: TABBY
PASS: ResNet50 instance kind
```

Based on the printed model config, we can see that `instance_group` field
has `kind` entry, which is set to `KIND_CPU`.

To change an `instance_group` parameter to `KIND_GPU`, a user can simply replace
`KIND_CPU` with `KIND_GPU` in the `config.pbtxt`. After restarting the server
with an updated config file, a successful inference request with `-v` argument
will result into the similar output, but with an updated `instance_group` entry:
```
{
    ...,
    "instance_group": [
        {
            "name": "resnet50_0",
            "kind": "KIND_GPU",
            "count": 1,
            "gpus": [
                0
            ],
            "secondary_devices": [],
            "profile": [],
            "passive": false,
            "host_policy": ""
        }
    ],
    ...
}
Results is class: TABBY
PASS: ResNet50 instance kind
```
It is also possible to load multiple model instances on CPU and GPU
if necessary.

Below the instance group setting will create two model instances,
one on CPU and other on GPU.
```
instance_group [{ kind: KIND_CPU }, { kind: KIND_GPU}]
```

For more information on possible model configurations,
check out the Triton Server documentation [here](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#model-configuration)