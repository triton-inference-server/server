<!--
# Copyright 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# JAX Example

In this section, we demonstrate an end-to-end example for using
[JAX](https://jax.readthedocs.io/en/latest/) in Python Backend.

## Create a JAX AddSub model repository

We will use the files that come with this example to create the model
repository.

First, download the [client.py](client.py), [config.pbtxt](config.pbtxt) and
[model.py](model.py) to your local machine.

Next, at the directory where the three files located, create the model
repository with the following commands:
```
mkdir -p models/jax/1
mv model.py models/jax/1
mv config.pbtxt models/jax
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

At the time of writing, the latest version is `23.04`, which translates to the
following commands:
```
docker pull nvcr.io/nvidia/tritonserver:23.04-py3
docker pull nvcr.io/nvidia/tritonserver:23.04-py3-sdk
```

Be sure to replace the `<yy.mm>` with the version pulled for all the remaining
parts of this example.

## Start the Triton Server

At the directory where we created the JAX models (at where the "models" folder
is located), run the following command:
```
docker run --gpus all -it --rm -p 8000:8000 -v `pwd`:/jax nvcr.io/nvidia/tritonserver:<yy.mm>-py3 /bin/bash
```

Inside the container, we need to install JAX to run this example.

We recommend using the `pip` method mentioned in the
[JAX documentation](https://github.com/google/jax#pip-installation-gpu-cuda).
Make sure that JAX is available in the same Python environment as other
dependencies.

To install for this example, run the following command:
```
pip3 install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Finally, we need to start the Triton Server, run the following command:
```
tritonserver --model-repository=/jax/models
```

To leave the container for the next step, press: `CTRL + P + Q`.

## Test inference

At the directory where the client.py is located, run the following command:
```
docker run --rm --net=host -v `pwd`:/jax nvcr.io/nvidia/tritonserver:<yy.mm>-py3-sdk python3 /jax/client.py
```

A successful inference will print the following at the end:
```
INPUT0 ([0.89262384 0.645457   0.18913145 0.17099917]) + INPUT1 ([0.5703733  0.21917151 0.22854741 0.97336507]) = OUTPUT0 ([1.4629972  0.86462855 0.41767886 1.1443642 ])
INPUT0 ([0.89262384 0.645457   0.18913145 0.17099917]) - INPUT1 ([0.5703733  0.21917151 0.22854741 0.97336507]) = OUTPUT0 ([ 0.32225055  0.4262855  -0.03941596 -0.8023659 ])
PASS: jax
```
Note: You inputs can be different from the above, but the outputs always
correspond to its inputs.
