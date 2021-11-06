<!--
# Copyright 2018-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# MLflow Triton

MLflow plugin for deploying your models from MLflow to Triton Inference Server. Scripts
are included for publishing models, which are in Triton recognized structure,
to your MLflow Model Registry.

## Requirements

* MLflow
* Triton Python HTTP client
* Triton Inference Server

## Installation

The plugin can be installed from source using the following commands

```
python setup.py install
```

## Quick Start

In this documentation, we will use the files in `examples` to showcase how
the plugin interacts with Triton Infernce Server.

### Start Triton Inference Server in EXPLICIT mode

The MLflow Triton plugin must work with a running Triton server, see
[documentation](https://github.com/triton-inference-server/server/blob/main/docs/quickstart.md)
of Triton Inference Server for how to start the server. Note that
the server should be run in EXPLICIT mode (`--model-control-mode=explicit`)
to exploit the deployment feature of the plugin.

Once the server has started, the following environment must be set so that the plugin
can interact with the server properly:
* `TRITON_URL`: The address to the Triton HTTP endpoint
* `TRITON_MODEL_REPO`: The path to the Triton model repository

### Publish models to MLflow

The `publish_model_to_mlflow.py` script is used to publish `triton` flavor models
to MLflow. A `triton` flavor model is a directory containing the model files
following the [model layout](https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md#repository-layout).
Below is an example usage:

```
cd /mlflow/scripts

python publish_model_to_mlflow.py --model_name onnx_float32_int32_int32 --model_directory <path-to-the-examples-directory>/onnx_float32_int32_int32 --flavor triton
```

### Deploy models tracked in MLflow to Triton

```
mlflow deployments create -t triton --flavor triton --name onnx_float32_int32_int32 -m models:/onnx_float32_int32_int32/1
```

### Perform inference

```
mlflow deployments predict -t triton --name onnx_float32_int32_int32 --input-path <path-to-the-examples-directory>/input_file --output-path output_file
```

The inference result will be written in `output_file` and you may compare it
with the results in `expected_output`

## MLflow Deployments

The following deployment functions are implemented within the plugin.

### Create Deployment

To create a deployment use the following command

##### CLI
```
mlflow deployments create -t triton --flavor triton --name model_name -m models:/model_name/1
```

##### Python API
```
from mlflow.deployments import get_deploy_client
client = get_deploy_client('triton')
client.create_deployment("model_name", "models:/model_name/1", flavor="triton")
```

### Delete Deployment

##### CLI
```
mlflow deployments delete -t triton --name model_name
```

##### Python API
```
client.delete_deployment("model_name")
```

### Update Deployment

##### CLI
```
mlflow deployments update -t triton --flavor triton --name model_name -m models:/model_name/2
```

##### Python API
```
client.update_deployment("model_name", "models:/model_name/2", flavor="triton")
```

### List Deployments

##### CLI
```
mlflow deployments list -t triton
```

##### Python API
```
client.list_deployments()
```

### Get Deployment

##### CLI
```
mlflow deployments get -t triton --name model_name
```

##### Python API
```
client.get_deployment("model_name")
```

### Run Inference on Deployments

##### CLI
```
mlflow deployments predict -t triton --name model_name --input-path input_file --output-path output_file

```

##### Python API
```
client.predict("model_name", inputs)
```
