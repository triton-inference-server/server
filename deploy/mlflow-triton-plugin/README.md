<!--
# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

MLflow plugin for deploying your models from MLflow to Triton Inference Server.
Scripts are included for publishing models, which are in Triton recognized
structure, to your MLflow Model Registry.

### Supported flavors

MLFlow Triton plugin currently supports the following flavors, you may
substitute the flavor specification in the example below according to the model
to be deployed.

* onnx
* triton

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
the plugin interacts with Triton Infernce Server. The `onnx_float32_int32_int32`
model in `examples` is a simple model that takes two float32 inputs, INPUT0 and
INPUT1, with shape [-1, 16], and produces two int32 outputs, OUTPUT0 and
OUTPUT1, where OUTPUT0 is the element-wise summation of INPUT0 and INPUT1 and
OUTPUT1 is the element-wise subtraction of INPUT0 and INPUT1.

### Start Triton Inference Server in EXPLICIT mode

The MLflow Triton plugin must work with a running Triton server, see
[documentation](https://github.com/triton-inference-server/server/blob/main/docs/getting_started/quickstart.md)
of Triton Inference Server for how to start the server. Note that
the server should be run in EXPLICIT mode (`--model-control-mode=explicit`)
to exploit the deployment feature of the plugin.

Once the server has started, the following environment must be set so that the plugin
can interact with the server properly:
* `TRITON_URL`: The address to the Triton HTTP endpoint
* `TRITON_MODEL_REPO`: The path to the Triton model repository

### Publish models to MLflow

#### ONNX flavor

The MLFlow ONNX built-in functionalities can be used to publish `onnx` flavor
models to MLFlow directly, and the MLFlow Triton plugin will prepare the model
to the format expected by Triton. You may also log
[`config.pbtxt`](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_configuration.md)
as additonal artifact which Triton will be used to serve the model. Otherwise,
the server should be run with auto-complete feature enabled
(`--strict-model-config=false`) to generate the model configuration.

```
import mlflow.onnx
import onnx
model = onnx.load("examples/onnx_float32_int32_int32/1/model.onnx")
mlflow.onnx.log_model(model, "triton", registered_model_name="onnx_float32_int32_int32")
```

#### Triton flavor

For other model frameworks that Triton supports but not yet recognized by
the MLFlow Triton plugin, the `publish_model_to_mlflow.py` script can be used to
publish `triton` flavor models to MLflow. A `triton` flavor model is a directory
containing the model files following the
[model layout](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md#repository-layout).
Below is an example usage:

```
cd /scripts

python publish_model_to_mlflow.py --model_name onnx_float32_int32_int32 --model_directory <path-to-the-examples-directory>/onnx_float32_int32_int32 --flavor triton
```

### Deploy models tracked in MLflow to Triton

Once a model is published and tracked in MLflow, it can be deployed to Triton
via MLflow's deployments command, the following command will download the model
to Triton's model repository and request Triton to load the model.

```
mlflow deployments create -t triton --flavor triton --name onnx_float32_int32_int32 -m models:/onnx_float32_int32_int32/1
```

### Perform inference

After the model is deployed, the following command is the CLI usage to send
inference request to a deployment.

```
mlflow deployments predict -t triton --name onnx_float32_int32_int32 --input-path <path-to-the-examples-directory>/input.json --output-path output.json
```

The inference result will be written in `output.json` and you may compare it
with the results in `expected_output.json`

## MLflow Deployments

"MLflow Deployments" is a set of MLflow APIs for deploying MLflow models to
custom serving tools. The MLflow Triton plugin implements the following
deployment functions to support the interaction with Triton server in MLflow.

### Create Deployment

MLflow deployments create API deploys a model to the Triton target, which will
download the model to Triton's model repository and request Triton to load the
model.

To create a MLflow deployment using CLI:

```
mlflow deployments create -t triton --flavor triton --name model_name -m models:/model_name/1
```

To create a MLflow deployment using Python API:

```
from mlflow.deployments import get_deploy_client
client = get_deploy_client('triton')
client.create_deployment("model_name", "models:/model_name/1", flavor="triton")
```

### Delete Deployment

MLflow deployments delete API removes an existing deployment from the Triton
target, which will remove the model in Triton's model repository and request
Triton to unload the model.

To delete a MLflow deployment using CLI

```
mlflow deployments delete -t triton --name model_name
```

To delete a MLflow deployment using CLI

```
from mlflow.deployments import get_deploy_client
client = get_deploy_client('triton')
client.delete_deployment("model_name")
```

### Update Deployment

MLflow deployments update API updates an existing deployment with another model
(version) tracked in MLflow, which will overwrite the model in Triton's model
repository and request Triton to reload the model.

To update a MLflow deployment using CLI

```
mlflow deployments update -t triton --flavor triton --name model_name -m models:/model_name/2
```

To update a MLflow deployment using Python API

```
from mlflow.deployments import get_deploy_client
client = get_deploy_client('triton')
client.update_deployment("model_name", "models:/model_name/2", flavor="triton")
```

### List Deployments

MLflow deployments list API lists all existing deployments in Triton target.

To list all MLflow deployments using CLI

```
mlflow deployments list -t triton
```

To list all MLflow deployments using Python API

```
from mlflow.deployments import get_deploy_client
client = get_deploy_client('triton')
client.list_deployments()
```

### Get Deployment

MLflow deployments get API returns information regarding a specific deployments
in Triton target.

To list a specific MLflow deployment using CLI
```
mlflow deployments get -t triton --name model_name
```

To list a specific MLflow deployment using Python API
```
from mlflow.deployments import get_deploy_client
client = get_deploy_client('triton')
client.get_deployment("model_name")
```

### Run Inference on Deployments

MLflow deployments predict API runs inference by preparing and sending the
request to Triton and returns the Triton response.

To run inference using CLI

```
mlflow deployments predict -t triton --name model_name --input-path input_file --output-path output_file

```

To run inference using Python API

```
from mlflow.deployments import get_deploy_client
client = get_deploy_client('triton')
client.predict("model_name", inputs)
```
