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

# Auto-Complete Example

This example shows how to implement
[`auto_complete_config`](https://github.com/triton-inference-server/python_backend/#auto_complete_config)
function in Python backend to provide
[`max_batch_size`](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#maximum-batch-size),
[`input`](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#inputs-and-outputs)
and [`output`](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#inputs-and-outputs)
properties. These properties will allow Triton to load the Python model with
[Minimal Model Configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#minimal-model-configuration)
in absence of a configuration file.

The
[model repository](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md)
should contain [nobatch_auto_complete](./nobatch_model.py), and
[batch_auto_complete](./batch_model.py) models.
The max_batch_size of [nobatch_auto_complete](./nobatch_model.py) model is set
to zero, whereas the max_batch_size of [batch_auto_complete](./batch_model.py)
model is set to 4. For models with a non-zero value of max_batch_size, the
configuration can specify a different value of max_batch_size as long as it
does not exceed the value set in the model file.

The
[nobatch_auto_complete](./nobatch_model.py) and
[batch_auto_complete](./batch_model.py) models calculate the sum and difference
of the `INPUT0` and `INPUT1` and put the results in `OUTPUT0` and `OUTPUT1`
respectively.

## Deploying the Auto-Complete Models

1. Create the model repository:

```console
mkdir -p models/nobatch_auto_complete/1/
mkdir -p models/batch_auto_complete/1/

# Copy the Python models
cp examples/auto_complete/nobatch_model.py models/nobatch_auto_complete/1/model.py
cp examples/auto_complete/batch_model.py models/batch_auto_complete/1/model.py
```
**Note that we don't need a model configuration file since Triton will use the
auto-complete model configuration provided in the Python model.**

2. Start the tritonserver:

```
tritonserver --model-repository `pwd`/models
```

## Running inferences on Nobatch and Batch models:

Send inference requests using [client.py](./client.py).

```
python3 examples/auto_complete/client.py
```

You should see an output similar to the output below:

```
'nobatch_auto_complete' configuration matches the expected auto complete configuration

'batch_auto_complete' configuration matches the expected auto complete configuration

PASS: auto_complete

```

The [nobatch_model.py](./nobatch_model.py) and [batch_model.py](./batch_model.py)
model files are heavily commented with explanations about how to utilize
`set_max_batch_size`, `add_input`, and `add_output`functions to set
`max_batch_size`, `input` and `output` properties of the model.

### Explanation of the Client Output

For each model, the [client.py](./client.py) first requests the model
configuration from Triton to validate if the model configuration has been
registered as expected. The client then sends an inference request to verify
whether the inference has run properly and the result is correct.
