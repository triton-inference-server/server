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

# Example of using BLS with decoupled models

In this section we demonstrate an end-to-end example for
[BLS](../../README.md#business-logic-scripting) in Python backend. The
[model repository](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md)
should contain [square](../decoupled) model. The [square](../decoupled) model
will send 'n' responses where 'n' is the value of input `IN`. For each response,
output `OUT` will equal the value of `IN`. This example is broken into two
sections. The first section demonstrates how to perform synchronous BLS requests
and the second section shows how to execute asynchronous BLS requests.

## Synchronous BLS Requests with Decoupled Models

The goal of `bls_decoupled_sync` model is to calculate the sum of the responses
returned from the [square](../decoupled) model and return the summation as the final response. The value of input 'IN' will be passed as an input to the
[square](../decoupled) model which determines how many responses the
[square](../decoupled) model will generate.

1. Create the model repository:

```console
mkdir -p models/bls_decoupled_sync/1
mkdir -p models/square_int32/1

# Copy the Python models
cp examples/bls_decoupled/sync_model.py models/bls_decoupled_sync/1/model.py
cp examples/bls_decoupled/sync_config.pbtxt models/bls_decoupled_sync/config.pbtxt
cp examples/decoupled/square_model.py models/square_int32/1/model.py
cp examples/decoupled/square_config.pbtxt models/square_int32/config.pbtxt
```

2. Start the tritonserver:

```
tritonserver --model-repository `pwd`/models
```

3. Send inference requests to server:

```
python3 examples/bls_decoupled/sync_client.py
```

You should see an output similar to the output below:

```
==========model result==========
The square value of [4] is [16]

==========model result==========
The square value of [2] is [4]

==========model result==========
The square value of [0] is [0]

==========model result==========
The square value of [1] is [1]

PASS: BLS Decoupled Sync
```

The [sync_model.py](./sync_model.py) model file is heavily commented with
explanations about each of the function calls.

### Explanation of the Client Output

The [client.py](./sync_client.py) sends 4 inference requests to the
`bls_decoupled_sync` model with the input as: [4], [2], [0] and [1]
respectively. In compliance with the behavior of the sync BLS model,
it will expect the output to be the square value of the input.

## Asynchronous BLS Requests with Decoupled Models

In this section we explain how to send multiple BLS requests without waiting for
their response. Asynchronous execution of BLS requests will not block your
model execution and can lead to speedups under certain conditions.

The `bls_decoupled_async` model will perform two async BLS requests on the
[square](../decoupled) model. Then, it will wait until the inference requests
are completed. It will calculate the sum of the output `OUT` from the
[square](../decoupled) model in both two requests to construct the final
inference response object using these tensors.

1. Create the model repository:

```console
mkdir -p models/bls_decoupled_async/1
mkdir -p models/square_int32/1

# Copy the Python models
cp examples/bls_decoupled/async_model.py models/bls_decoupled_async/1/model.py
cp examples/bls_decoupled/async_config.pbtxt models/bls_decoupled_async/config.pbtxt
cp examples/decoupled/square_model.py models/square_int32/1/model.py
cp examples/decoupled/square_config.pbtxt models/square_int32/config.pbtxt
```

2. Start the tritonserver:

```
tritonserver --model-repository `pwd`/models
```

3. Send inference requests to server:

```
python3 examples/bls_decoupled/async_client.py
```

You should see an output similar to the output below:

```
==========model result==========
Two times the square value of [4] is [32]

==========model result==========
Two times the square value of [2] is [8]

==========model result==========
Two times the square value of [0] is [0]

==========model result==========
Two times the square value of [1] is [2]

PASS: BLS Decoupled Async
```

The [async_model.py](./async_model.py) model file is heavily commented with
explanations about each of the function calls.

### Explanation of the Client Output

The [client.py](./async_client.py) sends 4 inference requests to the
'bls_decoupled_sync' model with the input as: [4], [2], [0] and [1]
respectively. In compliance with the behavior of sync BLS model model,
it will expect the output to be two time the square value of the input.
