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

# Decoupled Model Examples

In this section we demonstrate an end-to-end examples for developing and
serving [decoupled models](../../README.md#decoupled-mode) in Python backend.

[repeat_model.py](repeat_model.py) and [square_model.py](square_model.py) demonstrate
how to write a decoupled model where each request can generate 0 to many responses.
These files are heavily commented to describe each function call.
These example models are designed to show the flexibility available to decoupled models
and in no way should be used in production. These examples circumvents
the restriction placed by the
[instance count](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups)
and allows multiple requests to be in process even for single instance. In
real deployment, the model should not allow the caller thread to return from
`execute` until that instance is ready to handle another set of requests.

## Deploying the Decoupled Models

1. Create the model repository:

```console
mkdir -p models/repeat_int32/1
mkdir -p models/square_int32/1

# Copy the Python models
cp examples/decoupled/repeat_model.py models/repeat_int32/1/model.py
cp examples/decoupled/repeat_config.pbtxt models/repeat_int32/config.pbtxt
cp examples/decoupled/square_model.py models/square_int32/1/model.py
cp examples/decoupled/square_config.pbtxt models/square_int32/config.pbtxt
```

2. Start the tritonserver:

```
tritonserver --model-repository `pwd`/models
```

## Running inference on Repeat model:

Send inference requests to repeat model using [repeat_client.py](repeat_client.py).

```
python3 examples/decoupled/repeat_client.py
```

You should see an output similar to the output below:

```
stream started...
async_stream_infer
model_name: "repeat_int32"
id: "0"
inputs {
  name: "IN"
  datatype: "INT32"
  shape: 4
}
inputs {
  name: "DELAY"
  datatype: "UINT32"
  shape: 4
}
inputs {
  name: "WAIT"
  datatype: "UINT32"
  shape: 1
}
outputs {
  name: "OUT"
}
outputs {
  name: "IDX"
}
raw_input_contents: "\004\000\000\000\002\000\000\000\000\000\000\000\001\000\000\000"
raw_input_contents: "\001\000\000\000\002\000\000\000\003\000\000\000\004\000\000\000"
raw_input_contents: "\005\000\000\000"

enqueued request 0 to stream...
infer_response {
  model_name: "repeat_int32"
  model_version: "1"
  id: "0"
  outputs {
    name: "IDX"
    datatype: "UINT32"
    shape: 1
  }
  outputs {
    name: "OUT"
    datatype: "INT32"
    shape: 1
  }
  raw_output_contents: "\000\000\000\000"
  raw_output_contents: "\004\000\000\000"
}

infer_response {
  model_name: "repeat_int32"
  model_version: "1"
  id: "0"
  outputs {
    name: "IDX"
    datatype: "UINT32"
    shape: 1
  }
  outputs {
    name: "OUT"
    datatype: "INT32"
    shape: 1
  }
  raw_output_contents: "\001\000\000\000"
  raw_output_contents: "\002\000\000\000"
}

infer_response {
  model_name: "repeat_int32"
  model_version: "1"
  id: "0"
  outputs {
    name: "IDX"
    datatype: "UINT32"
    shape: 1
  }
  outputs {
    name: "OUT"
    datatype: "INT32"
    shape: 1
  }
  raw_output_contents: "\002\000\000\000"
  raw_output_contents: "\000\000\000\000"
}

infer_response {
  model_name: "repeat_int32"
  model_version: "1"
  id: "0"
  outputs {
    name: "IDX"
    datatype: "UINT32"
    shape: 1
  }
  outputs {
    name: "OUT"
    datatype: "INT32"
    shape: 1
  }
  raw_output_contents: "\003\000\000\000"
  raw_output_contents: "\001\000\000\000"
}

PASS: repeat_int32
stream stopped...

```

Look how a single request generated 4 responses.

## Running inference on Square model:

Send inference requests to square model using [square_client.py](square_client.py).

```
python3 examples/decoupled/square_client.py
```

You should see an output similar to the output below:

```
stream started...
async_stream_infer
model_name: "square_int32"
id: "0"
inputs {
  name: "IN"
  datatype: "INT32"
  shape: 1
}
outputs {
  name: "OUT"
}
raw_input_contents: "\004\000\000\000"

enqueued request 0 to stream...
async_stream_infer
model_name: "square_int32"
id: "1"
inputs {
  name: "IN"
  datatype: "INT32"
  shape: 1
}
outputs {
  name: "OUT"
}
raw_input_contents: "\002\000\000\000"

enqueued request 1 to stream...
async_stream_infer
model_name: "square_int32"
id: "2"
inputs {
  name: "IN"
  datatype: "INT32"
  shape: 1
}
outputs {
  name: "OUT"
}
raw_input_contents: "\000\000\000\000"

enqueued request 2 to stream...
async_stream_infer
model_name: "square_int32"
id: "3"
inputs {
  name: "IN"
  datatype: "INT32"
  shape: 1
}
outputs {
  name: "OUT"
}
raw_input_contents: "\001\000\000\000"

enqueued request 3 to stream...
infer_response {
  model_name: "square_int32"
  model_version: "1"
  id: "0"
  outputs {
    name: "OUT"
    datatype: "INT32"
    shape: 1
  }
  raw_output_contents: "\004\000\000\000"
}

infer_response {
  model_name: "square_int32"
  model_version: "1"
  id: "1"
  outputs {
    name: "OUT"
    datatype: "INT32"
    shape: 1
  }
  raw_output_contents: "\002\000\000\000"
}

infer_response {
  model_name: "square_int32"
  model_version: "1"
  id: "0"
  outputs {
    name: "OUT"
    datatype: "INT32"
    shape: 1
  }
  raw_output_contents: "\004\000\000\000"
}

infer_response {
  model_name: "square_int32"
  model_version: "1"
  id: "3"
  outputs {
    name: "OUT"
    datatype: "INT32"
    shape: 1
  }
  raw_output_contents: "\001\000\000\000"
}

infer_response {
  model_name: "square_int32"
  model_version: "1"
  id: "1"
  outputs {
    name: "OUT"
    datatype: "INT32"
    shape: 1
  }
  raw_output_contents: "\002\000\000\000"
}

infer_response {
  model_name: "square_int32"
  model_version: "1"
  id: "0"
  outputs {
    name: "OUT"
    datatype: "INT32"
    shape: 1
  }
  raw_output_contents: "\004\000\000\000"
}

infer_response {
  model_name: "square_int32"
  model_version: "1"
  id: "0"
  outputs {
    name: "OUT"
    datatype: "INT32"
    shape: 1
  }
  raw_output_contents: "\004\000\000\000"
}

PASS: square_int32
stream stopped...

```

Look how responses were delivered out-of-order of requests.
The generated responses can be tracked to their request using
the `id` field.
