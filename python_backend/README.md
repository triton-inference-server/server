<!--
# Copyright 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Python Backend

The Triton backend for Python. The goal of Python backend is to let you serve
models written in Python by Triton Inference Server without having to write
any C++ code.

## User Documentation

- [Python Backend](#python-backend)
  - [User Documentation](#user-documentation)
  - [Quick Start](#quick-start)
  - [Building from Source](#building-from-source)
  - [Usage](#usage)
    - [`auto_complete_config`](#auto_complete_config)
    - [`initialize`](#initialize)
    - [`execute`](#execute)
      - [Default Mode](#default-mode)
      - [Error Handling](#error-handling)
      - [Request Cancellation Handling](#request-cancellation-handling)
      - [Decoupled mode](#decoupled-mode)
        - [Use Cases](#use-cases)
        - [Async Execute](#async-execute)
      - [Request Rescheduling](#request-rescheduling)
    - [`finalize`](#finalize)
  - [Model Config File](#model-config-file)
  - [Inference Request Parameters](#inference-request-parameters)
  - [Inference Response Parameters](#inference-response-parameters)
  - [Managing Python Runtime and Libraries](#managing-python-runtime-and-libraries)
    - [Building Custom Python Backend Stub](#building-custom-python-backend-stub)
    - [Creating Custom Execution Environments](#creating-custom-execution-environments)
    - [Important Notes](#important-notes)
  - [Error Handling](#error-handling-1)
  - [Managing Shared Memory](#managing-shared-memory)
  - [Multiple Model Instance Support](#multiple-model-instance-support)
  - [Running Multiple Instances of Triton Server](#running-multiple-instances-of-triton-server)
- [Business Logic Scripting](#business-logic-scripting)
  - [Using BLS with Decoupled Models](#using-bls-with-decoupled-models)
  - [Model Loading API](#model-loading-api)
  - [Using BLS with Stateful Models](#using-bls-with-stateful-models)
  - [Limitation](#limitation)
- [Interoperability and GPU Support](#interoperability-and-gpu-support)
  - [`pb_utils.Tensor.to_dlpack() -> PyCapsule`](#pb_utilstensorto_dlpack---pycapsule)
  - [`pb_utils.Tensor.from_dlpack() -> Tensor`](#pb_utilstensorfrom_dlpack---tensor)
  - [`pb_utils.Tensor.is_cpu() -> bool`](#pb_utilstensoris_cpu---bool)
  - [Input Tensor Device Placement](#input-tensor-device-placement)
- [Frameworks](#frameworks)
  - [PyTorch](#pytorch)
    - [PyTorch Determinism](#pytorch-determinism)
  - [TensorFlow](#tensorflow)
    - [TensorFlow Determinism](#tensorflow-determinism)
- [Custom Metrics](#custom-metrics)
- [Examples](#examples)
  - [AddSub in NumPy](#addsub-in-numpy)
  - [AddSubNet in PyTorch](#addsubnet-in-pytorch)
  - [AddSub in JAX](#addsub-in-jax)
  - [Business Logic Scripting](#business-logic-scripting-1)
  - [Preprocessing](#preprocessing)
  - [Decoupled Models](#decoupled-models)
  - [Model Instance Kind](#model-instance-kind)
  - [Auto-complete config](#auto-complete-config)
  - [Custom Metrics](#custom-metrics-1)
- [Running with Inferentia](#running-with-inferentia)
- [Logging](#logging)
- [Development with VSCode](#development-with-vscode)
- [Reporting problems, asking questions](#reporting-problems-asking-questions)

## Quick Start

1. Run the Triton Inference Server container.
```
docker run --shm-size=1g --ulimit memlock=-1 -p 8000:8000 -p 8001:8001 -p 8002:8002 --ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:<xx.yy>-py3
```
Replace \<xx.yy\> with the Triton version (e.g. 21.05).

2. Inside the container, clone the Python backend repository.

```
git clone https://github.com/triton-inference-server/python_backend -b r<xx.yy>
```

3. Install example model.
```
cd python_backend
mkdir -p models/add_sub/1/
cp examples/add_sub/model.py models/add_sub/1/model.py
cp examples/add_sub/config.pbtxt models/add_sub/config.pbtxt
```

4. Start the Triton server.

```
tritonserver --model-repository `pwd`/models
```

5. In the host machine, start the client container.

```
docker run -ti --net host nvcr.io/nvidia/tritonserver:<xx.yy>-py3-sdk /bin/bash
```

6. In the client container, clone the Python backend repository.

```
git clone https://github.com/triton-inference-server/python_backend -b r<xx.yy>
```

7. Run the example client.
```
python3 python_backend/examples/add_sub/client.py
```

## Building from Source

1. Requirements

* cmake >= 3.17
* numpy
* rapidjson-dev
* libarchive-dev
* zlib1g-dev

```
pip3 install numpy
```

On Ubuntu or Debian you can use the command below to install `rapidjson`, `libarchive`, and `zlib`:
```
sudo apt-get install rapidjson-dev libarchive-dev zlib1g-dev
```

2. Build Python backend. Replace \<GIT\_BRANCH\_NAME\> with the GitHub branch
   that you want to compile. For release branches it should be r\<xx.yy\> (e.g.
   r21.06).

```
mkdir build
cd build
cmake -DTRITON_ENABLE_GPU=ON -DTRITON_BACKEND_REPO_TAG=<GIT_BRANCH_NAME> -DTRITON_COMMON_REPO_TAG=<GIT_BRANCH_NAME> -DTRITON_CORE_REPO_TAG=<GIT_BRANCH_NAME> -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
make install
```

The following required Triton repositories will be pulled and used in
the build. If the CMake variables below are not specified, "main" branch
of those repositories will be used. \<GIT\_BRANCH\_NAME\> should be the same
as the Python backend repository branch that you are trying to compile.

* triton-inference-server/backend: `-DTRITON_BACKEND_REPO_TAG=<GIT_BRANCH_NAME>`
* triton-inference-server/common: `-DTRITON_COMMON_REPO_TAG=<GIT_BRANCH_NAME>`
* triton-inference-server/core: `-DTRITON_CORE_REPO_TAG=<GIT_BRANCH_NAME>`


Set `-DCMAKE_INSTALL_PREFIX` to the location where the Triton Server is
installed. In the released containers, this location is `/opt/tritonserver`.

3. Copy example model and configuration

```
mkdir -p models/add_sub/1/
cp examples/add_sub/model.py models/add_sub/1/model.py
cp examples/add_sub/config.pbtxt models/add_sub/config.pbtxt
```

4. Start the Triton Server

```
/opt/tritonserver/bin/tritonserver --model-repository=`pwd`/models
```

5. Use the client app to perform inference

```
python3 examples/add_sub/client.py
```

## Usage

In order to use the Python backend, you need to create a Python file that
has a structure similar to below:

```python
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        """`auto_complete_config` is called only once when loading the model
        assuming the server was not started with
        `--disable-auto-complete-config`. Implementing this function is
        optional. No implementation of `auto_complete_config` will do nothing.
        This function can be used to set `max_batch_size`, `input` and `output`
        properties of the model using `set_max_batch_size`, `add_input`, and
        `add_output`. These properties will allow Triton to load the model with
        minimal model configuration in absence of a configuration file. This
        function returns the `pb_utils.ModelConfig` object with these
        properties. You can use the `as_dict` function to gain read-only access
        to the `pb_utils.ModelConfig` object. The `pb_utils.ModelConfig` object
        being returned from here will be used as the final configuration for
        the model.

        Note: The Python interpreter used to invoke this function will be
        destroyed upon returning from this function and as a result none of the
        objects created here will be available in the `initialize`, `execute`,
        or `finalize` functions.

        Parameters
        ----------
        auto_complete_model_config : pb_utils.ModelConfig
          An object containing the existing model configuration. You can build
          upon the configuration given by this object when setting the
          properties for this model.

        Returns
        -------
        pb_utils.ModelConfig
          An object containing the auto-completed model configuration
        """
        inputs = [{
            'name': 'INPUT0',
            'data_type': 'TYPE_FP32',
            'dims': [4],
            # this parameter will set `INPUT0 as an optional input`
            'optional': True
        }, {
            'name': 'INPUT1',
            'data_type': 'TYPE_FP32',
            'dims': [4]
        }]
        outputs = [{
            'name': 'OUTPUT0',
            'data_type': 'TYPE_FP32',
            'dims': [4]
        }, {
            'name': 'OUTPUT1',
            'data_type': 'TYPE_FP32',
            'dims': [4]
        }]

        # Demonstrate the usage of `as_dict`, `add_input`, `add_output`,
        # `set_max_batch_size`, and `set_dynamic_batching` functions.
        # Store the model configuration as a dictionary.
        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config['input']:
            input_names.append(input['name'])
        for output in config['output']:
            output_names.append(output['name'])

        for input in inputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_input` will check for conflicts and
            # raise errors if an input with the same name already exists in
            # the configuration but has different data_type or dims property.
            if input['name'] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_output` will check for conflicts and
            # raise errors if an output with the same name already exists in
            # the configuration but has different data_type or dims property.
            if output['name'] not in output_names:
                auto_complete_model_config.add_output(output)

        auto_complete_model_config.set_max_batch_size(0)

        # To enable a dynamic batcher with default settings, you can use
        # auto_complete_model_config set_dynamic_batching() function. It is
        # commented in this example because the max_batch_size is zero.
        #
        # auto_complete_model_config.set_dynamic_batching()

        return auto_complete_model_config

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device
            ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        print('Initialized...')

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them.
        # Reusing the same pb_utils.InferenceResponse object for multiple
        # requests may result in segmentation faults. You should avoid storing
        # any of the input Tensors in the class attributes as they will be
        # overridden in subsequent inference requests. You can make a copy of
        # the underlying NumPy array and store it if it is required.
        for request in requests:
            # Perform inference on the request and append it to responses
            # list...

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

```

Every Python backend can implement four main functions:

### `auto_complete_config`

`auto_complete_config` is called only once when loading the model assuming
the server was not started with
[`--disable-auto-complete-config`](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#auto-generated-model-configuration).

Implementing this function is optional. No implementation of
`auto_complete_config` will do nothing. This function can be used to set
[`max_batch_size`](
  https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#maximum-batch-size),
[dynamic_batching](
  https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher),
[`input`](
  https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#inputs-and-outputs)
and
[`output`](
  https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#inputs-and-outputs)
properties of the model using `set_max_batch_size`, `set_dynamic_batching`,
`add_input`, and `add_output`. These properties will allow Triton to load the
model with
[minimal model configuration](
  https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#minimal-model-configuration)
in absence of a configuration file. This function returns the
`pb_utils.ModelConfig` object with these properties. You can use the `as_dict`
function to gain read-only access to the `pb_utils.ModelConfig` object.
The `pb_utils.ModelConfig` object being returned from here will be used as the
final configuration for the model.

In addition to minimal properties, you can also set [model_transaction_policy](
  https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#model-transaction-policy)
through `auto_complete_config` using `set_model_transaction_policy`.
For example,
```python
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
      ...
      transaction_policy = {"decoupled": True}
      auto_complete_model_config.set_model_transaction_policy(transaction_policy)
      ...
```

Note: The Python interpreter used to invoke this function will be destroyed
upon returning from this function and as a result none of the objects
created here will be available in the `initialize`, `execute`, or `finalize`
functions.

### `initialize`

`initialize` is called once the model is being loaded. Implementing
`initialize` is optional. `initialize` allows you to do any necessary
initializations before execution. In the `initialize` function, you are given
an `args` variable. `args` is a Python dictionary. Both keys and
values for this Python dictionary are strings. You can find the available
keys in the `args` dictionary along with their description in the table
below:

| key                      | description                                      |
| ------------------------ | ------------------------------------------------ |
| model_config             | A JSON string containing the model configuration |
| model_instance_kind      | A string containing model instance kind          |
| model_instance_device_id | A string containing model instance device ID     |
| model_repository         | Model repository path                            |
| model_version            | Model version                                    |
| model_name               | Model name                                       |

### `execute`

`execute` function is called whenever an inference request is made. Every
Python model must implement `execute` function. In the `execute` function you
are given a list of `InferenceRequest` objects. There are two modes of
implementing this function. The mode you choose should depend on your use case.
That is whether or not you want to return decoupled responses from this model
or not.

#### Default Mode

This is the most generic way you would like to implement your model and
requires the `execute` function to return exactly one response per request.
This entails that in this mode, your `execute` function must return a list of
`InferenceResponse` objects that has the same length as `requests`. The work
flow in this mode is:

* `execute` function receives a batch of pb_utils.InferenceRequest as a
  length N array.

* Perform inference on the pb_utils.InferenceRequest and append the
  corresponding pb_utils.InferenceResponse to a response list.

* Return back the response list.

  * The length of response list being returned must be N.

  * Each element in the list should be the response for the corresponding
    element in the request array.

  * Each element must contain a response (a response can be either output
    tensors or an error); an element cannot be None.


Triton checks to ensure that these requirements on response list are
satisfied and if not returns an error response for all inference requests.
Upon return from the execute function all tensor data associated with the
InferenceRequest objects passed to the function are deleted, and so
InferenceRequest objects should not be retained by the Python model.

Starting from 24.06, models may choose to send the response using the
`InferenceResponseSender` as illustrated on [Decoupled mode](#decoupled-mode).
Since the model is in default mode, it must send exactly one response per
request. The `pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL` flag must be sent
either with the response or as a flag only response afterward.

#### Error Handling

In case one of the requests has an error, you can use the `TritonError` object
to set the error message for that specific request. Below is an example of
setting errors for an `InferenceResponse` object:

```python
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    ...

    def execute(self, requests):
        responses = []

        for request in requests:
            if an_error_occurred:
              # If there is an error, there is no need to pass the
              # "output_tensors" to the InferenceResponse. The "output_tensors"
              # that are passed in this case will be ignored.
              responses.append(pb_utils.InferenceResponse(
                error=pb_utils.TritonError("An Error Occurred")))

        return responses
```

Starting from 23.09, `pb_utils.TritonError` may be constructed with an optional
Triton error code on the second parameter. For example:

```python
pb_utils.TritonError("The file is not found", pb_utils.TritonError.NOT_FOUND)
```

If no code is specified, `pb_utils.TritonError.INTERNAL` will be used by default.

Supported error codes:
* `pb_utils.TritonError.UNKNOWN`
* `pb_utils.TritonError.INTERNAL`
* `pb_utils.TritonError.NOT_FOUND`
* `pb_utils.TritonError.INVALID_ARG`
* `pb_utils.TritonError.UNAVAILABLE`
* `pb_utils.TritonError.UNSUPPORTED`
* `pb_utils.TritonError.ALREADY_EXISTS`
* `pb_utils.TritonError.CANCELLED` (since 23.10)

#### Request Cancellation Handling

One or more requests may be cancelled by the client during execution. Starting
from 23.10, `request.is_cancelled()` returns whether the request is cancelled or
not. For example:

```python
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    ...

    def execute(self, requests):
        responses = []

        for request in requests:
            if request.is_cancelled():
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError("Message", pb_utils.TritonError.CANCELLED)))
            else:
                ...

        return responses
```

Although checking for request cancellation is optional, it is recommended to
check for cancellation at strategic request execution stages that can early
terminate the execution in the event of its response is no longer needed.

#### Decoupled mode

This mode allows user to send multiple responses for a request or
not send any responses for a request. A model may also send
responses out-of-order relative to the order that the request batches
are executed. Such models are called *decoupled* models. In
order to use this mode, the
[transaction policy](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#model-transaction-policy)
in the model configuration must be set to decoupled.


In decoupled mode, model must use `InferenceResponseSender` object per
request to keep creating and sending any number of responses for the
request. The workflow in this mode may look like:

* `execute` function receives a batch of pb_utils.InferenceRequest as a
  length N array.

* Iterate through each pb_utils.InferenceRequest and perform for the following
  steps for each pb_utils.InferenceRequest object:

  1. Get `InferenceResponseSender` object for the InferenceRequest using
     InferenceRequest.get_response_sender().

  2. Create and populate pb_utils.InferenceResponse to be sent back.

  3. Use InferenceResponseSender.send() to send the above response. If
     this is the last request then pass
     pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL as a flag with
     InferenceResponseSender.send(). Otherwise continue with Step 1 for sending
     next request.

* The return value for `execute` function in this mode should be None.

Similar to above, in case one of the requests has an error, you can use
the `TritonError` object to set the error message for that specific
request. After setting errors for an pb_utils.InferenceResponse
object, use InferenceResponseSender.send() to send response with the
error back to the user.

Starting from 23.10, request cancellation can be checked directly on the
`InferenceResponseSender` object using `response_sender.is_cancelled()`. Sending
the TRITONSERVER_RESPONSE_COMPLETE_FINAL flag at the end of response is still
needed even the request is cancelled.

##### Use Cases

The decoupled mode is powerful and supports various other use cases:

* If the model should not send any response for the request,
  then call InferenceResponseSender.send() with no response
  but flag parameter set to pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL.

* The model can also send responses out-of-order in which it received
  requests.

* The request data and `InferenceResponseSender` object can be passed to
  a separate thread in the model. This means main caller thread can exit
  from `execute` function and the model can still continue generating
  responses as long as it holds `InferenceResponseSender` object.


The [decoupled examples](examples/decoupled/README.md) demonstrate
full power of what can be achieved from decoupled API. Read
[Decoupled Backends and Models](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/decoupled_models.md)
for more details on how to host a decoupled model.

##### Async Execute

Starting from 24.04, `async def execute(self, requests):` is supported for
decoupled Python models. Its coroutine will be executed by an AsyncIO event loop
shared with requests executing in the same model instance. The next request for
the model instance can start executing while the current request is waiting.

This is useful for minimizing the number of model instances for models that
spend the majority of its time waiting, given requests can be executed
concurrently by AsyncIO. To take full advantage of the concurrency, it is vital
for the async execute function to not block the event loop from making progress
while it is waiting, i.e. downloading over the network.

Notes:
* The model should not modify the running event loop, as this might cause
unexpected issues.
* The server/backend do not control how many requests are added to the event
loop by a model instance.

#### Request Rescheduling

Starting from 23.11, Python backend supports request rescheduling. By calling
the `set_release_flags` function on the request object with the flag
`pb_utils.TRITONSERVER_REQUEST_RELEASE_RESCHEDULE`, you can reschedule the
request for further execution in a future batch. This feature is useful for
handling iterative sequences.

The model config must be configured to enable iterative sequence batching in
order to use the request rescheduling API:

```
sequence_batching {
  iterative_sequence : true
}
```

For non-decoupled models, there can only be one response for each request. Since
the rescheduled request is the same as the original, you must append a `None`
object to the response list for the rescheduled request. For example:

```python
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    ...

    def execute(self, requests):
        responses = []

        for request in requests:
            # Explicitly reschedule the first request
            if self.idx == 0:
                request.set_release_flags(
                    pb_utils.TRITONSERVER_REQUEST_RELEASE_RESCHEDULE
                )
                responses.append(None)
                self.idx += 1
            else:
                responses.append(inference_response)

        return responses
```

For decoupled models, it is required to reschedule a request *before* returning
from the `execute` function.
Below is an example of a decoupled model using request rescheduling. This model
takes 1 input tensor, an INT32 [ 1 ] input named "IN", and produces an output
tensor "OUT" with the same shape as the input tensor. The input value indicates
the total number of responses to be generated and the output value indicates the
number of remaining responses. For example, if the request input has value 2,
the model will:
  - Send a response with value 1.
  - Release request with RESCHEDULE flag.
  - When execute on the same request, send the last response with value 0.
  - Release request with ALL flag.

```python
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    ...

    def execute(self, requests):
        responses = []

        for request in requests:
            in_input = pb_utils.get_input_tensor_by_name(request, "IN").as_numpy()

            if self.reset_flag:
                self.remaining_response = in_input[0]
                self.reset_flag = False

            response_sender = request.get_response_sender()

            self.remaining_response -= 1

            out_output = pb_utils.Tensor(
                "OUT", np.array([self.remaining_response], np.int32)
            )
            response = pb_utils.InferenceResponse(output_tensors=[out_output])

            if self.remaining_response <= 0:
                response_sender.send(
                    response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
                self.reset_flag = True
            else:
                request.set_release_flags(
                    pb_utils.TRITONSERVER_REQUEST_RELEASE_RESCHEDULE
                )
                response_sender.send(response)

        return None
```

### `finalize`

Implementing `finalize` is optional. This function allows you to do any clean
ups necessary before the model is unloaded from Triton server.

You can look at the [add_sub example](examples/add_sub/model.py) which contains
a complete example of implementing all these functions for a Python model
that adds and subtracts the inputs given to it. After implementing all the
necessary functions, you should save this file as `model.py`.

## Model Config File

Every Python Triton model must provide a `config.pbtxt` file describing
the model configuration. In order to use this backend you must set the
`backend` field of your model `config.pbtxt` file to `python`. You
shouldn't set `platform` field of the configuration.

Your models directory should look like below:
```
models
└── add_sub
    ├── 1
    │   └── model.py
    └── config.pbtxt
```

## Inference Request Parameters

You can retrieve the parameters associated with an inference request
using the `inference_request.parameters()` function. This function
returns a JSON string where the keys are the keys of the parameters
object and the values are the values for the parameters field. Note that
you need to parse this string using `json.loads` to convert it to a dictionary.

Starting from 23.11 release, parameters may be provided to the `InferenceRequest`
object during construction. The parameters should be a dictionary of key value
pairs, where keys are `str` and values are `bool`, `int` or `str`.
```python
request = pb_utils.InferenceRequest(parameters={"key": "value"}, ...)
```

You can read more about the inference request parameters in the [parameters
extension](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_parameters.md)
documentation.

## Inference Response Parameters

Inference response parameters may be optionally set during the construction of
an inference response object. The parameters should be a dictionary of key value
pairs, where keys are `str` and values are `bool`, `int` or `str`. For example,
```python
response = pb_utils.InferenceResponse(
    output_tensors, parameters={"key": "value"}
)
```

You can read more about the inference response parameters in the [parameters
extension](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_parameters.md)
documentation.

The parameters associated with an inference response can be retrieved using the
`inference_response.parameters()` function. This function returns a JSON string
where the keys are the keys of the parameters object and the values are the
values for the parameters field. Note that you need to parse this string using
`json.loads` to convert it to a dictionary.

## Managing Python Runtime and Libraries

Python backend shipped in the [NVIDIA GPU Cloud](https://ngc.nvidia.com/)
containers uses Python 3.10. Python backend is able to use the libraries
that exist in the current Python environment. These libraries can
be installed in a virtualenv, conda environment, or the global system
Python. These libraries will only be used if the Python version matches
the Python version of the Python backend's stub executable. For example,
if you install a set of libraries in a Python 3.9 environment and your
Python backend stub is compiled with Python 3.10 these libraries will NOT
be available in your Python model served using Triton. You would need to
compile the stub executable with Python 3.9 using the instructions in
[Building Custom Python Backend Stub](#building-custom-python-backend-stub)
section.

### Building Custom Python Backend Stub

**Important Note: You only need to compile a custom Python backend stub if the
Python version is different from Python 3.10 which is shipped by
default in the Triton containers.**

Python backend uses a *stub* process to connect your `model.py` file to the
Triton C++ core. This stub process dynamically links to a specific
`libpython<X>.<Y>.so` version. If you intend to use a Python interpreter with
different version from the default Python backend stub, you need to compile
your own Python backend stub by following the steps below:

1. Install the software packages below:
* [cmake](https://cmake.org)
* rapidjson and libarchive (instructions for installing these packages in
Ubuntu or Debian are included in
[Building from Source Section](#building-from-source))

2. Make sure that the expected Python version is available in your environment.

If you are using `conda`, you should make sure to activate the environment by
`conda activate <conda-env-name>`. Note that you don't have to use `conda` and
can install Python however you wish. Python backend relies on
[pybind11](https://github.com/pybind/pybind11) to find the correct Python
version. If you noticed that the correct Python version is not picked up, you
can read more on how
[pybind11 decides which Python to use](https://pybind11.readthedocs.io/en/stable/faq.html?highlight=cmake#cmake-doesn-t-detect-the-right-python-version).

3. Clone the Python backend repository and compile the Python backend stub
   (replace \<GIT\_BRANCH\_NAME\> with the branch name that you want to use,
   for release branches it should be r\<xx.yy\>):
```bash
git clone https://github.com/triton-inference-server/python_backend -b
<GIT_BRANCH_NAME>
cd python_backend
mkdir build && cd build
cmake -DTRITON_ENABLE_GPU=ON -DTRITON_BACKEND_REPO_TAG=<GIT_BRANCH_NAME> -DTRITON_COMMON_REPO_TAG=<GIT_BRANCH_NAME> -DTRITON_CORE_REPO_TAG=<GIT_BRANCH_NAME> -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
make triton-python-backend-stub
```

Now, you have access to a Python backend stub with your Python version. You can verify
that using `ldd`:

```
ldd triton_python_backend_stub
...
libpython3.6m.so.1.0 => /home/ubuntu/envs/miniconda3/envs/python-3-6/lib/libpython3.6m.so.1.0 (0x00007fbb69cf3000)
...
```

There are many other shared libraries printed in addition to the library posted
above. However, it is important to see `libpython<major>.<minor>m.so.1.0` in the
list of linked shared libraries. If you use a different Python version, you
should see that version instead. You need to copy the
`triton_python_backend_stub` to the model directory of the models that want to
use the custom Python backend
stub. For example, if you have `model_a` in your
[model repository](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md),
the folder structure should look like below:

```
models
|-- model_a
    |-- 1
    |   |-- model.py
    |-- config.pbtxt
    `-- triton_python_backend_stub
```

Note the location of `triton_python_backend_stub` in the directory structure
above.

### Creating Custom Execution Environments

If you want to create a tar file that contains all your Python dependencies or
you want to use different Python environments for each Python model you need to
create a *Custom Execution Environment* in Python backend.
Currently, Python backend supports
[conda-pack](https://conda.github.io/conda-pack/) for this purpose.
[conda-pack](https://conda.github.io/conda-pack/) ensures that your conda
environment is portable. You can create a tar file for your conda environment
using `conda-pack` command:

```
conda-pack
Collecting packages...
Packing environment at '/home/iman/miniconda3/envs/python-3-6' to 'python-3-6.tar.gz'
[########################################] | 100% Completed |  4.5s
```

**Important Note:** Before installing the packages in your conda environment,
make sure that you have exported
[`PYTHONNOUSERSITE`](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONNOUSERSITE)
environment variable:

```
export PYTHONNOUSERSITE=True
```

If this variable is not exported and similar packages are installed outside your
conda environment, your tar file may not contain all the dependencies required
for an isolated Python environment.

Alternatively, Python backend also supports unpacked conda execution
environments, given it points to an activation script to setup the conda
environment. To do this, the execution environment can be first packed using
[conda-pack](https://conda.github.io/conda-pack/) and then unpacked, or created
using [conda create -p](https://docs.conda.io/projects/conda/en/latest/commands/create.html).
In this case, the conda activation script is located in:
```$path_to_conda_pack/lib/python<your.python.version>/site-packages/conda_pack/scripts/posix/activate```
This speeds up the server loading time for models.

After creating the packed file from the conda environment or creating a conda
environment with a custom activation script, you need to tell Python
backend to use that environment for your model. You can do this by adding the
lines below to the `config.pbtxt` file:

```
name: "model_a"
backend: "python"

...

parameters: {
  key: "EXECUTION_ENV_PATH",
  value: {string_value: "/home/iman/miniconda3/envs/python-3-6/python3.6.tar.gz"}
}
```

It is also possible to provide the execution environment path relative to the
model folder in model repository:

```
name: "model_a"
backend: "python"

...

parameters: {
  key: "EXECUTION_ENV_PATH",
  value: {string_value: "$$TRITON_MODEL_DIRECTORY/python3.6.tar.gz"}
}
```

In this case, `python3.tar.gz` should be placed in the model folder and the
model repository should look like below:

```
models
|-- model_a
|   |-- 1
|   |   `-- model.py
|   |-- config.pbtxt
|   |-- python3.6.tar.gz
|   `-- triton_python_backend_stub
```

In the example above, `$$TRITON_MODEL_DIRECTORY` is resolved to
`$pwd/models/model_a`.

To accelerate the loading time of `model_a`, you can follow the steps below to
unpack the conda environment in the model folder:

```bash
mkdir -p $pwd/models/model_a/python3.6
tar -xvf $pwd/models/model_a/python3.6.tar.gz -C $pwd/models/model_a/python3.6
```

Then you can change the `EXECUTION_ENV_PATH` to point to the unpacked directory:

```
parameters: {
  key: "EXECUTION_ENV_PATH",
  value: {string_value: "$$TRITON_MODEL_DIRECTORY/python3.6"}
}
```

This is useful if you want to use S3, GCS, or Azure and you do not have access
to the absolute path of the execution env that is stored in the cloud object
storage service.

### Important Notes

1. The version of the Python interpreter in the execution environment must match
the version of triton_python_backend_stub.

2. If you don't want to use a different Python interpreter, you can skip
[Building Custom Python Backend Stub](#building-custom-python-backend-stub).
In this case you only need to pack your environment using `conda-pack` and
provide the path to tar file in the model config. However, the previous note
still applies here and the version of the Python interpreter inside the conda
environment must match the Python version of stub used by Python backend. The
default version of the stub is Python 3.10.

3. You can share a single execution environment across multiple models. You
need to provide the path to the tar file in the `EXECUTION_ENV_PATH` in the
`config.pbtxt` of all the models that want to use the execution environment.

4. If `$$TRITON_MODEL_DIRECTORY` is used in the `EXECUTION_ENV_PATH`, the final
`EXECUTION_ENV_PATH` **must not** escape from the `$$TRITON_MODEL_DIRECTORY`,
as the behavior of accessing anywhere outside the `$$TRITON_MODEL_DIRECTORY` is
**undefined**.

5. If a non-`$$TRITON_MODEL_DIRECTORY` `EXECUTION_ENV_PATH` is used, only local
file system paths are currently supported. The behavior of using cloud paths is
**undefined**.

6. If you need to compile the Python backend stub, it is recommended that you
compile it in the official Triton NGC containers. Otherwise, your compiled stub
may use dependencies that are not available in the Triton container that you are
using for deployment. For example, compiling the Python backend stub on an OS
other than Ubuntu 22.04 can lead to unexpected errors.

7. If you encounter the "GLIBCXX_3.4.30 not found" error during runtime, we
recommend upgrading your conda version and installing `libstdcxx-ng=12` by
running `conda install -c conda-forge libstdcxx-ng=12 -y`. If this solution does
not resolve the issue, please feel free to open an issue on the
[GitHub issue page](https://github.com/triton-inference-server/server/issues)
following the provided
[instructions](https://github.com/triton-inference-server/server#reporting-problems-asking-questions).


## Error Handling

If there is an error that affects the `initialize`, `execute`, or `finalize`
function of the Python model you can use `TritonInferenceException`.
Example below shows how you can do error handling in `finalize`:

```python
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    ...

    def finalize(self):
      if error_during_finalize:
        raise pb_utils.TritonModelException(
          "An error occurred during finalize.")
```

## Managing Shared Memory

Starting from 21.04 release, Python backend uses shared memory to connect
user's code to Triton. Note that this change is completely transparent and
does not require any change to the existing user's model code.

Python backend, by default, allocates 1 MB for each model instance. Then,
it will grow the shared memory region by 1 MB chunks whenever an increase is
required. You can configure the default shared memory used by each model
instance using the `shm-default-byte-size` flag. The amount of shared memory
growth can be configured using the `shm-growth-byte-size`.

You can also configure the timeout used for connecting Triton main process
to the Python backend stubs using the `stub-timeout-seconds`. The default
value is 30 seconds.

The config values described above can be passed to Triton using
`--backend-config` flag:

```
/opt/tritonserver/bin/tritonserver --model-repository=`pwd`/models --backend-config=python,<config-key>=<config-value>
```

Also, if you are running Triton inside a Docker container you need to
properly set the `--shm-size` flag depending on the size of your inputs and
outputs. The default value for docker run command is `64MB` which is very
small.

## Multiple Model Instance Support

Python interpreter uses a global lock known as
[GIL](https://docs.python.org/3/c-api/init.html#thread-state-and-the-global-interpreter-lock).
Because of GIL, it is not possible have multiple threads running in the same
Python interpreter simultaneously as each thread requires to acquire the GIL
when accessing Python objects which will serialize all the operations. In order
to work around this issue, Python backend spawns a separate process for each
[model instance](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#multiple-model-instances).
This is in contrast with how other Triton backends such as
[ONNXRuntime](https://github.com/triton-inference-server/onnxruntime_backend),
[TensorFlow](https://github.com/triton-inference-server/tensorflow_backend),
and [PyTorch](https://github.com/triton-inference-server/pytorch_backend)
handle multiple instances. Increasing the instance count for these backends
will create additional threads instead of spawning separate processes.

## Running Multiple Instances of Triton Server

Starting from 24.04 release, Python backend uses UUID to generate unique
names for Python backend shared memory regions so that multiple instances of
the server can run at the same time without any conflicts.

If you're using a Python backend released before the 24.04 release, you need
to specify different `shm-region-prefix-name` using the `--backend-config` flag
to avoid conflicts between the shared memory regions. For example:

```
# Triton instance 1
tritonserver --model-repository=/models --backend-config=python,shm-region-prefix-name=prefix1

# Triton instance 2
tritonserver --model-repository=/models --backend-config=python,shm-region-prefix-name=prefix2
```

Note that the hangs would only occur if the `/dev/shm` is shared between
the two instances of the server. If you run the servers in different containers that
don't share this location, you don't need to specify `shm-region-prefix-name`.

# Business Logic Scripting

Triton's
[ensemble](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#ensemble-models)
feature supports many use cases where multiple models are composed into a
pipeline (or more generally a DAG, directed acyclic graph). However, there are
many other use cases that are not supported because as part of the model
pipeline they require loops, conditionals (if-then-else), data-dependent
control-flow and other custom logic to be intermixed with model execution. We
call this combination of custom logic and model executions *Business Logic
Scripting (BLS)*.

Starting from 21.08, you can implement BLS in your Python model. A new set of
utility functions allows you to execute inference requests on other models
being served by Triton as a part of executing your Python model. Note that BLS
should only be used inside the `execute` function and is not supported
in the `initialize` or `finalize` methods. Example below shows how to use this
feature:

```python
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
  ...
    def execute(self, requests):
      ...
      # Create an InferenceRequest object. `model_name`,
      # `requested_output_names`, and `inputs` are the required arguments and
      # must be provided when constructing an InferenceRequest object. Make
      # sure to replace `inputs` argument with a list of `pb_utils.Tensor`
      # objects.
      inference_request = pb_utils.InferenceRequest(
          model_name='model_name',
          requested_output_names=['REQUESTED_OUTPUT_1', 'REQUESTED_OUTPUT_2'],
          inputs=[<pb_utils.Tensor object>])

      # `pb_utils.InferenceRequest` supports request_id, correlation_id,
      # model version, timeout and preferred_memory in addition to the
      # arguments described above.
      # Note: Starting from the 24.03 release, the `correlation_id` parameter
      # supports both string and unsigned integer values.
      # These arguments are optional. An example containing all the arguments:
      # inference_request = pb_utils.InferenceRequest(model_name='model_name',
      #   requested_output_names=['REQUESTED_OUTPUT_1', 'REQUESTED_OUTPUT_2'],
      #   inputs=[<list of pb_utils.Tensor objects>],
      #   request_id="1", correlation_id=4, model_version=1, flags=0, timeout=5,
      #   preferred_memory=pb_utils.PreferredMemory(
      #     pb_utils.TRITONSERVER_MEMORY_GPU, # or pb_utils.TRITONSERVER_MEMORY_CPU
      #     0))

      # Execute the inference_request and wait for the response
      inference_response = inference_request.exec()

      # Check if the inference response has an error
      if inference_response.has_error():
          raise pb_utils.TritonModelException(
            inference_response.error().message())
      else:
          # Extract the output tensors from the inference response.
          output1 = pb_utils.get_output_tensor_by_name(
            inference_response, 'REQUESTED_OUTPUT_1')
          output2 = pb_utils.get_output_tensor_by_name(
            inference_response, 'REQUESTED_OUTPUT_2')

          # Decide the next steps for model execution based on the received
          # output tensors. It is possible to use the same output tensors
          # to for the final inference response too.
```


In addition to the `inference_request.exec` function that allows you to
execute blocking inference requests, `inference_request.async_exec` allows
you to perform async inference requests. This can be useful when you do not
need the result of the inference immediately. Using `async_exec` function, it
is possible to have multiple inflight inference requests and wait for the
responses only when needed. Example below shows how to use `async_exec`:

```python
import triton_python_backend_utils as pb_utils
import asyncio


class TritonPythonModel:
  ...

    # You must add the Python 'async' keyword to the beginning of `execute`
    # function if you want to use `async_exec` function.
    async def execute(self, requests):
      ...
      # Create an InferenceRequest object. `model_name`,
      # `requested_output_names`, and `inputs` are the required arguments and
      # must be provided when constructing an InferenceRequest object. Make
      # sure to replace `inputs` argument with a list of `pb_utils.Tensor`
      # objects.
      inference_request = pb_utils.InferenceRequest(
          model_name='model_name',
          requested_output_names=['REQUESTED_OUTPUT_1', 'REQUESTED_OUTPUT_2'],
          inputs=[<pb_utils.Tensor object>])

      infer_response_awaits = []
      for i in range(4):
        # async_exec function returns an
        # [Awaitable](https://docs.python.org/3/library/asyncio-task.html#awaitables)
        # object.
        infer_response_awaits.append(inference_request.async_exec())

      # Wait for all of the inference requests to complete.
      infer_responses = await asyncio.gather(*infer_response_awaits)

      for infer_response in infer_responses:
        # Check if the inference response has an error
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
              inference_response.error().message())
        else:
            # Extract the output tensors from the inference response.
            output1 = pb_utils.get_output_tensor_by_name(
              inference_response, 'REQUESTED_OUTPUT_1')
            output2 = pb_utils.get_output_tensor_by_name(
              inference_response, 'REQUESTED_OUTPUT_2')

            # Decide the next steps for model execution based on the received
            # output tensors.
```

A complete example for sync and async BLS in Python backend is included in the
[Examples](#examples) section.

## Using BLS with Decoupled Models

Starting from 23.03 release, you can execute inference requests on decoupled
models in both [default mode](#default-mode) and
[decoupled mode](#decoupled-mode). By setting the `decoupled` parameter to
`True`, the `exec` and `async_exec` function will return an
[iterator](https://docs.python.org/3/glossary.html#term-iterator) of
inference responses returned by a decoupled model. If the `decoupled` parameter
is set to `False`, the `exec` and `async_exec` function will return a single
response as shown in the example above. Besides, you can set the timeout via
the parameter 'timeout' in microseconds within the constructor of
`InferenceRequest`. If the request times out, the request will respond with an
error. The default of 'timeout' is 0 which indicates that the request has no
timeout.

Additionally, starting from the 23.04 release, you have the flexibility to
select a specific device to receive output tensors from BLS calls. This
can be achieved by setting the optional `preferred_memory` parameter within the
`InferenceRequest` constructor. To do this, you can create a `PreferredMemory`
object and specify the `preferred_memory_type` as either
`TRITONSERVER_MEMORY_GPU` or `TRITONSERVER_MEMORY_CPU`, as well as the
`preferred_device_id` as an integer to indicate the memory type and device ID
on which you wish to receive output tensors. If you do not specify the
`preferred_memory` parameter, the output tensors will be allocated on the
same device where the output tensors were received from the model to which the
BLS call is made.

Example below shows how to use this feature:

```python
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
  ...
    def execute(self, requests):
      ...
      # Create an InferenceRequest object. `model_name`,
      # `requested_output_names`, and `inputs` are the required arguments and
      # must be provided when constructing an InferenceRequest object. Make
      # sure to replace `inputs` argument with a list of `pb_utils.Tensor`
      # objects.
      inference_request = pb_utils.InferenceRequest(
          model_name='model_name',
          requested_output_names=['REQUESTED_OUTPUT_1', 'REQUESTED_OUTPUT_2'],
          inputs=[<pb_utils.Tensor object>])

      # `pb_utils.InferenceRequest` supports request_id, correlation_id,
      # model version, timeout and preferred_memory in addition to the
      # arguments described above.
      # Note: Starting from the 24.03 release, the `correlation_id` parameter
      # supports both string and unsigned integer values.
      # These arguments are optional. An example containing all the arguments:
      # inference_request = pb_utils.InferenceRequest(model_name='model_name',
      #   requested_output_names=['REQUESTED_OUTPUT_1', 'REQUESTED_OUTPUT_2'],
      #   inputs=[<list of pb_utils.Tensor objects>],
      #   request_id="1", correlation_id="ex-4", model_version=1, flags=0, timeout=5,
      #   preferred_memory=pb_utils.PreferredMemory(
      #     pb_utils.TRITONSERVER_MEMORY_GPU, # or pb_utils.TRITONSERVER_MEMORY_CPU
      #     0))

      # Execute the inference_request and wait for the response. Here we are
      # running a BLS request on a decoupled model, hence setting the parameter
      # 'decoupled' to 'True'.
      inference_responses = inference_request.exec(decoupled=True)

      for inference_response in inference_responses:
        # Check if the inference response has an error
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
              inference_response.error().message())

        # For some models, it is possible that the last response is empty
        if len(infer_response.output_tensors()) > 0:
          # Extract the output tensors from the inference response.
          output1 = pb_utils.get_output_tensor_by_name(
            inference_response, 'REQUESTED_OUTPUT_1')
          output2 = pb_utils.get_output_tensor_by_name(
            inference_response, 'REQUESTED_OUTPUT_2')

          # Decide the next steps for model execution based on the received
          # output tensors. It is possible to use the same output tensors to
          # for the final inference response too.
```


In addition to the `inference_request.exec(decoupled=True)` function that
allows you to execute blocking inference requests on decoupled models,
`inference_request.async_exec(decoupled=True)` allows you to perform async
inference requests. This can be useful when you do not need the result of the
inference immediately. Using `async_exec` function, it is possible to have
multiple inflight inference requests and wait for the responses only when
needed. Example below shows how to use `async_exec`:

```python
import triton_python_backend_utils as pb_utils
import asyncio


class TritonPythonModel:
  ...

    # You must add the Python 'async' keyword to the beginning of `execute`
    # function if you want to use `async_exec` function.
    async def execute(self, requests):
      ...
      # Create an InferenceRequest object. `model_name`,
      # `requested_output_names`, and `inputs` are the required arguments and
      # must be provided when constructing an InferenceRequest object. Make
      # sure to replace `inputs` argument with a list of `pb_utils.Tensor`
      # objects.
      inference_request = pb_utils.InferenceRequest(
          model_name='model_name',
          requested_output_names=['REQUESTED_OUTPUT_1', 'REQUESTED_OUTPUT_2'],
          inputs=[<pb_utils.Tensor object>])

      infer_response_awaits = []
      for i in range(4):
        # async_exec function returns an
        # [Awaitable](https://docs.python.org/3/library/asyncio-task.html#awaitables)
        # object.
        infer_response_awaits.append(
          inference_request.async_exec(decoupled=True))

      # Wait for all of the inference requests to complete.
      async_responses = await asyncio.gather(*infer_response_awaits)

      for infer_responses in async_responses:
        for infer_response in infer_responses:
          # Check if the inference response has an error
          if inference_response.has_error():
              raise pb_utils.TritonModelException(
                inference_response.error().message())

          # For some models, it is possible that the last response is empty
          if len(infer_response.output_tensors()) > 0:
              # Extract the output tensors from the inference response.
              output1 = pb_utils.get_output_tensor_by_name(
                inference_response, 'REQUESTED_OUTPUT_1')
              output2 = pb_utils.get_output_tensor_by_name(
                inference_response, 'REQUESTED_OUTPUT_2')

              # Decide the next steps for model execution based on the received
              # output tensors.
```

A complete example for sync and async BLS for decoupled models is included in
the [Examples](#examples) section.

Note: Async BLS is not supported on Python 3.6 or lower due to the `async`
keyword and `asyncio.run` being introduced in Python 3.7.

Starting from the 22.04 release, the lifetime of the BLS output tensors have
been improved such that if a tensor is no longer needed in your Python model it
will be automatically deallocated. This can increase the number of BLS requests
that you can execute in your model without running into the out of GPU or
shared memory error.

### Cancelling decoupled BLS requests
A decoupled BLS inference request may be cancelled by calling the `cancel()`
method on the response iterator returned from the method executing the BLS
inference request. For example,

```python
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    ...
    def execute(self, requests):
        ...
        bls_response_iterator = bls_request.exec(decoupled=True)
        ...
        bls_response_iterator.cancel()
        ...
```

You may also call the `cancel()` method on the response iterator returned from
the `async_exec()` method of the inference request. For example,

```python
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    ...
    async def execute(self, requests):
        ...
        bls_response_iterator = await bls_request.async_exec(decoupled=True)
        ...
        bls_response_iterator.cancel()
        ...
```

Note: Whether the decoupled model returns a cancellation error and stops executing
the request depends on the model's backend implementation. Please refer to the
documentation for more details [Handing in Backend](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/request_cancellation.md#handling-in-backend)

## Model Loading API

Starting from 23.07 release, you can use the model loading API to load models
required by your BLS model. The model loading API is equivalent to the Triton C
API for loading models which are documented in
[tritonserver.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h).
Below is an example of how to use the model loading API:

```python
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model_name="onnx_model"
        # Check if the model is ready, and load the model if it is not ready.
        # You can specify the model version in string format. The version is
        # optional, and if not provided, the server will choose a version based
        # on the model and internal policy.
        if not pb_utils.is_model_ready(model_name=self.model_name,
                                       model_version="1"):
            # Load the model from the model repository
            pb_utils.load_model(model_name=self.model_name)

            # Load the model with an optional override model config in JSON
            # representation. If provided, this config will be used for
            # loading the model.
            config = "{\"backend\":\"onnxruntime\", \"version_policy\":{\"specific\":{\"versions\":[1]}}}"
            pb_utils.load_model(model_name=self.model_name, config=config)

            # Load the mode with optional override files. The override files are
            # specified as a dictionary where the key is the file path (with
            # "file:" prefix) and the value is the file content as bytes. The
            # files will form the model directory that the model will be loaded
            # from. If specified, 'config' must be provided to be the model
            # configuration of the override model directory.
            with open('models/onnx_int32_int32_int32/1/model.onnx', 'rb') as file:
                data = file.read()
            files = {"file:1/model.onnx": data}
            pb_utils.load_model(model_name=self.model_name,
                                config=config, files=files)

    def execute(self, requests):
        # Execute the model
        ...
        # If the model is no longer needed, you can unload it. You can also
        # specify whether the dependents of the model should also be unloaded by
        # setting the 'unload_dependents' parameter to True. The default value
        # is False. Need to be careful when unloading the model as it can affect
        # other model instances or other models that depend on it.
        pb_utils.unload_model(model_name=self.model_name,
                              unload_dependents=True)

```

Note that the model loading API is only supported if the server is running in
[explicit model control mode](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_management.md#model-control-mode-explicit).
Additionally, the model loading API should only be used after the server has
been running, which means that the BLS model should not be loaded during server
startup. You can use different
[client endpoints](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md)
to load the model after the server has been started. The model loading API is
currently not supported during the `auto_complete_config` and `finalize`
functions.

## Using BLS with Stateful Models

[Stateful models](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#stateful-models)
require setting additional flags in the inference request to indicate the
start and end of a sequence. The `flags` argument in the `pb_utils.InferenceRequest`
object can be used to indicate whether the request is the first or last request
in the sequence. An example indicating that the request is starting the
sequence:

```python
inference_request = pb_utils.InferenceRequest(model_name='model_name',
  requested_output_names=['REQUESTED_OUTPUT_1', 'REQUESTED_OUTPUT_2'],
  inputs=[<list of pb_utils.Tensor objects>],
  request_id="1", correlation_id=4,
  flags=pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_START)
```

For indicating the ending of the sequence you can use the
`pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_END` flag. If the request is both
starting and ending a sequence at the same time (i.e. the sequence has only a
single request), you can use the bitwise OR operator to enable both of the
flags:

```
flags = pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_START | pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_END
```

## Limitation

- You need to make sure that the inference requests performed as a part of your
model do not create a circular dependency. For example, if model A performs an
inference request on itself and there are no more model instances ready to
execute the inference request, the model will block on the inference execution
forever.

- Async BLS is not supported when running a Python model in decoupled mode.

# Interoperability and GPU Support

Starting from 21.09 release, Python backend supports
[DLPack](https://github.com/dmlc/dlpack) for zero-copy transfer of Python
backend tensors to other frameworks. The methods below are added to the
`pb_utils.Tensor` object to facilitate the same:

## `pb_utils.Tensor.to_dlpack() -> PyCapsule`

This method can be called on existing instantiated tensors to convert
a Tensor to DLPack. The code snippet below shows how this works with PyTorch:

```python
from torch.utils.dlpack import from_dlpack
import triton_python_backend_utils as pb_utils

class TritonPythonModel:

  def execute(self, requests):
    ...
    input0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")

    # We have converted a Python backend tensor to a PyTorch tensor without
    # making any copies.
    pytorch_tensor = from_dlpack(input0.to_dlpack())
```

## `pb_utils.Tensor.from_dlpack() -> Tensor`

This static method can be used for creating a `Tensor` object from the DLPack
encoding of the tensor. For example:

```python
from torch.utils.dlpack import to_dlpack
import torch
import triton_python_backend_utils as pb_utils

class TritonPythonModel:

  def execute(self, requests):
    ...
    pytorch_tensor = torch.tensor([1, 2, 3], device='cuda')

    # Create a Python backend tensor from the DLPack encoding of a PyTorch
    # tensor.
    input0 = pb_utils.Tensor.from_dlpack("INPUT0", to_dlpack(pytorch_tensor))
```
Python backend allows tensors implementing
[`__dlpack__`](https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.array.__dlpack__.html)
and [`__dlpack_device__`](https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.array.__dlpack_device__.html)
[interface](https://dmlc.github.io/dlpack/latest/python_spec.html)
to be converted to Python backend tensors. For instance:

```python
input0 = pb_utils.Tensor.from_dlpack("INPUT0", pytorch_tensor)
```

This method only supports contiguous Tensors that are in C-order. If the tensor
is not C-order contiguous an exception will be raised.

For python models with input or output tensors of type BFloat16 (BF16), the
`as_numpy()` method is not supported, and the `from_dlpack` and `to_dlpack`
methods must be used instead.

## `pb_utils.Tensor.is_cpu() -> bool`

This function can be used to check whether a tensor is placed in CPU or not.

## Input Tensor Device Placement

By default, the Python backend moves all input tensors to CPU before providing
them to the Python model. Starting from 21.09, you can change this default
behavior. By setting `FORCE_CPU_ONLY_INPUT_TENSORS` to "no", Triton will not
move input tensors to CPU for the Python model. Instead, Triton will provide the
input tensors to the Python model in either CPU or GPU memory, depending on how
those tensors were last used. You cannot predict which memory will be used for
each input tensor so your Python model must be able to handle tensors in both
CPU and GPU memory. To enable this setting, you need to add this setting to the
`parameters` section of model configuration:

```
parameters: { key: "FORCE_CPU_ONLY_INPUT_TENSORS" value: {string_value:"no"}}
```

# Frameworks

Since Python Backend models can support most python packages, it is a common
workflow for users to use Deep Learning Frameworks like PyTorch in their
`model.py` implementation. This section will document some notes and FAQ about
this workflow.

> **Note**
>
> Using a deep learning framework/package in a Python Backend model is
> not necessarily the same as using the corresponding Triton Backend
> implementation. For example, the
> [PyTorch Backend](https://github.com/triton-inference-server/pytorch_backend)
> is different from using a Python Backend model that uses `import torch`.
> If you are seeing significantly different results from a model executed by
> the framework (ex: PyTorch) compared to the Python Backend model running the
> same framework, some of the first things you should check is that the
> framework versions being used and the input/output preparation are the same.

## PyTorch

For a simple example of using PyTorch in a Python Backend model, see the
[AddSubNet PyTorch example](#addsubnet-in-pytorch).

### PyTorch Determinism

When running PyTorch code, you may notice slight differences in output values
across runs or across servers depending on hardware, system load, driver, or even
batch size. These differences are generally related to the selection of CUDA
kernels used to execute the operations, based on the factors mentioned.

For most intents and purposes, these differences aren't large enough to affect
a model's final prediction. However, to understand where these differences come
from, see this [doc](https://pytorch.org/docs/stable/notes/randomness.html).

On Ampere devices and later, there is an optimization related to
FP32 operations called
[TensorFloat32 (TF32)](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/).
Typically this optimization will improve overall performance at the cost of
minor precision loss, but similarly this precision loss is acceptable for most
model predictions. For more info on TF32 in PyTorch and how to enable/disable
it as needed, see
[here](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices).

## TensorFlow

### TensorFlow Determinism

Similar to the PyTorch determinism section above, TensorFlow can have slight
differences in outputs based on various factors like hardware, system
configurations, or batch sizes due to the library's internal CUDA kernel
selection process. For more information on improving the determinism of outputs
in TensorFlow, see
[here](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism).

# Custom Metrics

Starting from 23.05, you can utlize Custom Metrics API to register and collect
custom metrics in the `initialize`, `execute`, and `finalize` functions of your
Python model. The Custom Metrics API is the Python equivalent of the
[TRITON C API custom metrics](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/metrics.md#custom-metrics)
support. You will need to take the ownership of the custom metrics created
through the APIs and must manage their lifetime. Note that a `MetricFamily`
object should be deleted only after all the `Metric` objects under it are
deleted if you'd like to explicitly delete the custom metrics objects.

Example below shows how to use this feature:

```python
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
      # Create a MetricFamily object to report the latency of the model
      # execution. The 'kind' parameter must be either 'COUNTER',
      # 'GAUGE' or 'HISTOGRAM'.
      self.metric_family = pb_utils.MetricFamily(
          name="preprocess_latency_ns",
          description="Cumulative time spent pre-processing requests",
          kind=pb_utils.MetricFamily.COUNTER
      )

      # Create a Metric object under the MetricFamily object. The 'labels'
      # is a dictionary of key-value pairs.
      self.metric = self.metric_family.Metric(
        labels={"model" : "model_name", "version" : "1"}
      )

    def execute(self, requests):
      responses = []

      for request in requests:
        # Pre-processing - time it to capture latency
        start_ns = time.time_ns()
        self.preprocess(request)
        end_ns = time.time_ns()

        # Update metric to track cumulative pre-processing latency
        self.metric.increment(end_ns - start_ns)

      ...

        print("Cumulative pre-processing latency:", self.metric.value())

      return responses
```

You can look at the [custom_metrics example](examples/custom_metrics/model.py)
which contains a complete example of demonstrating the Custom Metrics API for a
Python model.

# Examples

For using the Triton Python client in these examples you need to install
the
[Triton Python Client Library](https://github.com/triton-inference-server/client#getting-the-client-libraries-and-examples).
The Python client for each of the examples is in the `client.py` file.

## AddSub in NumPy

There is no dependencies required for the AddSub NumPy example. Instructions
on how to use this model is explained in the quick start section. You can
find the files in [examples/add_sub](examples/add_sub).

## AddSubNet in PyTorch

In order to use this model, you need to install PyTorch. We recommend using
`pip` method mentioned in the
[PyTorch website](https://pytorch.org/get-started/locally/).
Make sure that PyTorch is available in the same Python environment as other
dependencies. Alternatively, you can create a
[Python Execution Environment](#creating-custom-execution-environments).
You can find the files for this example in [examples/pytorch](examples/pytorch).

## AddSub in JAX

The JAX example shows how to serve JAX in Triton using Python Backend.
You can find the complete example instructions in
[examples/jax](examples/jax/README.md).

## Business Logic Scripting

The BLS example needs the dependencies required for both of the above examples.
You can find the complete example instructions in
[examples/bls](examples/bls/README.md) and
[examples/bls_decoupled](examples/bls_decoupled/README.md).

## Preprocessing

The Preprocessing example shows how to use Python Backend to do model
preprocessing.
You can find the complete example instructions in
[examples/preprocessing](examples/preprocessing/README.md).

## Decoupled Models

The examples of decoupled models shows how to develop and serve
[decoupled models](#decoupled-mode) in Triton using Python backend.
You can find the complete example instructions in
[examples/decoupled](examples/decoupled/README.md).

## Model Instance Kind

Triton model configuration allows users to provide kind to [instance group
settings.](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#instance-groups)
A python backend model can be written to respect the kind setting to control
the execution of a model instance either on CPU or GPU.

In the [model instance kind example](examples/instance_kind/README.md)
we demonstrate how this can be achieved for your python model.

## Auto-complete config

The auto-complete config example demonstrates how to use the
`auto_complete_config` function to define
[minimal model configuration](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#minimal-model-configuration)
when a configuration file is not available. You can find the complete example
instructions in [examples/auto_complete](examples/auto_complete/README.md).

## Custom Metrics

The example shows how to use custom metrics API in Python Backend. You can find
the complete example instructions in
[examples/custom_metrics](examples/custom_metrics/README.md).

# Running with Inferentia

Please see the
[README.md](https://github.com/triton-inference-server/python_backend/tree/main/inferentia/README.md)
located in the python_backend/inferentia sub folder.

# Logging

Starting from 22.09 release, your Python model can log information using the
following methods:

```python
import triton_python_backend_utils as pb_utils

class TritonPythonModel:

  def execute(self, requests):
    ...
    logger = pb_utils.Logger
    logger.log_info("Info Msg!")
    logger.log_warn("Warning Msg!")
    logger.log_error("Error Msg!")
    logger.log_verbose("Verbose Msg!")

```
*Note:* The logger can be defined and used in following class methods:

* initialize
* execute
* finalize

Log messages can also be sent with their log-level explcitiy specified:
```python
# log-level options: INFO, WARNING, ERROR, VERBOSE
logger.log("Specific Msg!", logger.INFO)
```
If no log-level is specified, this method will log INFO level messages.

Note that the Triton server's settings determine which log messages appear
within the server log. For example, if a model attempts to log a verbose-level
message, but Triton is not set to log verbose-level messages, it will not
appear in the server log. For more information on Triton's log settings and
how to adjust them dynamically, please see Triton's
[logging extension](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_logging.md)
documentation.

# Adding Custom Parameters in the Model Configuration

If your model requires custom parameters in the configuration, you can specify
that in the `parameters` section of the model config. For example:

```
parameters {
  key: "custom_key"
  value: {
    string_value: "custom_value"
  }
}
```

Now you can access this parameter in the `args` argument of the `initialize`
function:

```python
def initialize(self, args):
    print(json.loads(args['model_config'])['parameters'])
    # Should print {'custom_key': {'string_value': 'custom_value'}}
```

# Development with VSCode

The repository includes a `.devcontainer` folder that contains a `Dockerfile`
and `devcontainer.json` file to help you develop the Python backend
using
[Visual Studio Code](https://code.visualstudio.com/docs/devcontainers/containers).

In order to build the backend, you can execute the "Build Python Backend" task in the
[VSCode tasks](https://code.visualstudio.com/docs/editor/tasks). This will build
the Python backend and install the artifacts in
`/opt/tritonserver/backends/python`.


# Reporting problems, asking questions

We appreciate any feedback, questions or bug reporting regarding this
project. When help with code is needed, follow the process outlined in
the Stack Overflow (https://stackoverflow.com/help/mcve)
document. Ensure posted examples are:

* minimal – use as little code as possible that still produces the
  same problem

* complete – provide all parts needed to reproduce the problem. Check
  if you can strip external dependency and still show the problem. The
  less time we spend on reproducing problems the more time we have to
  fix it

* verifiable – test the code you're about to provide to make sure it
  reproduces the problem. Remove all other problems that are not
  related to your request/question.
