<!--
# Copyright 2018-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Business Logic Scripting

Triton's
[ensemble](ensemble_models.md#ensemble-models)
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
[Examples](../python_backend/README.md#examples) section.

## Using BLS with Decoupled Models

Starting from 23.03 release, you can execute inference requests on decoupled
models in both [default mode](../python_backend/README.md#default-mode) and
[decoupled mode](../python_backend/README.md#decoupled-mode). By setting the `decoupled` parameter to
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
the [Examples](../python_backend/README.md#examples) section.

Starting from the 22.04 release, the lifetime of the BLS output tensors have
been improved such that if a tensor is no longer needed in your Python model it
will be automatically deallocated. This can increase the number of BLS requests
that you can execute in your model without running into the out of GPU or
shared memory error.

Note: Async BLS is not supported on Python 3.6 or lower due to the `async`
keyword and `asyncio.run` being introduced in Python 3.7.

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