..
  # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

.. _section-backends:

Backends
========

A *backend* is the implementation that executes a model. A backend can
be a wrapper around a deep learning framework, like PyTorch,
TensorFlow, TensorRT or ONNX. Or a backend can be custom C/C++ logic
performing any operation, for example, image pre-processing.

All backends must implement the
:ref:`section-triton-backend-api`. Triton uses this API to send
requests to the backend for execution and the backend uses the API to
communicate with Triton.

Every model must be associated with a backend. A model's backend is
specified in the :ref:`section-model-configuration` using the
:cpp:var:`backend <nvidia::inferenceserver::ModelConfig::backend>` and
:cpp:var:`platform <nvidia::inferenceserver::ModelConfig::platform>`
properties. Depending on the backend one or the other of these
properties is optional:

* For TensorRT, :cpp:var:`backend
  <nvidia::inferenceserver::ModelConfig::backend>` must be set to
  *tensorrt* or :cpp:var:`platform
  <nvidia::inferenceserver::ModelConfig::platform>` must be set to
  *tensorrt\_plan*.

* For PyTorch, :cpp:var:`backend
  <nvidia::inferenceserver::ModelConfig::backend>` must be set to
  *pytorch* or :cpp:var:`platform
  <nvidia::inferenceserver::ModelConfig::platform>` must be set to
  *pytorch\_libtorch*.

* For ONNX, :cpp:var:`backend
  <nvidia::inferenceserver::ModelConfig::backend>` must be set to
  *onnxruntime* or :cpp:var:`platform
  <nvidia::inferenceserver::ModelConfig::platform>` must be set to
  *onnxruntime\_onnx*.

* For TensorFlow, :cpp:var:`platform
  <nvidia::inferenceserver::ModelConfig::platform>` must be set to
  *tensorflow\_graphdef* or *tensorflow\_savedmodel*. Optionally
  :cpp:var:`backend <nvidia::inferenceserver::ModelConfig::backend>`
  can be set to *tensorflow*.

* For all other backends, :cpp:var:`backend
  <nvidia::inferenceserver::ModelConfig::backend>` must be set to the
  name of the backend. The :cpp:var:`platform
  <nvidia::inferenceserver::ModelConfig::platform>` property is
  optional.

.. _section-backend-shared-library:

Backend Shared Library
^^^^^^^^^^^^^^^^^^^^^^

Each backend must be implemented as a shared library and the name of
the shared library must be *libtriton\_<backend-name>.so*. For example,
if the name of the backend is "mybackend", a model indicates that it
uses the backend by setting the model configuration :cpp:var:`backend
<nvidia::inferenceserver::ModelConfig::backend>` to *mybackend*, and
Triton looks for *libtriton\_mybackend.so* as the shared library that
implements the backend.

For a model, *M* that specifies backend *B*, Triton searches for the
backend shared library in the following places, in this order:

* <model\_repository>/M/<version\_directory>/libtriton\_B.so

* <model\_repository>/M/libtriton\_B.so

* <backend\_directory>/B/libtriton\_B.so

Where <backend\_directory> is by default /opt/tritonserver/backends.
The -\\-backend-directory flag can be used to override the default.

.. _section-triton-backend-api:

Triton Backend API
^^^^^^^^^^^^^^^^^^

A Triton backend must implement the C interface defined in
`tritonbackend.h
<https://github.com/NVIDIA/triton-inference-server/blob/master/src/backends/backend/tritonbackend.h>`_.

Triton Backend Objects
......................

The following abstractions are used by the API.

TRITONBACKEND\_Backend
---------------------

A TRITONBACKEND\_Backend object represents the backend itself. The
same backend object is shared across all models that use the
backend. The associated API, like TRITONBACKEND\_BackendName, is used
to get information about the backend and to associate a user-defined
state with the backend.

A backend can optionally implement TRITONBACKEND\_Initialize and
TRITONBACKEND\_Finalize to get notification of when the backend object
is created and destroyed (see :ref:`section-backend-lifecycles` for
more information about backend lifecycle). Most backends do not
require a user-defined state that spans all models using the backend
and so do not need to implement these functions.

TRITONBACKEND\_Model
-------------------

A TRITONBACKEND\_Model object represents a model. Each model loaded by
Triton is associated with a TRITONBACKEND\_Model. Each model can use
the TRITONBACKEND\_ModelBackend API to get the backend object
representing the backend that is used by the model.

The same model object is shared across all instances of that
model. The associated API, like TRITONBACKEND\_ModelName, is used to
get information about the model and to associate a user-defined state
with the model.

Most backends will implement TRITONBACKEND\_ModelInitialize and
TRITONBACKEND\_ModelFinalize to initialize the backend for a given
model and to manage the user-defined state associated with the model
(see :ref:`section-backend-lifecycles` for more information about
model lifecycle).

The backend must take into account threading concerns when
implementing TRITONBACKEND\_ModelInitialize and
TRITONBACKEND\_ModelFinalize.  Triton will not perform multiple
simultaneous calls to these functions for a given model; however, if a
backend is used by multiple models Triton may simultaneously call the
functions with a different thread for each model. As a result, the
backend must be able to handle multiple simultaneous calls to the
functions. Best practice for backend implementations is to use only
function-local and model-specific user-defined state in these
functions, as is shown in the :ref:`example backends
<section-example-backends>` like *identity*.

TRITONBACKEND\_ModelInstance
---------------------------

A TRITONBACKEND\_ModelInstance object represents a model
*instance*. Triton creates one or more instances (that is, copies) of
the model based on the :ref:`section-instance-groups` specified in the
model configuration. Each of these instances is associated with a
TRITONBACKEND\_ModelInstance object.

The only function that the backend must implement is
TRITONBACKEND\_ModelInstanceExecute. The
TRITONBACKEND\_ModelInstanceExecute function is called by Triton to
perform inference/computation on a batch of inference requests. Most
backends will also implement TRITONBACKEND\_ModelInstanceInitialize
and TRITONBACKEND\_ModelInstanceFinalize to initialize the backend for
a given model instance and to manage the user-defined state associated
with the model (see :ref:`section-backend-lifecycles` for more
information about model instance lifecycle).

The backend must take into account threading concerns when
implementing TRITONBACKEND\_ModelInstanceInitialize,
TRITONBACKEND\_ModelInstanceFinalize and
TRITONBACKEND\_ModelInstanceExecute.  Triton will not perform multiple
simultaneous calls to these functions for a given model instance;
however, if a backend is used by a model with multiple instances or by
multiple models Triton may simultaneously call the functions with a
different thread for each model instance. As a result, the backend
must be able to handle multiple simultaneous calls to the
functions. Best practice for backend implementations is to use only
function-local and model-specific user-defined state in these
functions, as is shown in the :ref:`example backends
<section-example-backends>` like *identity*.

TRITONBACKEND\_Request
---------------------

A TRITONBACKEND\_Request object represents an inference request made
to the model. The backend takes ownership of the request object(s) in
TRITONBACKEND\_ModelInstanceExecute and must release each request by
calling TRITONBACKEND\_RequestRelease. See
:ref:`section-backend-lifecycles` for more information about request
lifecycle.

The Triton Backend API allows the backend to get information about the
request as well as the input and request output tensors of the
request. Each request input is represented by a TRITONBACKEND\_Input
object.

TRITONBACKEND\_Response
----------------------

A TRITONBACKEND\_Response object represents a response sent by the
backend for a specific request. The backend uses the response API to
set the name, shape, datatype and tensor values for each output tensor
included in the response. The response can indicate either a failed or
a successful request. See :ref:`section-backend-lifecycles` for more
information about request-response lifecycle.

.. _section-backend-lifecycles:

Backend Lifecycles
..................

A backend must carefully manage the lifecycle of the backend itself,
the models and model instances that use the backend and the inference
requests that execute on the model instances using the backend.

Backend and Model
-----------------

Backend, model and model instance initialization is triggered when
Triton :ref:`loads a model <section-model-management>`:

* If the model requires a backend that is not already in use by an
  already loaded model, then:

  * Triton :ref:`loads the shared library
    <section-backend-shared-library>` that implements the backend
    required by the model.

  * Triton creates the TRITONBACKEND\_Backend object that represents
    the backend.

  * Triton calls TRITONBACKEND\_Initialize if it is implemented in the
    backend shared library. TRITONBACKEND\_Initialize should not return
    until the backend is completely initialized. If
    TRITONBACKEND\_Initialize returns an error, Triton will unload the
    backend shared library and show that the model failed to load.

* Triton creates the TRITONBACKEND\_Model object that represents the
  model. Triton calls TRITONBACKEND\_ModelInitialize if it is
  implemented in the backend shared library.
  TRITONBACKEND\_ModelInitialize should not return until the backend
  is completely initialized for the model. If
  TRITONBACKEND\_ModelInitialize returns an error, Triton will show
  that the model failed to load.

* For each model instance specified for the model in the model
  configuration:

  * Triton creates the TRITONBACKEND\_ModelInstance object that
    represents the model instance.

  * Triton calls TRITONBACKEND\_ModelInstanceInitialize if it is
    implemented in the backend shared library.
    TRITONBACKEND\_ModelInstanceInitialize should not return until the
    backend is completely initialized for the instance. If
    TRITONBACKEND\_ModelInstanceInitialize returns an error, Triton
    will show that the model failed to load.

Backend, model and model instance finalization is triggered when
Triton :ref:`unloads a model <section-model-management>`:

* For each model instance:

  * Triton calls TRITONBACKEND\_ModelInstanceFinalize if it is
    implemented in the backend shared library.
    TRITONBACKEND\_ModelInstanceFinalize should not return until the
    backend is completely finalized, including stopping any threads
    create for the model instance and freeing any user-defined state
    created for the model instance.

  * Triton destroys the TRITONBACKEND\_ModelInstance object that
    represents the model instance.

* Triton calls TRITONBACKEND\_ModelFinalize if it is implemented in the
  backend shared library. TRITONBACKEND\_ModelFinalize should not
  return until the backend is completely finalized, including stopping
  any threads create for the model and freeing any user-defined state
  created for the model.

* Triton destroys the TRITONBACKEND\_Model object that represents the
  model.

* If no other loaded model requires the backend, then:

  * Triton calls TRITONBACKEND\_Finalize if it is implemented in the
    backend shared library. TRITONBACKEND\_ModelFinalize should not
    return until the backend is completely finalized, including
    stopping any threads create for the backend and freeing any
    user-defined state created for the backend.

  * Triton destroys the TRITONBACKEND\_Backend object that represents
    the backend.

  * Triton :ref:`unloads the shared library
    <section-backend-shared-library>` that implements the backend.

Inference Requests and Responses
--------------------------------

Triton calls TRITONBACKEND\_ModelInstanceExecute to execute inference
requests on a model instance. Each call to
TRITONBACKEND\_ModelInstanceExecute communicates a batch of requests
to execute and the instance of the model that should be used to
execute those requests. The backend should not allow the scheduler
thread to return from TRITONBACKEND\_ModelInstanceExecute until that
instance is ready to handle another set of requests. Typically this
means that the TRITONBACKEND\_ModelInstanceExecute function will
create responses and release the requests before returning.

Most backends will create a single response for each request. For that
kind of backend executing a single inference requests requires the
following steps:

* Create a response for the request using TRITONBACKEND\_ResponseNew.

* For each request input tensor use TRITONBACKEND\_InputProperties to
  get shape and datatype of the input as well as the buffer(s)
  containing the tensor contents.

* For each output tensor that the request expects to be returned, use
  TRITONBACKEND\_ResponseOutput to create the output tensor of the
  required datatype and shape. Use TRITONBACKEND\_OutputBuffer to get a
  pointer to the buffer where the tensor's contents should be written.

* Use the inputs to perform the inference computation that produces
  the requested output tensor contents into the appropriate output
  buffers.

* Optionally set parameters in the response.

* Send the response using TRITONBACKEND\_ResponseSend.

* Release the request using TRITONBACKEND\_RequestRelease.

For a batch of requests the backend should attempt to combine the
execution of the individual requests as much as possible to increase
performance.

It is also possible for a backend to send multiple responses for a
request or not send any responses for a request. A backend may also
send responses out-of-order relative to the order that the request
batches are executed. Backends and models that operate in this way are
referred to as *decoupled* backends and models, and are typically much
more difficult to implement. The :ref:`repeat example
<section-example-backends>` shows a simplified implementation of a
decoupled backend.

.. _section-example-backends:

Example Backends
^^^^^^^^^^^^^^^^

Triton backend implementations can be found in the `src/backends
<https://github.com/NVIDIA/triton-inference-server/tree/master/src/backends>`_. The
*identity* backend is a simple example backend that uses and explains
most of the Triton Backend API. The *repeat* backend shows a more
advanced example of how a backend can produce multiple responses per
request. These examples are implemented to illustrate the backend API
and not for performance; and so should not necessarily be used as the
baseline for a high-performance backend.

Legacy Custom Backends
^^^^^^^^^^^^^^^^^^^^^^

In previous version of Triton, custom backends could be created using
the *custom.h* interface. New backends should be written using the
:ref:`section-triton-backend-api` but models that use *custom.h* based
backends will continue to be supported by Triton.
