..
  # Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

.. _section-model-configuration:

Model Configuration
===================

Each model in a :ref:`section-model-repository` must include a model
configuration that provides required and optional information about
the model. Typically, this configuration is provided in a config.pbtxt
file specified as :doc:`ModelConfig <protobuf_api/model_config.proto>`
protobuf. In some cases, discussed in
:ref:`section-generated-model-configuration`, the model configuration
can be generated automatically by the inference server and so does not
need to be provided explicitly.

A minimal model configuration must specify :cpp:var:`name
<nvidia::inferenceserver::ModelConfig::name>`, :cpp:var:`platform
<nvidia::inferenceserver::ModelConfig::platform>`,
:cpp:var:`max_batch_size
<nvidia::inferenceserver::ModelConfig::max_batch_size>`,
:cpp:var:`input <nvidia::inferenceserver::ModelConfig::input>`, and
:cpp:var:`output <nvidia::inferenceserver::ModelConfig::output>`.

As a running example consider a TensorRT model called *mymodel* that
has two inputs, *input0* and *input1*, and one output, *output0*, all
of which are 16 entry float32 tensors. The minimal configuration is::

  name: "mymodel"
  platform: "tensorrt_plan"
  max_batch_size: 8
  input [
    {
      name: "input0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    },
    {
      name: "input1"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
  output [
    {
      name: "output0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]

**PyTorch Naming Convention:** Due to the absence of names for inputs and outputs
in the model, the "name" attribute of both the inputs and outputs in the
configuration must follow a specific naming convention i.e. "\<name\>__\<index\>".
Where <name> can be any string and <index> refers to the position of the
corresponding input/output. This means if there are two inputs and two outputs
they must be name as: "INPUT__0", "INPUT__1" and "OUTPUT__0", "OUTPUT__1" such
that "INPUT__0" refers to first input and INPUT__1 refers to the second input.

The name of the model must match the :cpp:var:`name
<nvidia::inferenceserver::ModelConfig::name>` of the model repository
directory containing the model. The :cpp:var:`platform
<nvidia::inferenceserver::ModelConfig::platform>` must be one of
**tensorrt_plan**, **tensorflow_graphdef**, **tensorflow_savedmodel**,
**caffe2_netdef**, **onnxruntime_onnx**, **pytorch_libtorch** or **custom**.

The datatypes allowed for input and output tensors varies based on the
type of the model. Section :ref:`section-datatypes` describes the
allowed datatypes and how they map to the datatypes of each model
type.

Input shape specified by :cpp:var:`dims
<nvidia::inferenceserver::ModelInput::dims>` indicates the shape of
input expected by the inference API.  Output shape specified by
:cpp:var:`dims <nvidia::inferenceserver::ModelOutput::dims>` indicates
the shape of output returned by the inference API. Both input and
output shape must have rank >= 1, that is, the empty shape **[ ]** is
not allowed. The :ref:`reshape <section-reshape>` property must be
used if the underlying framework model or custom backend requires an
input or output with an empty shape.

For models that support batched inputs the :cpp:var:`max_batch_size
<nvidia::inferenceserver::ModelConfig::max_batch_size>` value must be
>= 1. The TensorRT Inference Server assumes that the batching occurs
along a first dimension that is not listed in the inputs or
outputs. For the above example, the server expects to receive input
tensors with shape **[ x, 16 ]** and produces an output tensor with
shape **[ x, 16 ]**, where **x** is the batch size of the request.

For models that do not support batched inputs the
:cpp:var:`max_batch_size
<nvidia::inferenceserver::ModelConfig::max_batch_size>` value must be
zero. If the above example specified a :cpp:var:`max_batch_size
<nvidia::inferenceserver::ModelConfig::max_batch_size>` of zero, the
inference server would expect to receive input tensors with shape **[
16 ]**, and would produce an output tensor with shape **[ 16 ]**.

For models that support input and output tensors with variable-size
dimensions, those dimensions can be listed as -1 in the input and
output configuration. For example, if a model requires a 2-dimensional
input tensor where the first dimension must be size 4 but the second
dimension can be any size, the model configuration for that input
would include **dims: [ 4, -1 ]**. The inference server would then
accept inference requests where that input tensor's second dimension
was any value >= 0. The model configuration can be more restrictive
than what is allowed by the underlying model. For example, even though
the model allows the second dimension to be any size, the model
configuration could be specific as **dims: [ 4, 4 ]**. In this case,
the inference server would only accept inference requests where the
input tensor's shape was exactly **[ 4, 4 ]**.

.. _section-generated-model-configuration:

Generated Model Configuration
-----------------------------

By default, the model configuration file containing the required
settings must be provided with each model. However, if the server is
started with the -\\-strict-model-config=false option, then in some
cases the required portions of the model configuration file can be
generated automatically by the inference server. The required portion
of the model configuration are those settings shown in the example
minimal configuration above. Specifically:

* :ref:`TensorRT Plan <section-tensorrt-models>` models do not require
  a model configuration file because the inference server can derive
  all the required settings automatically.

* :ref:`TensorFlow SavedModel <section-tensorflow-models>` models do
  not require a model configuration file because the inference server
  can derive all the required settings automatically.

* :ref:`ONNX Runtime ONNX <section-onnx-models>` models do not require
  a model configuration file because the inference server can derive
  all the required settings automatically. However, if the model supports
  batching, the initial batch dimension must be variable-size for all inputs
  and outputs.

* :ref:`PyTorch TorchScript <section-pytorch-models>` models have an optional
  output configuration in the model configuration file to support cases where
  there are variable number and/or datatypes of output.

When using -\\-strict-model-config=false you can see the model
configuration that was generated for a model by using the :ref:`Status
API <section-api-status>`.

The TensorRT Inference Server only generates the required portion of
the model configuration file. You must still provide the optional
portions of the model configuration if necessary, such as
:cpp:var:`version_policy
<nvidia::inferenceserver::ModelConfig::version_policy>`,
:cpp:var:`optimization
<nvidia::inferenceserver::ModelConfig::optimization>`,
:cpp:var:`scheduling and batching
<nvidia::inferenceserver::ModelConfig::scheduling_choice>`,
:cpp:var:`instance_group
<nvidia::inferenceserver::ModelConfig::instance_group>`,
:cpp:var:`default_model_filename
<nvidia::inferenceserver::ModelConfig::default_model_filename>`,
:cpp:var:`cc_model_filenames
<nvidia::inferenceserver::ModelConfig::cc_model_filenames>`, and
:cpp:var:`tags <nvidia::inferenceserver::ModelConfig::tags>`.

When serving a classification model, keep in mind that :cpp:var:`label_filename
<nvidia::inferenceserver::ModelOutput::label_filename>` can not be automatically
derived. You will need to either create a **config.pbtxt** file specifying all
required :cpp:var:`output<nvidia::inferenceserver::ModelOutput>` along with the
:cpp:var:`label_filename<nvidia::inferenceserver::ModelOutput::label_filename>`,
or handle the mapping from model output to label in the client code directly.

.. _section-datatypes:

Datatypes
---------

The following table shows the tensor datatypes supported by the
TensorRT Inference Server. The first column shows the name of the
datatype as it appears in the model configuration file. The other
columns show the corresponding datatype for the model frameworks
supported by the server and for the Python numpy library. If a model
framework does not have an entry for a given datatype, then the
inference server does not support that datatype for that model.

+--------------+--------------+--------------+--------------+--------------+---------+--------------+
|Type          |TensorRT      |TensorFlow    |Caffe2        |ONNX Runtime  |PyTorch  |NumPy         |
+==============+==============+==============+==============+==============+=========+==============+
|TYPE_BOOL     |              |DT_BOOL       |BOOL          |BOOL          |kBool    |bool          |
+--------------+--------------+--------------+--------------+--------------+---------+--------------+
|TYPE_UINT8    |              |DT_UINT8      |UINT8         |UINT8         |kByte    |uint8         |
+--------------+--------------+--------------+--------------+--------------+---------+--------------+
|TYPE_UINT16   |              |DT_UINT16     |UINT16        |UINT16        |         |uint16        |
+--------------+--------------+--------------+--------------+--------------+---------+--------------+
|TYPE_UINT32   |              |DT_UINT32     |              |UINT32        |         |uint32        |
+--------------+--------------+--------------+--------------+--------------+---------+--------------+
|TYPE_UINT64   |              |DT_UINT64     |              |UINT64        |         |uint64        |
+--------------+--------------+--------------+--------------+--------------+---------+--------------+
|TYPE_INT8     | kINT8        |DT_INT8       |INT8          |INT8          |kChar    |int8          |
+--------------+--------------+--------------+--------------+--------------+---------+--------------+
|TYPE_INT16    |              |DT_INT16      |INT16         |INT16         |kShort   |int16         |
+--------------+--------------+--------------+--------------+--------------+---------+--------------+
|TYPE_INT32    | kINT32       |DT_INT32      |INT32         |INT32         |kInt     |int32         |
+--------------+--------------+--------------+--------------+--------------+---------+--------------+
|TYPE_INT64    |              |DT_INT64      |INT64         |INT64         |kLong    |int64         |
+--------------+--------------+--------------+--------------+--------------+---------+--------------+
|TYPE_FP16     | kHALF        |DT_HALF       |FLOAT16       |FLOAT16       |         |float16       |
+--------------+--------------+--------------+--------------+--------------+---------+--------------+
|TYPE_FP32     | kFLOAT       |DT_FLOAT      |FLOAT         |FLOAT         |kFloat   |float32       |
+--------------+--------------+--------------+--------------+--------------+---------+--------------+
|TYPE_FP64     |              |DT_DOUBLE     |DOUBLE        |DOUBLE        |kDouble  |float64       |
+--------------+--------------+--------------+--------------+--------------+---------+--------------+
|TYPE_STRING   |              |DT_STRING     |              |STRING        |         |dtype(object) |
+--------------+--------------+--------------+--------------+--------------+---------+--------------+

For TensorRT each value is in the nvinfer1::DataType namespace. For
example, nvinfer1::DataType::kFLOAT is the 32-bit floating-point
datatype.

For TensorFlow each value is in the tensorflow namespace. For example,
tensorflow::DT_FLOAT is the 32-bit floating-point value.

For Caffe2 each value is in the caffe2 namespace and is prepended with
TensorProto\_DataType\_. For example, caffe2::TensorProto_DataType_FLOAT
is the 32-bit floating-point datatype.

For ONNX Runtime each value is prepended with ONNX_TENSOR_ELEMENT_DATA_TYPE_.
For example, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT is the 32-bit floating-point
datatype.

For PyTorch each value is in the torch namespace. For example, torch::kFloat
is the 32-bit floating-point datatype.

For Numpy each value is in the numpy module. For example, numpy.float32
is the 32-bit floating-point datatype.

.. _section-reshape:

Reshape
-------

The :cpp:var:`ModelTensorReshape
<nvidia::inferenceserver::ModelTensorReshape>` property on a model
configuration input or output is used to indicate that the input or
output shape accepted by the inference API differs from the input or
output shape expected or produced by the underlying framework model or
custom backend.

For an input, :cpp:var:`reshape
<nvidia::inferenceserver::ModelInput::reshape>` can be used to reshape
the input tensor to a different shape expected by the framework or
backend. A common use-case is where a model that supports batching
expects a batched input to have shape **[ batch-size ]**, which means
that the batch dimension fully describes the shape. For the inference
API the equivalent shape **[ batch-size, 1 ]** must be specified since
each input in the batch must specify a non-empty shape. For this case
the input should be specified as::

  input [
    {
      name: "in"
      dims: [ 1 ]
      reshape: { shape: [ ] }
    }
    ...

For an output, :cpp:var:`reshape
<nvidia::inferenceserver::ModelOutput::reshape>` can be used to
reshape the output tensor produced by the framework or backend to a
different shape that is returned by the inference API. A common
use-case is where a model that supports batching expects a batched
output to have shape **[ batch-size ]**, which means that the batch
dimension fully describes the shape. For the inference API the
equivalent shape **[ batch-size, 1 ]** must be specified since each
output in the batch must specify a non-empty shape. For this case the
output should be specified as::

  output [
    {
      name: "in"
      dims: [ 1 ]
      reshape: { shape: [ ] }
    }
    ...

.. _section-version-policy:

Version Policy
--------------

Each model can have one or more :ref:`versions available in the model
repository <section-model-versions>`. The
:cpp:var:`nvidia::inferenceserver::ModelVersionPolicy` schema allows
the following policies.

* :cpp:var:`All
  <nvidia::inferenceserver::ModelVersionPolicy::All>`: All versions
  of the model that are available in the model repository are
  available for inferencing.

* :cpp:var:`Latest
  <nvidia::inferenceserver::ModelVersionPolicy::Latest>`: Only the
  latest ‘n’ versions of the model in the repository are available for
  inferencing. The latest versions of the model are the numerically
  greatest version numbers.

* :cpp:var:`Specific
  <nvidia::inferenceserver::ModelVersionPolicy::Specific>`: Only the
  specifically listed versions of the model are available for
  inferencing.

If no version policy is specified, then :cpp:var:`Latest
<nvidia::inferenceserver::ModelVersionPolicy::Latest>` (with
num_version = 1) is used as the default, indicating that only the most
recent version of the model is made available by the inference
server. In all cases, the addition or removal of version
subdirectories from the model repository can change which model
version is used on subsequent inference requests.

Continuing the above example, the following configuration specifies
that all versions of the model will be available from the server::

  name: "mymodel"
  platform: "tensorrt_plan"
  max_batch_size: 8
  input [
    {
      name: "input0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    },
    {
      name: "input1"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
  output [
    {
      name: "output0"
      data_type: TYPE_FP32
      dims: [ 16 ]
    }
  ]
  version_policy: { all { }}

.. _section-instance-groups:

Instance Groups
---------------

The TensorRT Inference Server can provide multiple :ref:`execution
instances <section-concurrent-model-execution>` of a model so that
multiple simultaneous inference requests for that model can be handled
simultaneously. The model configuration :cpp:var:`ModelInstanceGroup
<nvidia::inferenceserver::ModelInstanceGroup>` is used to specify the
number of execution instances that should be made available and what
compute resource should be used for those instances.

By default, a single execution instance of the model is created for
each GPU available in the system. The instance-group setting can be
used to place multiple execution instances of a model on every GPU or
on only certain GPUs. For example, the following configuration will
place two execution instances of the model to be available on each
system GPU::

  instance_group [
    {
      count: 2
      kind: KIND_GPU
    }
  ]

And the following configuration will place one execution instance on
GPU 0 and two execution instances on GPUs 1 and 2::

  instance_group [
    {
      count: 1
      kind: KIND_GPU
      gpus: [ 0 ]
    },
    {
      count: 2
      kind: KIND_GPU
      gpus: [ 1, 2 ]
    }
  ]

The instance group setting is also used to enable exection of a model
on the CPU. A model can be executed on the CPU even if there is a GPU
available in the system. The following places two execution instances
on the CPU::

  instance_group [
    {
      count: 2
      kind: KIND_CPU
    }
  ]

.. _section-scheduling-and-batching:

Scheduling And Batching
-----------------------

The TensorRT Inference Server supports batch inferencing by allowing
individual inference requests to specify a batch of inputs. The
inferencing for a batch of inputs is performed at the same time which
is especially important for GPUs since it can greatly increase
inferencing throughput. In many use-cases the individual inference
requests are not batched, therefore, they do not benefit from the
throughput benefits of batching.

The inference server contains multiple scheduling and batching
algorithms that support many different model types and use-cases. More
information about model types and schedulers can be found in
:ref:`section-models-and-schedulers`.

.. _section-default-scheduler:

Default Scheduler
^^^^^^^^^^^^^^^^^

The default scheduler is used for a model if none of the
:cpp:var:`scheduling_choice
<nvidia::inferenceserver::ModelConfig::scheduling_choice>`
configurations are specified. This scheduler distributes inference
requests to all :ref:`instances <section-instance-groups>` configured for
the model.

.. _section-dynamic-batcher:

Dynamic Batcher
^^^^^^^^^^^^^^^

Dynamic batching is a feature of the inference server that allows
inference requests to be combined by the server, so that a batch is
created dynamically, resulting in the same increased throughput seen
for batched inference requests. The dynamic batcher should be used for
:ref:`stateless <section-models-and-schedulers>` models. The
dynamically created batches are distributed to all :ref:`instances
<section-instance-groups>` configured for the model.

Dynamic batching is enabled and configured independently for each
model using the :cpp:var:`ModelDynamicBatching
<nvidia::inferenceserver::ModelDynamicBatching>` settings in the model
configuration. These settings control the preferred size(s) of the
dynamically created batches as well as a maximum time that requests
can be delayed in the scheduler to allow other requests to join the
dynamic batch.

The following configuration enables dynamic batching with preferred
batch sizes of 4 and 8, and a maximum delay time of 100 microseconds::

  dynamic_batching {
    preferred_batch_size: [ 4, 8 ]
    max_queue_delay_microseconds: 100
  }

The size of generated batches can be examined in aggregate using Count
metrics, see :ref:`section-metrics`. Inference server verbose logging
can be used to examine the size of individual batches.

.. _section-sequence-batcher:

Sequence Batcher
^^^^^^^^^^^^^^^^

Like the dynamic batcher, the sequence batcher combines non-batched
inference requests, so that a batch is created dynamically. Unlike the
dynamic batcher, the sequence batcher should be used for
:ref:`stateful <section-models-and-schedulers>` models where a
sequence of inference requests must be routed to the same model
instance. The dynamically created batches are distributed to all
:ref:`instances <section-instance-groups>` configured for the model.

Sequence batching is enabled and configured independently for each
model using the :cpp:var:`ModelSequenceBatching
<nvidia::inferenceserver::ModelSequenceBatching>` settings in the
model configuration. These settings control the sequence timeout as
well as configuring how the inference server will send control signals
to the model indicating sequence start, end, ready and
correlation ID. See :ref:`section-models-and-schedulers` for more
information and examples.

The size of generated batches can be examined in aggregate using Count
metrics, see :ref:`section-metrics`. Inference server verbose logging
can be used to examine the size of individual batches.

.. _section-ensemble-scheduler:

Ensemble Scheduler
^^^^^^^^^^^^^^^^^^

The ensemble scheduler must be used for :ref:`ensemble models
<section-ensemble-models>` and cannot be used for any other type of model.

The ensemble scheduler is enabled and configured independently for each
model using the :cpp:var:`ModelEnsembleScheduling
<nvidia::inferenceserver::ModelEnsembleScheduling>` settings in the
model configuration. The settings describe the models that are included in the
ensemble and the flow of tensor values between the models. See
:ref:`section-ensemble-models` for more information and examples.

.. _section-optimization-policy:

Optimization Policy
-------------------

The model configuration :cpp:var:`ModelOptimizationPolicy
<nvidia::inferenceserver::ModelOptimizationPolicy>` is used to specify
optimization and prioritization settings for a model. These settings
control if/how a model is optimized by the backend framework and how
it is scheduled and executed by the inference server. See the protobuf
documentation for the currently available settings.

.. _section-optimization-policy-tensorrt:

TensorRT Optimization
^^^^^^^^^^^^^^^^^^^^^

The TensorRT optimization is an especially powerful optimization that
can be enabled for TensorFlow and ONNX models. When enabled for a
model, TensorRT optimization will be applied to the model at load time
or when it first receives inference requests. TensorRT optimizations
include specializing and fusing model layers, and using reduced
precision (for example 16-bit floating-point) to provide significant
throughput and latency improvements.

.. _section-model-warm-up:

Model Warmup
------------

For some framework backends, model initialization may be delayed until the
first inference is requested, TF-TRT optimization for example, which introduces
unexpected latency seen by the client. The model configuration
:cpp:var:`ModelWarmup <nvidia::inferenceserver::ModelWarmup>` is used to specify
warmup settings for a model. The settings define a series of inference requests
that the inference server should create to warm-up each model instance. A model
instance will be served only if it completes the requests successfully.
Note that the effect of warming up models varies depending on the framework
backend, and it will cause the server to be less responsive to model update, so
the users should experiment and choose the configuration that suits their need.
See the protobuf documentation for the currently available settings.
