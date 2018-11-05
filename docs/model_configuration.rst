..
  # Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

Each model in a :ref:`section-model-repository` must include a file
called config.pbtxt that contains the configuration information for
the model. The model configuration must be specified as
:doc:`ModelConfig <protobuf_api/model_config.proto>` protobuf.

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

The name of the model must match the :cpp:var:`name
<nvidia::inferenceserver::ModelConfig::name>` of the model repository
directory containing the model. The :cpp:var:`platform
<nvidia::inferenceserver::ModelConfig::platform>` must be one of
**tensorrt_plan**, **tensorflow_graphdef**, **tensorflow_savedmodel**,
or **caffe2_netdef**.

For models that support batched inputs the :cpp:var:`max_batch_size
<nvidia::inferenceserver::ModelConfig::max_batch_size>` value must be
>= 1. The inference server assumes that the batching occurs along a
first dimension that is not listed in the inputs or outputs. For the
above example the inference server expects to receive input tensors
with shape **[ x, 16 ]** and produces an output tensor with shape **[
x, 16 ]**, where **x** is the batch size of the request.

For models that do not support batched inputs the
:cpp:var:`max_batch_size
<nvidia::inferenceserver::ModelConfig::max_batch_size>` value must be
zero. If the above example specified a :cpp:var:`max_batch_size
<nvidia::inferenceserver::ModelConfig::max_batch_size>` of zero the
inference server would expect to receive input tensors with shape **[
16 ]**, and would produce an output tensor with shape **[ 16 ]**.

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
that all versions of the model will be available from the inference
server::

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

The inference server can provide multiple :ref:`execution instances
<section-concurrent-model-execution>` of a model so that multiple
simultaneous inference requests for that model can be handled
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
on the CPU. The following places two execution instances on the CPU::

  instance_group [
    {
      count: 2
      kind: KIND_CPU
    }
  ]

.. _section-dynamic-batching:

Dynamic Batching
----------------

The inference server supports batch inferencing by allowing individual
inference requests to specify a batch of inputs. The inferencing for a
batch of inputs is processed at the same time which is especially
important for GPUs since it can greatly increase inferencing
throughput. In many use-cases the individual inference requests are
not batched, therefore, they do not benefit from the throughput
benefits of batching.

Dynamic batching is a feature of the inference server that allows
non-batched inference requests to be combined by the inference server,
so that a batch is created dynamically, resulting in the same
increased throughput seen for batched inference requests.

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

.. _section-optimization-policy:

Optimization Policy
-------------------

The model configuration :cpp:var:`ModelOptimizationPolicy
<nvidia::inferenceserver::ModelOptimizationPolicy>` is used to specify
optimization settings for a model. These settings control if/how a
model is optimized by the backend framework. See the protobuf
documentation for the currently available settings.
