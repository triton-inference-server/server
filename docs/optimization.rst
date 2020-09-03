..
  # Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

.. _section-optimization:

Optimization
============

The Triton Inference Server has many features that you can use to
decrease latency and increase throughput for your model. This section
discusses these features and demonstrates how you can use them to
improve the performance of your model. As a prerequisite you should
follow the :ref:`section-quickstart` to get Triton and client examples
running with the example model repository.

This section focuses on understanding latecy and throughput tradeoffs
for a single model. The :ref:`Model Analyzer <section-model-analyzer>`
section describes a tool that helps you understand the GPU memory and
compute utilization of your models so you can decide how to best run
multiple models on a single GPU.

Unless you already have a client application suitable for measuring
the performance of your model on Triton, you should familiarize
yourself with :ref:`perf\_client <section-perf-client>`. The
perf\_client application is an essential tool for optimizing your
model's performance.

As a running example demonstrating the optimization features and
options, we will use a Caffe2 ResNet50 model that you can obtain by
following the :ref:`section-quickstart`. As a baseline we use
perf\_client to determine the performance of the model using a `basic
model configuration that does not enable any performance features
<https://github.com/triton-inference-server/server/blob/master/docs/examples/model_repository/resnet50_netdef/config.pbtxt>`_::

  $ perf_client -m resnet50_netdef --percentile=95 --concurrency-range 1:4
  ...
  Inferences/Second vs. Client p95 Batch Latency
  Concurrency: 1, 159 infer/sec, latency 6701 usec
  Concurrency: 2, 204.8 infer/sec, latency 9807 usec
  Concurrency: 3, 204.2 infer/sec, latency 14846 usec
  Concurrency: 4, 199.6 infer/sec, latency 20499 usec

The results show that our non-optimized model configuration gives a
throughput of about 200 inferences per second. Note how there is a
significant throughput increase going from one concurrent request to
two concurrent requests and then throughput levels off. With one
concurrent request Triton is idle during the time when the response is
returned to the client and the next request is received at the
server. Throughput increases with a concurrency of 2 because Triton
overlaps the processing of one request with the communication of the
other. Because we are running perf\_client on the same system as
Triton, 2 requests are enough to completely hide the communication
latency.

Optimization Settings
---------------------

For most models, the Triton feature that provides the largest
performance improvement is the :ref:`section-dynamic-batcher`. If your
model does not support batching then you can skip ahead to
:ref:`section-opt-model-instances`.

.. _section-opt-dynamic-batcher:

Dynamic Batcher
^^^^^^^^^^^^^^^

The dynamic batcher combines individual inference requests into a
larger batch that will often execute much more efficiently than
executing the individual requests independently. To enable the dynamic
batcher stop Triton, add the following lines to the end of the model
configuration file for resnet50\_netdef, and then restart Triton::

  dynamic_batching { }

The dynamic batcher allows Triton to handle a higher number of
concurrent requests because those requests are combined for
inference. So run perf\_client with request concurrency from 1 to 8::

  $ perf_client -m resnet50_netdef --percentile=95 --concurrency-range 1:8
  ...
  Inferences/Second vs. Client p95 Batch Latency
  Concurrency: 1, 154.2 infer/sec, latency 6662 usec
  Concurrency: 2, 203.6 infer/sec, latency 9931 usec
  Concurrency: 3, 242.4 infer/sec, latency 12421 usec
  Concurrency: 4, 335.6 infer/sec, latency 12423 usec
  Concurrency: 5, 335.2 infer/sec, latency 16034 usec
  Concurrency: 6, 363 infer/sec, latency 19990 usec
  Concurrency: 7, 369.6 infer/sec, latency 21382 usec
  Concurrency: 8, 426.6 infer/sec, latency 19526 usec

With eight concurrent requests the dynamic batcher allows Triton to
provide about 425 inferences per second without increasing latency
compared to not using the dynamic batcher.

You can also explicitly specify what batch sizes you would like the
dynamic batcher to prefer when creating batches. For example, to
indicate that you would like the dynamic batcher to prefer size 4
batches you can modify the model configuration like this (multiple
preferred sizes can be given but in this case we just have one)::

  dynamic_batching { preferred_batch_size: [ 4 ]}

Instead of having perf\_client collect data for a range of request
concurrency values we can instead use a simple rule that typically
applies when perf\_client is running on the same system as Triton. The
rule is that for maximum throughput set the request concurrency to be
2 * <preferred batch size> * <model instance count>. We will discuss
model instances :ref:`below <section-opt-model-instances>`, for now we
are working with one model instance. So for preferred-batch-size 4 we
want to run perf\_client with request concurrency of 2 * 4 * 1 = 8::

  $ perf_client -m resnet50_netdef --percentile=95 --concurrency-range 8
  ...
  Inferences/Second vs. Client p95 Batch Latency
  Concurrency: 8, 420.2 infer/sec, latency 19524 usec

.. _section-opt-model-instances:

Model Instances
^^^^^^^^^^^^^^^

Triton allows you to specify how many copies of each model you want to
make available for inferencing. By default you get one copy of each
model, but you can specify any number of instances in the model
configuration by using :ref:`section-instance-groups`. Typically,
having two instances of a model will improve performance because it
allows overlap of memory transfer operations (for example, CPU to/from
GPU) with inference compute. Multiple instances also improve GPU
utilization by allowing more inference work to be executed
simultaneously on the GPU. Smaller models may benefit from more than
two instances; you can use perf\_client to experiment.

To specify two instances of the resnet50\_netdef model: stop Triton,
remove any dynamic batching settings you may have previously added to
the model configuration (we discuss combining dynamic batcher and
multiple model instances below), add the following lines to the end of
the model configuration file, and then restart Triton::

  instance_group [ { count: 2 }]

Now run perf\_client using the same options as for the baseline::

  $ perf_client -m resnet50_netdef --percentile=95 --concurrency-range 1:4
  ...
  Inferences/Second vs. Client p95 Batch Latency
  Concurrency: 1, 129.4 infer/sec, latency 8434 usec
  Concurrency: 2, 257.4 infer/sec, latency 8126 usec
  Concurrency: 3, 289.6 infer/sec, latency 12621 usec
  Concurrency: 4, 287.8 infer/sec, latency 14296 usec

In this case having two instances of the model increases throughput
from about 200 inference per second to about 290 inferences per second
compared with one instance.

It is possible to enable both the dynamic batcher and multiple model
instances, for example::

  dynamic_batching { preferred_batch_size: [ 4 ] }
  instance_group [ { count: 2 }]

When we run perf\_client with the same options used for just the
dynamic batcher above::

  $ perf_client -m resnet50_netdef --percentile=95 --concurrency-range 8
  ...
  Inferences/Second vs. Client p95 Batch Latency
  Concurrency: 8, 409.2 infer/sec, latency 24284 usec

We see that two instances does not improve throughput or latency. This
occurs because for this model the dynamic batcher alone is capable of
fully utilizing the GPU and so adding additional model instances does
not provide any performance advantage. In general the benefit of the
dynamic batcher and multiple instances is model specific, so you
should experiment with perf\_client to determine the settings that
best satisfy your throughput and latency requirements.

Framework-Specific Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Triton has several optimization settings that apply to only a subset
of the supported model frameworks. These optimization settings are
controlled by the model configuration :ref:`optimization policy
<section-optimization-policy>`.

.. _section-opt-onnx-tensorrt:

ONNX with TensorRT Optimization
...............................

One especially powerful optimization is to use
:ref:`section-optimization-policy-tensorrt` in conjunction with an
ONNX model. As an example of TensorRT optimization applied to an ONNX
model, we will use an ONNX DenseNet model that you can obtain by
following the :ref:`section-quickstart`. As a baseline we use
perf\_client to determine the performance of the model using a `basic
model configuration that does not enable any performance features
<https://github.com/triton-inference-server/server/blob/master/docs/examples/model_repository/densenet_onnx/config.pbtxt>`_::

  $ perf_client -m densenet_onnx --percentile=95 --concurrency-range 1:4
  ...
  Inferences/Second vs. Client p95 Batch Latency
  Concurrency: 1, 113.2 infer/sec, latency 8939 usec
  Concurrency: 2, 138.2 infer/sec, latency 14548 usec
  Concurrency: 3, 137.2 infer/sec, latency 21947 usec
  Concurrency: 4, 136.8 infer/sec, latency 29661 usec

To enable TensorRT optimization for the model: stop Triton, add the
following lines to the end of the model configuration file, and then
restart Triton::

  optimization { execution_accelerators {
    gpu_execution_accelerator : [ { name : "tensorrt" } ]
  }}

As Triton starts you should check the console output and wait until
Triton prints the "Staring endpoints" message. ONNX model loading can
be significantly slower when TensorRT optimization is enabled.  Now
run perf\_client using the same options as for the baseline::

  $ perf_client -m densenet_onnx --percentile=95 --concurrency-range 1:4
  ...
  Inferences/Second vs. Client p95 Batch Latency
  Concurrency: 1, 190.6 infer/sec, latency 5384 usec
  Concurrency: 2, 273.8 infer/sec, latency 7347 usec
  Concurrency: 3, 272.2 infer/sec, latency 11046 usec
  Concurrency: 4, 266.8 infer/sec, latency 15089 usec

The TensorRT optimization provided 2x throughput improvement while
cutting latency in half. The benefit provided by TensorRT will vary
based on the model, but in general it can provide significant
performance improvement.

.. _section-opt-tensorflow-tensorrt:

TensorFlow with TensorRT Optimization
.....................................

TensorRT optimization applied to a TensorFlow model works similarly to
TensorRT and ONNX described above. To enable TensorRT optimization you
must set the model configuration appropriately. For TensorRT
optimization of TensorFlow models there are several options that you
can enable, including selection of the compute precision. For
example::

  optimization { execution_accelerators {
    gpu_execution_accelerator : [ {
      name : "tensorrt"
      parameters { key: "precision_mode" value: "FP16" }}]
  }}

The options are described in detail in the
:cpp:var:`ModelOptimizationPolicy
<nvidia::inferenceserver::ModelOptimizationPolicy>` section of the
model configuration protobuf.

As an example of TensorRT optimization applied to a TensorFlow model,
we will use a TensorFlow Inception model that you can obtain by
following the :ref:`section-quickstart`. As a baseline we use
perf\_client to determine the performance of the model using a `basic
model configuration that does not enable any performance features
<https://github.com/triton-inference-server/server/blob/master/docs/examples/model_repository/inception_graphdef/config.pbtxt>`_::

  $ perf_client -m inception_graphdef --percentile=95 --concurrency-range 1:4
  ...
  Inferences/Second vs. Client p95 Batch Latency
  Concurrency: 1, 105.6 infer/sec, latency 12865 usec
  Concurrency: 2, 120.6 infer/sec, latency 20888 usec
  Concurrency: 3, 122.8 infer/sec, latency 30308 usec
  Concurrency: 4, 123.4 infer/sec, latency 39465 usec

To enable TensorRT optimization for the model: stop Triton, add the
lines from above to the end of the model configuration file, and then
restart Triton. As Triton starts you should check the console output
and wait until the server prints the "Staring endpoints" message. Now
run perf\_client using the same options as for the baseline. Note that
the first run of perf\_client might timeout because the TensorRT
optimization is performed when the inference request is received and
may take significant time. If this happens just run perf\_client
again::

  $ perf_client -m inception_graphdef --percentile=95 --concurrency-range 1:4
  ...
  Inferences/Second vs. Client p95 Batch Latency
  Concurrency: 1, 172 infer/sec, latency 6912 usec
  Concurrency: 2, 265.2 infer/sec, latency 8905 usec
  Concurrency: 3, 254.2 infer/sec, latency 13506 usec
  Concurrency: 4, 257 infer/sec, latency 17715 usec

The TensorRT optimization provided 2x throughput improvement while
cutting latency in half. The benefit provided by TensorRT will vary
based on the model, but in general it can provide significant
performance improvement.

TensorFlow Automatic FP16 Optimization
......................................

TensorFlow has another option to provide FP16 optimization that can be
enabled in the model configuration. As with the TensorRT optimization
described above, you can enable this optimization by using the
gpu_execution_accelerator property::

  optimization { execution_accelerators {
    gpu_execution_accelerator : [ {
      name : "auto_mixed_precision"
  }}

The options are described in detail in the
:cpp:var:`ModelOptimizationPolicy
<nvidia::inferenceserver::ModelOptimizationPolicy>` section of the
model configuration protobuf.

You can follow the steps described above for TensorRT to see how this
automatic FP16 optimization benefits a model by using perf\_client to
evaluate the model's performance with and without the optimization.

.. include:: model_analyzer.rst
.. include:: perf_client.rst
.. include:: trace.rst
