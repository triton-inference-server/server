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


# Batchers

## Dynamic Batcher

Dynamic batching is a feature of Triton that allows inference requests
to be combined by the server, so that a batch is created
dynamically. Creating a batch of requests typically results in
increased throughput. The dynamic batcher should be used for
[stateless models](architecture.md#stateless-models). The dynamically created
batches are distributed to all [model instances](model_configuration.md#instance-groups)
configured for the model.

Dynamic batching is enabled and configured independently for each
model using the *ModelDynamicBatching* property in the model
configuration. These settings control the preferred size(s) of the
dynamically created batches, the maximum time that requests can be
delayed in the scheduler to allow other requests to join the dynamic
batch, and queue properties such a queue size, priorities, and
time-outs. Refer to
[this guide](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_2-improving_resource_utilization#what-is-dynamic-batching)
for a more detailed example of dynamic batching.

### Recommended Configuration Process

The individual settings are described in detail below. The following
steps are the recommended process for tuning the dynamic batcher for
each model. It is also possible to use the [Model
Analyzer](model_analyzer.md) to automatically search across different
dynamic batcher configurations.

* Decide on a [maximum batch size](#maximum-batch-size) for the model.

* Add the following to the model configuration to enable the dynamic
  batcher with all default settings. By default the dynamic batcher
  will create batches as large as possible up to the maximum batch
  size and will not [delay](#delayed-batching) when forming batches.

```
  dynamic_batching { }
```

* Use the
  [Performance Analyzer](https://github.com/triton-inference-server/client/blob/main/src/c++/perf_analyzer/README.md)
  to determine the latency and throughput provided by the default dynamic
  batcher configuration.

* If the default configuration results in latency values that are
  within your latency budget, try one or both of the following to
  trade off increased latency for increased throughput:

  * Increase maximum batch size.

  * Set [batch delay](#delayed-batching) to a non-zero value. Try
    increasing delay values until the latency budget is exceeded to
    see the impact on throughput.

* [Preferred batch sizes](#preferred-batch-sizes) should not be used
  for most models. A preferred batch size(s) should only be configured
  if that batch size results in significantly higher performance than
  other batch sizes.

### Preferred Batch Sizes

The *preferred_batch_size* property indicates the batch sizes that the
dynamic batcher should attempt to create. For most models,
*preferred_batch_size* should not be specified, as described in
[Recommended Configuration
Process](#recommended-configuration-process). An exception is TensorRT
models that specify multiple optimization profiles for different batch
sizes. In this case, because some optimization profiles may give
significant performance improvement compared to others, it may make
sense to use *preferred_batch_size* for the batch sizes supported by
those higher-performance optimization profiles.

The following example shows the configuration that enables dynamic
batching with preferred batch sizes of 4 and 8.

```
  dynamic_batching {
    preferred_batch_size: [ 4, 8 ]
  }
```

When a model instance becomes available for inferencing, the dynamic
batcher will attempt to create batches from the requests that are
available in the scheduler. Requests are added to the batch in the
order the requests were received. If the dynamic batcher can form a
batch of a preferred size(s) it will create a batch of the largest
possible preferred size and send it for inferencing. If the dynamic
batcher cannot form a batch of a preferred size (or if the dynamic
batcher is not configured with any preferred batch sizes), it will
send a batch of the largest size possible that is less than the
maximum batch size allowed by the model (but see the following section
for the delay option that changes this behavior).

The size of generated batches can be examined in aggregate using
[count metrics](metrics.md#inference-request-metrics).

### Delayed Batching

The dynamic batcher can be configured to allow requests to be delayed
for a limited time in the scheduler to allow other requests to join
the dynamic batch. For example, the following configuration sets the
maximum delay time of 100 microseconds for a request.

```
  dynamic_batching {
    max_queue_delay_microseconds: 100
  }
```

The *max_queue_delay_microseconds* property setting changes the
dynamic batcher behavior when a maximum size (or preferred size) batch
cannot be created. When a batch of a maximum or preferred size cannot
be created from the available requests, the dynamic batcher will delay
sending the batch as long as no request is delayed longer than the
configured *max_queue_delay_microseconds* value. If a new request
arrives during this delay and allows the dynamic batcher to form a
batch of a maximum or preferred batch size, then that batch is sent
immediately for inferencing. If the delay expires the dynamic batcher
sends the batch as is, even though it is not a maximum or preferred
size.

### Preserve Ordering

The *preserve_ordering* property is used to force all responses to be
returned in the same order as requests were received. See the
[protobuf
documentation](https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto)
for details.

### Priority Levels

By default the dynamic batcher maintains a single queue that holds all
inference requests for a model. The requests are processed and batched
in order.  The *priority_levels* property can be used to create
multiple priority levels within the dynamic batcher so that requests
with higher priority are allowed to bypass requests with lower
priority. Requests at the same priority level are processed in
order. Inference requests that do not set a priority are scheduled
using the *default_priority_level* property.

### Queue Policy

The dynamic batcher provides several settings that control how
requests are queued for batching.

When *priority_levels* is not defined, the *ModelQueuePolicy* for the
single queue can be set with *default_queue_policy*.  When
*priority_levels* is defined, each priority level can have a different
*ModelQueuePolicy* as specified by *default_queue_policy* and *priority_queue_policy*.

The *ModelQueuePolicy* property allows a maximum queue size to be set
using the *max_queue_size*. The *timeout_action*,
*default_timeout_microseconds* and *allow_timeout_override* settings
allow the queue to be configured so that individual requests are
rejected or deferred if their time in the queue exceeds a specified
timeout.

## Custom Batching

You can set custom batching rules that work _in addition to_ the specified behavior of the dynamic batcher.
To do so, you would implement five functions in [tritonbackend.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonbackend.h)
and create a shared library. These functions are described below.

| Function | Description|
| :--          |   :--           |
| TRITONBACKEND_ModelBatchIncludeRequest | Determines whether a request should be included in the current batch |
| TRITONBACKEND_ModelBatchInitialize | Initializes a record-keeping data structure for a new batch |
| TRITONBACKEND_ModelBatchFinalize | Deallocates the record-keeping data structure after a batch is formed |
| TRITONBACKEND_ModelBatcherInitialize | Initializes a read-only data structure for use with all batches |
| TRITONBACKEND_ModelBatcherFinalize | Deallocates the read-only data structure after the model is unloaded |

The path to the shared library can be passed into the model configuration via the parameter
`TRITON_BATCH_STRATEGY_PATH`. If not provided, the dynamic batcher will look for a custom
batching strategy named batchstrategy.so in the model version, model, and backend directories,
in that order. If found, it will load it. This lets you easily share a custom batching strategy
among all models using the same backend.

For a tutorial of how to create and use a custom batching library, please see the
[backend examples directory](https://github.com/triton-inference-server/backend/tree/main/examples#volume-batching).

## Sequence Batcher

Like the dynamic batcher, the sequence batcher combines non-batched
inference requests, so that a batch is created dynamically. Unlike the
dynamic batcher, the sequence batcher should be used for
[stateful models](architecture.md#stateful-models) where a sequence of
inference requests must be routed to the same model instance. The
dynamically created batches are distributed to all [model
instances](#instance-groups) configured for the model.

Sequence batching is enabled and configured independently for each
model using the *ModelSequenceBatching* property in the model
configuration. These settings control the sequence timeout as well as
configuring how Triton will send control signals to the model
indicating sequence start, end, ready and correlation ID. See
[Stateful Models](architecture.md#stateful-models) for more
information and examples.

## Iterative Sequences

> [!NOTE]
> Iterative sequences are *provisional* and likely to change in future versions.
The sequence batcher supports stateful execution of "iterative
sequences" where a single request is processed over a number of
scheduling iterations. "Iterative sequences" enable the scheduler to
batch multiple inflight requests at each step and allow the model or
backend to complete a request at any iteration.

For models and backends that support "iterative sequences", users can
enable support in the sequence batcher by specifying:

```
  sequence_batching {
    iterative_sequence: true
  }
```

An "iterative sequence" refers to stateful models that iteratively
process a single request until a complete response is generated.  When
iterative sequence is enabled, the sequence scheduler will expect a
single incoming request to initiate the sequence. Backends that
support iterative sequences can then yield back to the sequence
batcher to reschedule the request for further execution in a future
batch.

Because only one request is used to represent the "iterative
sequence", the user doesn't need to set [control
inputs](architecture.md#control-inputs) mentioned in the previous
section. They will be filled internally by the scheduler.

"Iterative sequences" can be [decoupled](architecture.md#decoupled) where more than
one response can be generated during execution or non-decoupled where
a single response is generated when the full response is complete.

The main advantage of "iterative sequences" is the ability to use
Triton's native batching capabilities to form batches of requests at
different iteration stages without having to maintain additional state
in the backend. Typically batches executed by backends are completed
in the same execution which can waste resources if the execution of
one of the requests in the batch takes much longer than the rest. With
"iterative sequences", processing for each request in a batch can be
broken down into multiple iterations and a backend can start
processing new requests as soon as any request is complete.

### Continuous/Inflight Batching with Iterative Sequences

Continuous batching, iteration level batching, and inflight batching
are terms used in large language model (LLM) inferencing to describe
batching strategies that form batches of requests at each iteration
step. By forming batches "continuously" inference servers can increase
throughput by reusing batch slots as soon as they are free without
waiting for all requests in a batch to complete.

As the number of steps required to process a request can vary
significantly, batching existing requests and new requests continuously
can have a significant improvement on throughput and latency.

To achieve inflight batching with iterative sequences, the backend
should break request processing into a number of steps, where each
step corresponds to one Triton model instance execution. At the end of
each step, the model instance will release requests that have been
completed and reschedule requests that are still inflight. Triton will
then form and schedule the next batch of requests that mixes new and
rescheduled requests.