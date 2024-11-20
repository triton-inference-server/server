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

# Metrics

Triton provides [Prometheus](https://prometheus.io/) metrics
indicating GPU and request statistics. By default, these metrics are
available at http://localhost:8002/metrics. The metrics are only
available by accessing the endpoint, and are not pushed or published
to any remote server. The metric format is plain text so you can view
them directly, for example:

```
$ curl localhost:8002/metrics
```

The `tritonserver --allow-metrics=false` option can be used to disable
all metric reporting, while the `--allow-gpu-metrics=false` and
`--allow-cpu-metrics=false` can be used to disable just the GPU and CPU
metrics respectively.

The `--metrics-port` option can be used to select a different port. By default,
Triton reuses the `--http-address` option for the metrics endpoint and binds the
http and metrics endpoints to the same specific address when http service is
enabled. If http service is not enabled, the metric address will bind to `0.0.0.0`
by default. To uniquely specify the metric endpoint, `--metrics-address` option
can be used. See the `tritonserver --help` output for more info on these CLI options.

To change the interval at which metrics are polled/updated, see the `--metrics-interval-ms` flag. Metrics that are updated "Per Request" are unaffected by this interval setting. This interval only applies to metrics that are designated as "Per Interval" in the tables of each section below:

- [Inference Request Metrics](#inference-request-metrics)
- [GPU Metrics](#gpu-metrics)
- [CPU Metrics](#cpu-metrics)
- [Pinned Memory Metrics](#pinned-memory-metrics)
- [Response Cache Metrics](#response-cache-metrics)
- [Custom Metrics](#custom-metrics)

## Inference Request Metrics

### Counts

For models that do not support batching, *Request Count*, *Inference
Count* and *Execution Count* will be equal, indicating that each
inference request is executed separately.

For models that support batching, the count metrics can be interpreted
to determine average batch size as *Inference Count* / *Execution
Count*. The count metrics are illustrated by the following examples:

* Client sends a single batch-1 inference request. *Request Count* =
  1, *Inference Count* = 1, *Execution Count* = 1.

* Client sends a single batch-8 inference request. *Request Count* =
  1, *Inference Count* = 8, *Execution Count* = 1.

* Client sends 2 requests: batch-1 and batch-8. Dynamic batcher is not
  enabled for the model. *Request Count* = 2, *Inference Count* = 9,
  *Execution Count* = 2.

* Client sends 2 requests: batch-1 and batch-1. Dynamic batcher is
  enabled for the model and the 2 requests are dynamically batched by
  the server. *Request Count* = 2, *Inference Count* = 2, *Execution
  Count* = 1.

* Client sends 2 requests: batch-1 and batch-8. Dynamic batcher is
  enabled for the model and the 2 requests are dynamically batched by
  the server. *Request Count* = 2, *Inference Count* = 9, *Execution
  Count* = 1.

|Category      |Metric          |Metric Name |Description                            |Granularity|Frequency    |
|--------------|----------------|------------|---------------------------|-----------|-------------|
|Count         |Success Count   |`nv_inference_request_success` |Number of successful inference requests received by Triton (each request is counted as 1, even if the request contains a batch) |Per model  |Per request  |
|              |Failure Count   |`nv_inference_request_failure` |Number of failed inference requests received by Triton (each request is counted as 1, even if the request contains a batch) |Per model  |Per request  |
|              |Inference Count |`nv_inference_count` |Number of inferences performed (a batch of "n" is counted as "n" inferences, does not include cached requests)|Per model|Per request|
|              |Execution Count |`nv_inference_exec_count` |Number of inference batch executions (see [Inference Request Metrics](#inference-request-metrics), does not include cached requests)|Per model|Per request|
|              |Pending Request Count |`nv_inference_pending_request_count` |Number of inference requests awaiting execution by a backend. This number is incremented when a request is enqueued to the server (`TRITONSERVER_ServerInferAsync`) and is decremented when a backend is about to start executing the request. More details can be found below. |Per model|Per request|

#### Failure Count Categories

| Failed Request Reason |Description |
|------------|------------|
| REJECTED  | Number of inference failures due to request timeout in the scheduler. |
| CANCELED  |  Number of inference failures due to request cancellation in the core. |
| BACKEND |  Number of inference failures during execution of requests in the backend/model. |
| OTHER  | Number of inference failures due to other uncategorized reasons in the core. |

> **Note**
>
> Ensemble failure metrics will reflect the failure counts of their composing models as well as the parent model, but currently do not capture the same granularity for the "reason" label and will default to the "OTHER" reason.
>
> For example, if EnsembleA contains ModelA, and ModelA experiences a failed request due to a queue/backlog timeout in the scheduler, ModelA will have a failed request metric reflecting `reason=REJECTED` and `count=1`.
> Additionally, EnsembleA will have a failed request metric reflecting `reason=OTHER` and `count=2`.
> The `count=2` reflects 1 from the internally failed request captured by ModelA, as well as 1 from the failed top-level request sent to EnsembleA by the user/client.
> The `reason=OTHER` reflects that fact that the ensemble doesn't currently capture the specific reason why
> ModelA's request failed at this time.

#### Pending Request Count (Queue Size) Per-Model

The *Pending Request Count* reflects the number of requests that have been
received by Triton core via `TRITONSERVER_InferAsync`, but have not yet
started execution by a backend model instance
(`TRITONBACKEND_ModelInstanceExecute`).

For all intents and purposes, the
"pending request count" and "queue size" per-model can be used
interchangeably, and the number reflected in the metric should
intuitively represent the number of requests that are not currently
being executed by any model instances. In simple terms, if you send a 100
requests to a model that can only handle 5 requests concurrently, then you
should see a pending count of 95 for that model in most cases.

For those interested in more technical details, the term "pending request count"
is a bit more accurate than "queue size" because Triton is highly configurable,
and there are many places in Triton that a request be considered pending rather
than a single queue. Some of the most common will be called out below:
- Default Scheduler backlogs any requests not currently executing.
  - Assuming 1 available model instance with the default scheduler settings,
    and 10 requests are sent in rapid succession.
  - The 1st request should be picked up for
    execution immediately, and the remaining 9 requests should be considered
    pending for this model, until the 1st request is finished. Afterwards, the
    next request should be picked up and the pending count should be decremented
    to 8, and so on until all requests are finished and the pending count is 0.
- Dynamic Batcher queue for dynamically creating batches from requests.
  - Assuming 1 available model instance with the dynamic batch scheduler
    configured with `max_batch_size: 4` and a sufficiently large
    `max_queue_delay_microseconds` (or queue of requests),
    and 10 requests are sent in rapid succession.
  - The first 4 requests, or as large of a batch the scheduler could form,
    should be picked up for execution immediately, and the remaining 6 requests
    should be considered pending. After the batch finishes, the next batch
    should be picked up, decrementing the pending count again to 2 pending.
    Then finally since only 2 requests remain, the final 2 requests will be
    batched and picked up by the backend, decrementing the pending count to 0.
- Sequence Batcher queues and backlogs for ongoing sequence requests, some may
  be assigned sequence slots, some may not.
  - Sequence Batchers of both strategies (direct and oldest) will have pending
    counts that generally follow the same trend as the dynamic batching
    description above. The sequence batchers will immediately execute as many
    requests in a batch as it can based on the model/scheduler config settings,
    and any further requests will be considered pending until the previous batch
    finishes and the next batch can start.
- Rate Limiter queues for prepared batches of requests.
  - When rate limiting is enabled, requests can be held back from execution
    to satisfy the rate limit constraints that were configured.

There are some places where a request would not be considered pending:
- Ensemble Scheduler
  - The Ensemble Scheduler almost immediately enqueues any requests it receives
    into the composing model schedulers at the first step in the ensemble.
    Therefore, the requests could be considered pending by the composing model
    scheduler's, however from the ensemble's perspective, these requests have been
    scheduled.
- Frontends (HTTP/GRPC Servers)
  - Any requests sent from a client to a frontend server in-front of Triton
    may spend some time in the corresponding server's code mapping
    protocol-specific metadata to Triton metadata. Though this time is
    generally brief, it will not be considered pending from Triton's
    perspective until Triton core has received the request from the frontend.

### Latencies

Starting in 23.04, Triton exposes the ability to choose the types of metrics
that are published through the `--metrics-config` CLI options.

#### Counters

By default, the following
[Counter](https://prometheus.io/docs/concepts/metric_types/#counter)
metrics are used for latencies:

|Category      |Metric          |Metric Name |Description                            |Granularity|Frequency    |
|--------------|----------------|------------|---------------------------|-----------|-------------|
|Latency       |Request Time    |`nv_inference_request_duration_us` |Cumulative end-to-end inference request handling time (includes cached requests) |Per model  |Per request  |
|              |Queue Time      |`nv_inference_queue_duration_us` |Cumulative time requests spend waiting in the scheduling queue (includes cached requests) |Per model  |Per request  |
|              |Compute Input Time|`nv_inference_compute_input_duration_us` |Cumulative time requests spend processing inference inputs (in the framework backend, does not include cached requests)     |Per model  |Per request  |
|              |Compute Time    |`nv_inference_compute_infer_duration_us` |Cumulative time requests spend executing the inference model (in the framework backend, does not include cached requests)     |Per model  |Per request  |
|              |Compute Output Time|`nv_inference_compute_output_duration_us` |Cumulative time requests spend processing inference outputs (in the framework backend, does not include cached requests)     |Per model  |Per request  |

To disable these metrics specifically, you can set `--metrics-config counter_latencies=false`

#### Histograms

> **Note**
>
> The following Histogram feature is experimental for the time being and may be
> subject to change based on user feedback.

By default, the following
[Histogram](https://prometheus.io/docs/concepts/metric_types/#histogram)
metrics are used for latencies:

|Category      |Metric          |Metric Name |Description                |Granularity|Frequency    |Model Type
|--------------|----------------|------------|---------------------------|-----------|-------------|-------------|
|Latency       |Request to First Response Time    |`nv_inference_first_response_histogram_ms` |Histogram of end-to-end inference request to the first response time |Per model  |Per request  | Decoupled |

To enable these metrics specifically, you can set `--metrics-config histogram_latencies=true`

Each histogram above is composed of several sub-metrics. For each histogram
metric, there is a set of `le` (less than or equal to) thresholds tracking
the counter for each bucket. Additionally, there are `_count` and `_sum`
metrics that aggregate the count and observed values for each. For example,
see the following information exposed by the "Time to First Response" histogram
metrics:
```
# HELP nv_first_response_histogram_ms Duration from request to first response in milliseconds
# TYPE nv_first_response_histogram_ms histogram
nv_inference_first_response_histogram_ms_count{model="my_model",version="1"} 37
nv_inference_first_response_histogram_ms_sum{model="my_model",version="1"} 10771
nv_inference_first_response_histogram_ms{model="my_model",version="1", le="100"} 8
nv_inference_first_response_histogram_ms{model="my_model",version="1", le="500"} 30
nv_inference_first_response_histogram_ms{model="my_model",version="1", le="2000"} 36
nv_inference_first_response_histogram_ms{model="my_model",version="1", le="5000"} 37
nv_inference_first_response_histogram_ms{model="my_model",version="1", le="+Inf"} 37
```

Triton initializes histograms with default buckets for each, as shown above.
Buckets can be overridden per family by specifying `model_metrics` in the
model configuration. For example:
```
// config.pbtxt
model_metrics {
  metric_control: [
    {
      metric_identifier: {
        family: "nv_inference_first_response_histogram_ms"
      }
      histogram_options: {
        buckets: [ 1, 2, 4, 8 ]
      }
    }
  ]
}
```

> **Note**
>
> To apply changes to metric options dynamically, the model must be completely
> unloaded and then reloaded for the updates to take effect.

Currently, the following histogram families support custom buckets.
```
nv_inference_first_response_histogram_ms  // Time to First Response
```

#### Summaries

> **Note**
>
> The following Summary feature is experimental for the time being and may be
> subject to change based on user feedback.

To get configurable quantiles over a sliding time window, Triton supports
a set a [Summary](https://prometheus.io/docs/concepts/metric_types/#summary)
metrics for latencies as well. These metrics are disabled by default, but can
be enabled by setting `--metrics-config summary_latencies=true`.

For more information on how the quantiles are calculated, see
[this explanation](https://grafana.com/blog/2022/03/01/how-summary-metrics-work-in-prometheus/).

The following summary metrics are available:

|Category      |Metric          |Metric Name |Description                            |Granularity|Frequency    |
|--------------|----------------|------------|---------------------------|-----------|-------------|
|Latency       |Request Time    |`nv_inference_request_summary_us` |Summary of end-to-end inference request handling times (includes cached requests) |Per model  |Per request  |
|              |Queue Time      |`nv_inference_queue_summary_us` |Summary of time requests spend waiting in the scheduling queue (includes cached requests) |Per model  |Per request  |
|              |Compute Input Time|`nv_inference_compute_input_summary_us` |Summary time requests spend processing inference inputs (in the framework backend, does not include cached requests)     |Per model  |Per request  |
|              |Compute Time    |`nv_inference_compute_infer_summary_us` |Summary of time requests spend executing the inference model (in the framework backend, does not include cached requests)     |Per model  |Per request  |
|              |Compute Output Time|`nv_inference_compute_output_summary_us` |Summary of time requests spend processing inference outputs (in the framework backend, does not include cached requests)     |Per model  |Per request  |

Each summary above is actually composed of several sub-metrics. For each
metric, there is a set of `quantile` metrics tracking the latency for each
quantile. Additionally, there are `_count` and `_sum` metrics that aggregate
the count and observed values for each. For example, see the following
information exposed by the Inference Queue Summary metrics:
```
# HELP nv_inference_queue_summary_us Summary of inference queuing duration in microseconds (includes cached requests)
# TYPE nv_inference_queue_summary_us summary
nv_inference_queue_summary_us_count{model="my_model",version="1"} 161
nv_inference_queue_summary_us_sum{model="my_model",version="1"} 11110
nv_inference_queue_summary_us{model="my_model",version="1",quantile="0.5"} 55
nv_inference_queue_summary_us{model="my_model",version="1",quantile="0.9"} 97
nv_inference_queue_summary_us{model="my_model",version="1",quantile="0.95"} 98
nv_inference_queue_summary_us{model="my_model",version="1",quantile="0.99"} 101
nv_inference_queue_summary_us{model="my_model",version="1",quantile="0.999"} 101
```

The count and sum for the summary above show that stats have been recorded for
161 requests, and took a combined total of 11110 microseconds. The `_count` and
`_sum` of a summary should generally match the counter metric equivalents when
applicable, such as:
```
nv_inference_request_success{model="my_model",version="1"} 161
nv_inference_queue_duration_us{model="my_model",version="1"} 11110
```

Triton has a set of default quantiles to track, as shown above. To set
custom quantiles, you can use the `--metrics-config` CLI option. The format is:
```
tritonserver --metrics-config summary_quantiles="<quantile1>:<error1>,...,<quantileN>:<errorN>"`
```

For example:
```
tritonserver --metrics-config summary_quantiles="0.5:0.05,0.9:0.01,0.95:0.001,0.99:0.001"`
```

To better understand the setting of error values for computing each quantile, see the
[best practices for histograms and summaries](https://prometheus.io/docs/practices/histograms/#histograms-and-summaries).


## GPU Metrics

GPU metrics are collected through the use of [DCGM](https://developer.nvidia.com/dcgm).
Collection of GPU metrics can be toggled with the `--allow-gpu-metrics` CLI flag.
If building Triton locally, the `TRITON_ENABLE_METRICS_GPU` CMake build flag can be used to toggle building the relevant code entirely.

|Category        |Metric            |Metric Name                 |Description                                            |Granularity|Frequency    |
|----------------|------------------|----------------------------|-------------------------------------------------------|-----------|-------------|
|GPU Utilization |Power Usage       |`nv_gpu_power_usage`        |GPU instantaneous power, in watts                      |Per GPU    |Per interval |
|                |Power Limit       |`nv_gpu_power_limit`        |Maximum GPU power limit, in watts                      |Per GPU    |Per interval |
|                |Energy Consumption|`nv_energy_consumption`     |GPU energy consumption since Triton started, in joules |Per GPU    |Per interval |
|                |GPU Utilization   |`nv_gpu_utilization`        |GPU utilization rate (0.0 - 1.0)                       |Per GPU    |Per interval |
|GPU Memory      |GPU Total Memory  |`nv_gpu_memory_total_bytes` |Total GPU memory, in bytes                             |Per GPU    |Per interval |
|                |GPU Used Memory   |`nv_gpu_memory_used_bytes`  |Used GPU memory, in bytes                              |Per GPU    |Per interval |


## CPU Metrics

Collection of CPU metrics can be toggled with the `--allow-cpu-metrics` CLI flag.
If building Triton locally, the `TRITON_ENABLE_METRICS_CPU` CMake build flag can be used to toggle building the relevant code entirely.

> **Note**
>
> CPU Metrics are currently only supported on Linux.
> They collect information from the [/proc filesystem](https://www.kernel.org/doc/html/latest/filesystems/proc.html) such as `/proc/stat` and `/proc/meminfo`.

|Category      |Metric          |Metric Name |Description                            |Granularity|Frequency    |
|--------------|----------------|------------|---------------------------|-----------|-------------|
|CPU Utilization | CPU Utilization | `nv_cpu_utilization` | Total CPU utilization rate [0.0 - 1.0] | Aggregated across all cores since last interval | Per interval |
|CPU Memory      | CPU Total Memory | `nv_cpu_memory_total_bytes` | Total CPU memory (RAM), in bytes | System-wide | Per interval |
|                | CPU Used Memory | `nv_cpu_memory_used_bytes` | Used CPU memory (RAM), in bytes | System-wide | Per interval |

## Pinned Memory Metrics

Starting in 24.01, Triton offers Pinned Memory metrics to monitor the utilization of the Pinned Memory pool.

|Category        |Metric            |Metric Name                 |Description                                            |Granularity|Frequency    |
|----------------|------------------|----------------------------|-------------------------------------------------------|-----------|-------------|
|Pinned Memory   |Total Pinned memory |`nv_pinned_memory_pool_total_bytes`        |Total Pinned memory, in bytes                      |All models    |Per interval |
|                |Used Pinned memory |`nv_pinned_memory_pool_used_bytes`        |Used Pinned memory, in bytes                      |All models    |Per interval |

## Response Cache Metrics

Cache metrics can be reported in two ways:

1. A base set of cache metrics will be reported
by Triton directly, such as the cache hit/miss counts and durations described
below.

2. As of 23.03, additional cache metrics may be reported depending on the
[cache implementation](response_cache.md#cache-implementations)
being used through Triton's [Metrics API](#custom-metrics).

### Triton-reported Response Cache Metrics

Compute latency metrics in the
[Inference Request Metrics table](#inference-request-metrics) above are
calculated for the time spent in model inference backends. If the response
cache is enabled for a given model (see [Response Cache](response_cache.md)
docs for more info), total inference times may be affected by response cache
lookup times.

On cache hits, "Cache Hit Time" indicates the time spent looking up the
response, and "Compute Input Time" /  "Compute Time" / "Compute Output Time"
are not recorded.

On cache misses, "Cache Miss Time" indicates the time spent looking up
the request hash and inserting the computed output tensor data into the cache.
Otherwise, "Compute Input Time" /  "Compute Time" / "Compute Output Time" will
be recorded as usual.

|Category      |Metric          |Metric Name |Description                            |Granularity|Frequency    |
|--------------|----------------|------------|---------------------------|-----------|-------------|
|Count         |Cache Hit Count |`nv_cache_num_hits_per_model` |Number of response cache hits per model |Per model |Per request |
|              |Cache Miss Count |`nv_cache_num_misses_per_model` |Number of response cache misses per model |Per model |Per request |
|Latency       |Cache Hit Time |`nv_cache_hit_duration_per_model` |Cumulative time requests spend retrieving a cached response per model on cache hits (microseconds) |Per model |Per request |
|              |Cache Miss Time |`nv_cache_miss_duration_per_model` |Cumulative time requests spend looking up and inserting responses into the cache on a cache miss (microseconds) |Per model |Per request |

Similar to the Summaries section above for Inference Request Metrics, the
per-model cache hit/miss latency metrics also support Summaries.

> **Note**
>
> For models with response caching enabled, the inference request **summary** metric
> is currently disabled. This is due to extra time spent internally on cache
> management that wouldn't be reflected correctly in the end to end request time.
> Other summary metrics are unaffected.

## Custom Metrics

Triton exposes a C API to allow users and backends to register and collect
custom metrics with the existing Triton metrics endpoint. The user takes the
ownership of the custom metrics created through the APIs and must manage their
lifetime following the API documentation.

The
[identity_backend](https://github.com/triton-inference-server/identity_backend/blob/main/README.md#custom-metric-example)
demonstrates a practical example of adding a custom metric to a backend.

Further documentation can be found in the `TRITONSERVER_MetricFamily*` and
`TRITONSERVER_Metric*` API annotations in
[tritonserver.h](https://github.com/triton-inference-server/core/blob/main/include/triton/core/tritonserver.h).

### TensorRT-LLM Backend Metrics

The TRT-LLM backend uses the custom metrics API to track and expose specific metrics about
LLMs, KV Cache, and Inflight Batching to Triton:
https://github.com/triton-inference-server/tensorrtllm_backend?tab=readme-ov-file#triton-metrics

### vLLM Backend Metrics

The vLLM backend uses the custom metrics API to track and expose specific metrics about
LLMs to Triton:
https://github.com/triton-inference-server/vllm_backend?tab=readme-ov-file#triton-metrics
