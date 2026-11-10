<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Metrics catalog (port 8002)

```
curl -s http://<host>:8002/metrics | grep nv_
```

## Per-model counters

| Metric | Meaning |
|---|---|
| `nv_inference_count` | Total requests served (success + failure). |
| `nv_inference_request_success` | Requests completed successfully. |
| `nv_inference_request_failure{reason=...}` | Broken out by `BACKEND`, `REJECTED`, `CANCELED`, `OTHER`. |
| `nv_inference_exec_count` | Backend invocations. Ratio `nv_inference_count / nv_inference_exec_count` is batcher efficiency. Higher means more coalescing. |
| `nv_inference_pending_request_count` | In-flight requests right now. |
| `nv_model_load_duration_secs` | How long the model took to load. |

## Per-model latency

| Metric | Meaning |
|---|---|
| `nv_inference_request_duration_us` | End-to-end wall-clock, including queue. |
| `nv_inference_queue_duration_us` | Time waiting for the batcher. High queue + low compute means the batcher is misconfigured. |
| `nv_inference_compute_input_duration_us` | Pre-compute (input assembly). |
| `nv_inference_compute_infer_duration_us` | Compute (the backend itself). |
| `nv_inference_compute_output_duration_us` | Post-compute (output assembly). |

## Per-model cache (only when `response_cache` is enabled on the model AND `--cache-config` is set on the server)

| Metric | Meaning |
|---|---|
| `nv_cache_num_lookups_per_model` | Cache lookups attempted. |
| `nv_cache_num_hits_per_model` | Lookups that hit. |
| `nv_cache_num_misses_per_model` | Lookups that missed. |

## Per-GPU

| Metric | Meaning |
|---|---|
| `nv_gpu_utilization{gpu_uuid=...}` | GPU utilization, 0.0 to 1.0. |
| `nv_gpu_memory_used_bytes{gpu_uuid=...}` | GPU memory used. |
| `nv_gpu_memory_total_bytes{gpu_uuid=...}` | GPU memory total. |
| `nv_gpu_power_usage{gpu_uuid=...}` | Current draw in watts. |

## Reading the ratios

- **`request_count / exec_count`**: batcher efficiency. 1.0 means every request was its own batch (batcher not helping). 10.0 means the batcher is coalescing well.
- **`queue_duration / request_duration`**: how much of request latency is queue time. High means the batcher's `max_queue_delay_microseconds` is too long, or throughput is pressing the scheduler.
- **`cache_hits / cache_lookups`**: hit rate. Only meaningful for deterministic workloads.

## Scraping

Format is Prometheus exposition. Standard Prometheus + Grafana stack picks it up directly. Scrape interval of 15s is typical.
