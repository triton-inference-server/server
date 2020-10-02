<!--
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

The tritonserver --allow-metrics=false option can be used to disable
all metric reporting and --allow-gpu-metrics=false can be used to
disable just the GPU Utilization and GPU Memory metrics. The
--metrics-port option can be used to select a different port.

The following table describes the available metrics.

|Category      |Metric          |Description                            |Granularity|Frequency    |
|--------------|----------------|---------------------------------------|-----------|-------------|
|GPU Utilization |Power Usage   |GPU instantaneous power                |Per GPU    |Per second   |
|              |Power Limit     |Maximum GPU power limit                |Per GPU    |Per second   |
|              |Energy Consumption|GPU energy consumption in joules since Triton started|Per GPU|Per second|
|              |GPU Utilization |GPU utilization rate (0.0 - 1.0)       |Per GPU    |Per second   |
|GPU Memory    |GPU Total Memory|Total GPU memory, in bytes             |Per GPU    |Per second   |
|              |GPU Used Memory |Used GPU memory, in bytes              |Per GPU    |Per second   |
|Count         |Request Count   |Number of inference requests           |Per model  |Per request  |
|              |Execution Count |Number of inference executions (request count / execution count = average dynamic batch size)|Per model|Per request|
|              |Inference Count |Number of inferences performed (one request counts as "batch size" inferences)|Per model|Per request|
|Latency       |Request Time    |Cumulative end-to-end inference request handling time    |Per model  |Per request  |
|              |Queue Time      |Cumulative time requests spend waiting in the scheduling queue     |Per model  |Per request  |
|              |Compute Input Time|Cumulative time requests spend processing inference inputs (in the framework backend)     |Per model  |Per request  |
|              |Compute Time    |Cumulative time requests spend executing the inference model (in the framework backend)     |Per model  |Per request  |
|              |Compute Output Time|Cumulative time requests spend processing inference outputs (in the framework backend)     |Per model  |Per request  |
