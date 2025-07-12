<!--
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Custom Metrics Example

In this section we demonstrate an end-to-end example for
[Custom Metrics API](../../README.md#custom-metrics) in Python backend. The
[model repository](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md)
should contain [custom_metrics](./model.py) model. The
[custom_metrics](./model.py) model uses
[Custom Metrics API](../../README.md#custom-metrics) to register and collect
custom metrics.

## Deploying the Custom Metrics Models

1. Create the model repository:

```console
mkdir -p models/custom_metrics/1/

# Copy the Python models
cp examples/custom_metrics/model.py models/custom_metrics/1/model.py
cp examples/custom_metrics/config.pbtxt models/custom_metrics/config.pbtxt
```

2. Start the tritonserver:

```
tritonserver --model-repository `pwd`/models
```

3. Send inference requests to server:

```
python3 examples/custom_metrics/client.py
```

You should see an output similar to the output below in the client terminal:

```
custom_metrics example: found pattern '# HELP requests_process_latency_ns Cumulative time spent processing requests' in metrics
custom_metrics example: found pattern '# TYPE requests_process_latency_ns counter' in metrics
custom_metrics example: found pattern 'requests_process_latency_ns{model="custom_metrics",version="1"}' in metrics
PASS: custom_metrics
```

In the terminal that runs Triton Server, you should see an output similar to
the output below:
```
Cumulative requests processing latency: 223406.0
```

The [model.py](./model.py) model file is heavily commented with
explanations about each of the function calls.

### Explanation of the Client Output

The [client.py](./client.py) sends a HTTP request with url
`http://localhost:8002/metrics` to fetch the metrics from Triton server. The
client then verifies if the custom metrics added in the model file are
correctly reported.
