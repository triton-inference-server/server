<!--
# Copyright 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Triton Server Trace

Triton includes that capability to generate a detailed trace for
individual inference requests. Tracing is enable by command-line
arguments when running the tritonserver executable.

Starting in 23.04 release, Triton has one flag `--trace-config` to specify 
global or trace mode specific configuration setting. The format of this flag 
is `--trace-config=<mode>,<setting>=<value>`, where `<mode>` 
is either `triton` or `opentelemetry`. By default, the trace mode is set to `triton`,
and the server will use Triton's trace APIs. For `opentelemetry` mode, 
the server will use the [OpenTelemetry's APIs](#opentelemetry-trace-support) to generate, collect and export 
traces for individual inference requests.

To specify global trace settings (level, rate, count, or mode), 
the format is `--trace-config=<setting>=<value>`.

An example usage, which invokes Triton's trace APIs:

```
$ tritonserver \
    --trace-config=triton,file=/tmp/trace.json \
    --trace-config=triton,log-frequency=50 \
    --trace-config=rate=100 \
    --trace-config=level=TIMESTAMPS \
    --trace-config=count=100 ...
```

## Trace Settings
### Global Settings
The following table shows available global trace settings to pass to `--trace-config`

| Setting     | Default Value |  Description                                                                                                                           |
|:------------|:--------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `rate`      | 1000          | Specifies the sampling rate. The same as deprecated `--trace-rate`. In the above example every 100-th inference request will be traced.|
| `level`     | OFF           | Indicates the level of trace detail that should be collected and may be specified multiple times to trace multiple informations. The same as deprecated `--trace-level`. Choices are `TIMESTAMPS` and `TENSORS`. Note that `opentelemetry` mode does not currently support `TENSORS` level. |
| `count`     | -1            | Specifies the remaining number of traces to be collected. In the example Triton will stop tracing requests after 100 traces are collected. The same as deprecated `--trace-count`. |
| `mode`      | triton        | Specifies which trace APIs to use for collecting traces. The choices are "triton" or "opentelemetry". |

### Triton trace APIs settings

The following table shows available Triton trace APIs settings for `--trace-config=triton,<setting>=<value>`.

| Setting     | Default Value |  Description                                                                                                                           |
|:------------|:--------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `file`      |     -         | Indicates where the trace output should be written. The same as deprecated `--trace-file`.|
| `log-frequency`| 0          | Specifies the rate that the traces are written to file. In the example Triton will log to file for every 50 traces collected. The same as deprecated `--trace-log-frequency`|

In addition to configure trace settings for Triton trace APIs 
in command line arguments, The user may
modify the trace setting when Triton server
is running via the trace APIs, more information can be found in [trace
protocol](../protocol/extension_trace.md). This option is currently not supported,
when trace mode is set to `opentelemetry`.

**Note**: the following flags are **depricated**:

The `--trace-file` option indicates where the trace output should be
written. The `--trace-rate` option specifies the sampling rate. In
this example every 100-th inference request will be traced. The
`--trace-level` option indicates the level of trace detail that should
be collected. `--trace-level` option may be specified multiple times to 
trace multiple informations. The `--trace-log-frequency` option specifies the
rate that the traces are written to file. In this example Triton will log to
file for every 50 traces collected. The `--trace-count` option specifies the
remaining number of traces to be collected. In this example Triton will stop
tracing more requests after 100 traces are collected.  Use the `--help` option
to get more information.

## Supported Trace Level Option

- `TIMESTAMPS`: Tracing execution timestamps of each request.
- `TENSORS`: Tracing input and output tensors during the execution. 

## JSON Trace Output

The trace output is a JSON file with the following schema.

```
[
  {
    "model_name": $string,
    "model_version": $number,
    "id": $number,
    "request_id": $string,
    "parent_id": $number
  },
  {
    "id": $number,
    "timestamps": [
      { "name" : $string, "ns" : $number }
    ]
  },
  {
    "id": $number
    "activity": $string,
    "tensor":{
      "name": $string,
      "data": $string,
      "shape": $string,
      "dtype": $string
    }
  },
  ...
]
```

Each trace is assigned a "id", which indicates the model name and 
version of the inference request. If the trace is from a
model run as part of an ensemble, the "parent_id" will indicate the
"id" of the containing ensemble.
For example:
```
[
  {
    "id": 1,
    "model_name": "simple",
    "model_version": 1
  },
  ...
]
```

Each `TIMESTAMPS` trace will have one or more "timestamps" with 
each timestamp having a name and the timestamp in nanoseconds ("ns"). 
For example:

```
[
  {"id": 1, "timestamps": [{ "name": "HTTP_RECV_START", "ns": 2356425054587444 }] },
  {"id": 1, "timestamps": [{ "name": "HTTP_RECV_END", "ns": 2356425054632308 }] },
  {"id": 1, "timestamps": [{ "name": "REQUEST_START", "ns": 2356425054785863 }] },
  {"id": 1, "timestamps": [{ "name": "QUEUE_START", "ns": 2356425054791517 }] },
  {"id": 1, "timestamps": [{ "name": "INFER_RESPONSE_COMPLETE", "ns": 2356425057587919 }] },
  {"id": 1, "timestamps": [{ "name": "COMPUTE_START", "ns": 2356425054887198 }] },
  {"id": 1, "timestamps": [{ "name": "COMPUTE_INPUT_END", "ns": 2356425057152908 }] },
  {"id": 1, "timestamps": [{ "name": "COMPUTE_OUTPUT_START", "ns": 2356425057497763 }] },
  {"id": 1, "timestamps": [{ "name": "COMPUTE_END", "ns": 2356425057540989 }] },
  {"id": 1, "timestamps": [{ "name": "REQUEST_END", "ns": 2356425057643164 }] },
  {"id": 1, "timestamps": [{ "name": "HTTP_SEND_START", "ns": 2356425057681578 }] },
  {"id": 1, "timestamps": [{ "name": "HTTP_SEND_END", "ns": 2356425057712991 }] }
]
```

Each `TENSORS` trace will contain an "activity" and a "tensor". 
"activity" indicates the type of tensor, including "TENSOR_QUEUE_INPUT" 
and "TENSOR_BACKEND_OUTPUT" by now. "tensor" has the detail of tensor, 
including its "name", "data" and "dtype". For example:

```
[
  {
    "id": 1,
    "activity": "TENSOR_QUEUE_INPUT",
    "tensor":{
      "name": "input",
      "data": "0.1,0.1,0.1,...",
      "shape": "1,16",
      "dtype": "FP32"
    }
  }
]
```

## Trace Summary Tool

An example [trace summary tool](https://github.com/triton-inference-server/server/blob/main/qa/common/trace_summary.py) can be
used to summarize a set of traces collected from Triton. Basic usage
is:

```
$ trace_summary.py <trace file>
```

This produces a summary report for all traces in the file. HTTP and
GRPC inference requests are reported separately.

```
File: trace.json
Summary for simple (-1): trace count = 1
HTTP infer request (avg): 378us
	Receive (avg): 21us
	Send (avg): 7us
	Overhead (avg): 79us
	Handler (avg): 269us
  		Overhead (avg): 11us
  		Queue (avg): 15us
  		Compute (avg): 242us
  			Input (avg): 18us
  			Infer (avg): 208us
  			Output (avg): 15us
Summary for simple (-1): trace count = 1
GRPC infer request (avg): 21441us
	Wait/Read (avg): 20923us
	Send (avg): 74us
	Overhead (avg): 46us
	Handler (avg): 395us
  		Overhead (avg): 16us
  		Queue (avg): 47us
  		Compute (avg): 331us
  			Input (avg): 30us
  			Infer (avg): 286us
  			Output (avg): 14us
```

Use the -t option to get a summary for each trace in the file. This
summary shows the time, in microseconds, between different points in
the processing of an inference request. For example, the below output
shows that it took 15us from the start of handling the request until
the request was enqueued in the scheduling queue.

```
$ trace_summary.py -t <trace file>
...
simple (-1):
  	grpc wait/read start
  		26529us
  	grpc wait/read end
  		39us
  	request handler start
  		15us
  	queue start
  		20us
  	compute start
  		266us
  	compute end
  		4us
  	request handler end
  		19us
  	grpc send start
  		77us
  	grpc send end
...
```

The script can also show the data flow of the first request if there are 
`TENSORS` traces in the file. If the `TENSORS` traces are from an ensemble, 
the data flow will be shown with the dependency of each model.

```
...
Data Flow:
	==========================================================
	Name:   ensemble
	Version:1
	QUEUE_INPUT:
		input: [[0.705676  0.830855  0.833153]]
	BACKEND_OUTPUT:
		output: [[1. 2. 7. 0. 4. 7. 9. 3. 4. 9.]]
	==========================================================
		==================================================
		Name:   test_trt1
		Version:1
		QUEUE_INPUT:
			input: [[0.705676  0.830855  0.833153]]
		BACKEND_OUTPUT:
			output1: [[1. 1. ...]]
		==================================================
		==================================================
		Name:   test_trt2
		Version:1
		QUEUE_INPUT:
			input: [[0.705676  0.830855  0.833153]]
		BACKEND_OUTPUT:
			output2: [[2. 2. ...]]
		==================================================
		==================================================
		Name:   test_py
		Version:1
		QUEUE_INPUT:
			output1: [[1. 1. ...]]
		QUEUE_INPUT:
			output2: [[2. 2. ...]]
		BACKEND_OUTPUT:
			output: [[1. 2. 7. 0. 4. 7. 9. 3. 4. 9.]]
		==================================================
...
```

The meaning of the trace timestamps is:

* GRPC Request Wait/Read: Collected only for inference requests that use the
  GRPC protocol. The time spent waiting for a request to arrive at the
  server and for that request to be read. Because wait time is
  included in the time it is not a useful measure of how much time is
  spent reading a request from the network. Tracing an HTTP request
  will provide an accurate measure of the read time.

* HTTP Request Receive: Collected only for inference requests that use the
  HTTP protocol. The time required to read the inference request from
  the network.

* Send: The time required to send the inference response.

* Overhead: Additional time required in the HTTP or GRPC endpoint to
  process the inference request and response.

* Handler: The total time spent handling the inference request, not
  including the HTTP and GRPC request/response handling.

  * Queue: The time the inference request spent in the scheduling queue.

  * Compute: The time the inference request spent executing the actual
    inference. This time includes the time spent copying input and
    output tensors. If --trace-level=TIMESTAMPS then a breakdown of the
    compute time will be provided as follows:

    * Input: The time to copy input tensor data as required by the
      inference framework / backend. This includes the time to copy
      input tensor data to the GPU.

    * Infer: The time spent executing the model to perform the
      inference.

    * Output: The time to copy output tensor data as required by the
      inference framework / backend. This includes the time to copy
      output tensor data from the GPU.

  * Overhead: Additional time required for request handling not
    covered by Queue or Compute times.

* Data Flow: The data flow of the first request. It contains the input and 
  output tensors of each part of execution.

  * Name: The name of model.

  * Version: The version of model.

  * QUEUE_INPUT: The tensor entering the queue of a backend to wait for 
    scheduling.

  * BACKEND_OUTPUT: The tensor in the response of a backend.

## Opentelemetry trace support

Starting with 23.04 release triton provides an option to generate and export traces, 
using [Opentelemetry APIs and SDKs](https://opentelemetry.io/). Currently this option is
only supported for Ubuntu based systems and does not support tracing for bls models.

To specify Opentelemetry as a tracing APIs, specify --trace-config flag as follows:

```
$ tritonserver --trace-config=mode=opentelemetry ...
```
At the moment, triton supports HTTP Exporter, provided by Opentelemetry. The endpoint 
can be specified as follows:
```
$ tritonserver --trace-config=mode=opentelemetry --trace-config=opentelemetry,url=<your endpoint> ...
```
By default, opentelemetry uses `0.0.0.0:4318/v2/traces` as an endpoint for HTTP exporter.