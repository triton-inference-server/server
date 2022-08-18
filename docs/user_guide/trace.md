<!--
# Copyright 2019-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
arguments when running the tritonserver executable. For example,

```
$ tritonserver --trace-file=/tmp/trace.json --trace-rate=100 --trace-level=TIMESTAMPS ...
```

The --trace-file option indicates where the trace output should be
written. The --trace-rate option specifies the sampling rate. In
this example every 100-th inference request will be traced. The
--trace-level option indicates the level of trace detail that should
be collected. --trace-level option may be specified multiple times to 
trace multiple informations. Use the --help option to get more information.

In addition to configure trace settings in command line arguments, The user may
modify the trace setting when Triton server
is running via the trace APIs, more information can be found in [trace
protocol](protocol/extension_trace.md).

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
    "id": $number
    "parent_id": $number,
    "timestamps": [
      { "name" : $string, "ns" : $number },
      ...
    ]
  },
  {
    "model_name": $string,
    "model_version": $number,
    "id": $number
    "activity": $string,
    "tensor":{
      "name": $string,
      "data": $string,
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

Each `TIMESTAMPS` trace will have one or more "timestamps" with 
each timestamp having a name and the timestamp in nanoseconds ("ns"). 
For example:

```
[
  {
    "model_name": "simple",
    "model_version": -1,
    "id": 1,
    "timestamps" : [
      { "name": "http recv start", "ns": 2259961222771924 },
      { "name": "http recv end", "ns": 2259961222820985 },
      { "name": "request handler start", "ns": 2259961223164078 },
      { "name": "queue start", "ns": 2259961223182400 },
      { "name": "compute start", "ns": 2259961223232405 },
      { "name": "compute end", "ns": 2259961230206777 },
      { "name": "request handler end", "ns": 2259961230211887 },
      { "name": "http send start", "ns": 2259961230529606 },
      { "name": "http send end", "ns": 2259961230543930 }
    ]
  }
]
```

Each `TENSORS` trace will contain an "activity" and a "tensor". 
"activity" indicates the type of tensor, including "TENSOR_QUEUE_INPUT" 
and "TENSOR_BACKEND_OUTPUT" by now. "tensor" has the detail of tensor, 
including its "name", "data" and "dtype". For example:

```
[
  {
    "model_name": "simple",
    "model_version": -1,
    "id": 1,
    "activity": "TENSOR_QUEUE_INPUT",
    "tensor":{
      "name": "input",
      "data": "0.1,0.1,0.1,...",
      "dtype": "FP32"
    }
  }
]
```

## Trace Summary Tool

An example [trace summary tool](../qa/common/trace_summary.py) can be
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
