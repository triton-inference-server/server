<!--
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

# Performance Analyzer

A critical part of optimizing the inference performance of your model
is being able to measure changes in performance as you experiment with
different optimization strategies. The perf_analyzer application
(previously known as perf_client) performs this task for the Triton
Inference Server. The perf_analyzer is included with the client
examples which are [available from several
sources](client_examples.md#getting-the-client-examples).

The perf_analyzer application generates inference requests to your
model and measures the throughput and latency of those requests. To
get representative results, perf_analyzer measures the throughput and
latency over a time window, and then repeats the measurements until it
gets stable values. By default perf_analyzer uses average latency to
determine stability but you can use the --percentile flag to stabilize
results based on that confidence level. For example, if
--percentile=95 is used the results will be stabilized using the 95-th
percentile request latency. For example,

```
$ perf_analyzer -m resnet50_netdef --percentile=95
*** Measurement Settings ***
  Batch size: 1
  Measurement window: 5000 msec
  Stabilizing using p95 latency

Request concurrency: 1
  Client:
    Request count: 809
    Throughput: 161.8 infer/sec
    p50 latency: 6178 usec
    p90 latency: 6237 usec
    p95 latency: 6260 usec
    p99 latency: 6339 usec
    Avg HTTP time: 6153 usec (send/recv 72 usec + response wait 6081 usec)
  Server:
    Request count: 971
    Avg request latency: 4824 usec (overhead 10 usec + queue 39 usec + compute 4775 usec)

Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, 161.8 infer/sec, latency 6260 usec
```

## Request Concurrency

By default perf_analyzer measures your model's latency and throughput
using the lowest possible load on the model. To do this perf_analyzer
sends one inference request to Triton and waits for the response.
When that response is received, the perf_analyzer immediately sends
another request, and then repeats this process during the measurement
windows. The number of outstanding inference requests is referred to
as the *request concurrency*, and so by default perf_analyzer uses a
request concurrency of 1.

Using the --concurrency-range <start>:<end>:<step> option you can have
perf_analyzer collect data for a range of request concurrency
levels. Use the --help option to see complete documentation for this
and other options. For example, to see the latency and throughput of
your model for request concurrency values from 1 to 4:

```
$ perf_analyzer -m resnet50_netdef --concurrency-range 1:4
*** Measurement Settings ***
  Batch size: 1
  Measurement window: 5000 msec
  Latency limit: 0 msec
  Concurrency limit: 4 concurrent requests
  Stabilizing using average latency

Request concurrency: 1
  Client:
    Request count: 804
    Throughput: 160.8 infer/sec
    Avg latency: 6207 usec (standard deviation 267 usec)
    p50 latency: 6212 usec
...
Request concurrency: 4
  Client:
    Request count: 1042
    Throughput: 208.4 infer/sec
    Avg latency: 19185 usec (standard deviation 105 usec)
    p50 latency: 19168 usec
    p90 latency: 19218 usec
    p95 latency: 19265 usec
    p99 latency: 19583 usec
    Avg HTTP time: 19156 usec (send/recv 79 usec + response wait 19077 usec)
  Server:
    Request count: 1250
    Avg request latency: 18099 usec (overhead 9 usec + queue 13314 usec + compute 4776 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, 160.8 infer/sec, latency 6207 usec
Concurrency: 2, 209.2 infer/sec, latency 9548 usec
Concurrency: 3, 207.8 infer/sec, latency 14423 usec
Concurrency: 4, 208.4 infer/sec, latency 19185 usec
```

## Understanding The Output

For each request concurrency level perf_analyzer reports latency and
throughput as seen from the *client* (that is, as seen by
perf_analyzer) and also the average request latency on the server.

The server latency measures the total time from when the request is
received at the server until the response is sent from the
server. Because of the HTTP and GRPC libraries used to implement the
server endpoints, total server latency is typically more accurate for
HTTP requests as it measures time from first byte received until last
byte sent. For both HTTP and GRPC the total server latency is
broken-down into the following components:

- *queue*: The average time spent in the inference schedule queue by a
  request waiting for an instance of the model to become available.
- *compute*: The average time spent performing the actual inference,
  including any time needed to copy data to/from the GPU.

The client latency time is broken-down further for HTTP and GRPC as
follows:

- HTTP: *send/recv* indicates the time on the client spent sending the
  request and receiving the response. *response wait* indicates time
  waiting for the response from the server.
- GRPC: *(un)marshal request/response* indicates the time spent
  marshalling the request data into the GRPC protobuf and
  unmarshalling the response data from the GRPC protobuf. *response
  wait* indicates time writing the GRPC request to the network,
  waiting for the response, and reading the GRPC response from the
  network.

Use the verbose (-v) option to perf_analyzer to see more output,
including the stabilization passes run for each request concurrency
level.

## Visualizing Latency vs. Throughput

The perf_analyzer provides the -f option to generate a file containing
CSV output of the results.

```
$ perf_analyzer -m resnet50_netdef --concurrency-range 1:4 -f perf.csv
$ cat perf.csv
Concurrency,Inferences/Second,Client Send,Network+Server Send/Recv,Server Queue,Server Compute Input,Server Compute Infer,Server Compute Output,Client Recv,p50 latency,p90 latency,p95 latency,p99 latency
1,163.6,69,1230,33,43,4719,5,9,6133,6191,6224,6415
2,208.6,180,1306,3299,43,4720,5,28,9482,9617,10746,10832
4,209.8,173,1268,12835,40,4705,4,27,19046,19133,19164,19290
3,210.2,175,1267,8052,40,4697,4,27,14259,14325,14350,14426
```

You can import the CSV file into a spreadsheet to help visualize
the latency vs inferences/second tradeoff as well as see some
components of the latency. Follow these steps:

- Open [this
  spreadsheet](https://docs.google.com/spreadsheets/d/1S8h0bWBBElHUoLd2SOvQPzZzRiQ55xjyqodm_9ireiw)
- Make a copy from the File menu "Make a copy..."
- Open the copy
- Select the A1 cell on the "Raw Data" tab
- From the File menu select "Import..."
- Select "Upload" and upload the file
- Select "Replace data at selected cell" and then select the "Import data" button

## Input Data

Use the --help option to see complete documentation for all input
data options. By default perf_analyzer sends random data to all the
inputs of your model. You can select a different input data mode with
the --input-data option:

- *random*: (default) Send random data for each input.
- *zero*: Send zeros for each input.
- directory path: A path to a directory containing a binary file for each input, named the same as the input. Each binary file must contain the data required for that input for a batch-1 request. Each file should contain the raw binary representation of the input in row-major order.
- file path: A path to a JSON file containing data to be used with every inference request. See the "Real Input Data" section for further details. --input-data can be provided multiple times with different file paths to specific multiple JSON files.

For tensors with with STRING datatype there are additional options
--string-length and --string-data that may be used in some cases
(see --help for full documentation).

For models that support batching you can use the -b option to indicate
the batch-size of the requests that perf_analyzer should send. For
models with variable-sized inputs you must provide the --shape
argument so that perf_analyzer knows what shape tensors to use. For
example, for a model that has an input called *IMAGE* that has shape [
3, N, M ], where N and M are variable-size dimensions, to tell
perf_analyzer to send batch-size 4 requests of shape [ 3, 224, 224 ]:

```
$ perf_analyzer -m mymodel -b 4 --shape IMAGE:3,224,224
```

## Real Input Data

The performance of some models is highly dependent on the data used.
For such cases you can provide data to be used with every inference
request made by analyzer in a JSON file. The perf_analyzer will use
the provided data in a round-robin order when sending inference
requests.

Each entry in the "data" array must specify all input tensors with the
exact size expected by the model from a single batch. The following
example describes data for a model with inputs named, INPUT0 and
INPUT1, shape [4, 4] and data type INT32:

```
  {
    "data" :
     [
        {
          "INPUT0" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          "INPUT1" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        },
        {
          "INPUT0" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          "INPUT1" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        },
        {
          "INPUT0" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          "INPUT1" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        },
        {
          "INPUT0" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          "INPUT1" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
        ...
      ]
  }
```

Note that the [4, 4] tensor has been flattened in a row-major format
for the inputs. In addition to specifying explicit tensors, you can
also provide Base64 encoded binary data for the tensors. Each data
object must list its data in a row-major order. The following example
highlights how this can be acheived:

```
  {
    "data" :
     [
        {
          "INPUT0" : {"b64": "YmFzZTY0IGRlY29kZXI="},
          "INPUT1" : {"b64": "YmFzZTY0IGRlY29kZXI="}
        },
        {
          "INPUT0" : {"b64": "YmFzZTY0IGRlY29kZXI="},
          "INPUT1" : {"b64": "YmFzZTY0IGRlY29kZXI="}
        },
        {
          "INPUT0" : {"b64": "YmFzZTY0IGRlY29kZXI="},
          "INPUT1" : {"b64": "YmFzZTY0IGRlY29kZXI="}
        },
        ...
      ]
  }
```

In case of sequence models, multiple data streams can be specified in
the JSON file. Each sequence will get a data stream of its own and the
analyzer will ensure the data from each stream is played back to the
same correlation id. The below example highlights how to specify data
for multiple streams for a sequence model with a single input named
INPUT, shape [1] and data type STRING:

```
  {
    "data" :
      [
        [
          {
            "INPUT" : ["1"]
          },
          {
            "INPUT" : ["2"]
          },
          {
            "INPUT" : ["3"]
          },
          {
            "INPUT" : ["4"]
          }
        ],
        [
          {
            "INPUT" : ["1"]
          },
          {
            "INPUT" : ["1"]
          },
          {
            "INPUT" : ["1"]
          }
        ],
        [
          {
            "INPUT" : ["1"]
          },
          {
            "INPUT" : ["1"]
          }
        ]
      ]
  }
```

The above example describes three data streams with lengths 4, 3 and 2
respectively.  The perf_analyzer will hence produce sequences of
length 4, 3 and 2 in this case.

You can also provide an optional "shape" field to the tensors. This is
especially useful while profiling the models with variable-sized
tensors as input. The specified shape values are treated as an
override and analyzer still expects default input shapes to be
provided as a command line option (see --shape) for variable-sized
inputs. In the absence of "shape" field, the provided defaults will be
used. Below is an example json file for a model with single input
"INPUT", shape [-1,-1] and data type INT32:

```
  {
    "data" :
     [
        {
          "INPUT" :
                {
                    "content": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "shape": [2,8]
                }
        },
        {
          "INPUT" :
                {
                    "content": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "shape": [8,2]
                }
        },
        {
          "INPUT" :
                {
                    "content": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                }
        },
        {
          "INPUT" :
                {
                    "content": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "shape": [4,4]
                }
        }
        ...
      ]
  }
```

## Shared Memory

By default perf_analyzer sends input tensor data and receives output
tensor data over the network. You can instead instruct perf_analyzer to
use system shared memory or CUDA shared memory to communicate tensor
data. By using these options you can model the performance that you
can achieve by using shared memory in your application. Use
--shared-memory=system to use system (CPU) shared memory or
--shared-memory=cuda to use CUDA shared memory.

## Communication Protocol

By default perf_analyzer uses HTTP to communicate with Triton. The GRPC
protocol can be specificed with the -i option. If GRPC is selected the
--streaming option can also be specified for GRPC streaming.
