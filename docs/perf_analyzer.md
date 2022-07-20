<!--
# Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
sources](https://github.com/triton-inference-server/client#getting-the-client-libraries-and-examples).

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
$ perf_analyzer -m inception_graphdef --percentile=95
*** Measurement Settings ***
  Batch size: 1
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using p95 latency

Request concurrency: 1
  Client:
    Request count: 348
    Throughput: 69.6 infer/sec
    p50 latency: 13936 usec
    p90 latency: 18682 usec
    p95 latency: 19673 usec
    p99 latency: 21859 usec
    Avg HTTP time: 14017 usec (send/recv 200 usec + response wait 13817 usec)
  Server:
    Inference count: 428
    Execution count: 428
    Successful request count: 428
    Avg request latency: 12005 usec (overhead 36 usec + queue 42 usec + compute input 164 usec + compute infer 11748 usec + compute output 15 usec)

Inferences/Second vs. Client p95 Batch Latency
Concurrency: 1, throughput: 69.6 infer/sec, latency 19673 usec
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

Using the --concurrency-range \<start\>:\<end\>:\<step\> option you can have
perf_analyzer collect data for a range of request concurrency
levels. Use the --help option to see complete documentation for this
and other options. For example, to see the latency and throughput of
your model for request concurrency values from 1 to 4:

```
$ perf_analyzer -m inception_graphdef --concurrency-range 1:4
*** Measurement Settings ***
  Batch size: 1
  Measurement window: 5000 msec
  Latency limit: 0 msec
  Concurrency limit: 4 concurrent requests
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 1
  Client:
    Request count: 339
    Throughput: 67.8 infer/sec
    Avg latency: 14710 usec (standard deviation 2539 usec)
    p50 latency: 13665 usec
...
Request concurrency: 4
  Client:
    Request count: 415
    Throughput: 83 infer/sec
    Avg latency: 48064 usec (standard deviation 6412 usec)
    p50 latency: 47975 usec
    p90 latency: 56670 usec
    p95 latency: 59118 usec
    p99 latency: 63609 usec
    Avg HTTP time: 48166 usec (send/recv 264 usec + response wait 47902 usec)
  Server:
    Inference count: 498
    Execution count: 498
    Successful request count: 498
    Avg request latency: 45602 usec (overhead 39 usec + queue 33577 usec + compute input 217 usec + compute infer 11753 usec + compute output 16 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 67.8 infer/sec, latency 14710 usec
Concurrency: 2, throughput: 89.8 infer/sec, latency 22280 usec
Concurrency: 3, throughput: 80.4 infer/sec, latency 37283 usec
Concurrency: 4, throughput: 83 infer/sec, latency 48064 usec
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
$ perf_analyzer -m inception_graphdef --concurrency-range 1:4 -f perf.csv
$ cat perf.csv
Concurrency,Inferences/Second,Client Send,Network+Server Send/Recv,Server Queue,Server Compute Input,Server Compute Infer,Server Compute Output,Client Recv,p50 latency,p90 latency,p95 latency,p99 latency
1,69.2,225,2148,64,206,11781,19,0,13891,18795,19753,21018
3,84.2,237,1768,21673,209,11742,17,0,35398,43984,47085,51701
4,84.2,279,1604,33669,233,11731,18,1,47045,56545,59225,64886
2,87.2,235,1973,9151,190,11346,17,0,21874,28557,29768,34766
```

NOTE: The rows in the CSV file are sorted in an increasing order of throughput (Inferences/Second).

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

For tensors with with STRING/BYTES datatype there are additional
options --string-length and --string-data that may be used in some
cases (see --help for full documentation).

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
object must list its data in a row-major order. Binary data must be in
little-endian byte order. The following example highlights how this
can be acheived:

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
tensors as input. Additionally note that when providing the "shape" field,
tensor contents must be provided separately in "content" field in row-major
order. The specified shape values will override default input shapes
provided as a command line option (see --shape) for variable-sized inputs.
In the absence of "shape" field, the provided defaults will be used. There
is no need to specify shape as a command line option if all the data steps
provide shape values for variable tensors. Below is an example json file
for a model with single input "INPUT", shape [-1,-1] and data type INT32:

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

The following is the example to provide contents as base64 string with explicit shapes:

```
{
  "data": [{ 
      "INPUT": {
                 "content": {"b64": "/9j/4AAQSkZ(...)"},
                 "shape": [7964]
               }},
    (...)]
}
```

Note that for STRING type an element is represented by a 4-byte unsigned integer giving
the length followed by the actual bytes. The byte array to be encoded using base64 must
include the 4-byte unsigned integers.

### Output Validation

When real input data is provided, it is optional to request perf analyzer to
validate the inference output for the input data.

Validation output can be specified in "validation_data" field in the same format
as "data" field for real input. Note that the entries in "validation_data" must
align with "data" for proper mapping. The following example describes validation
data for a model with inputs named, INPUT0 and INPUT1, outputs named, OUTPUT0
and OUTPUT1, all tensors have shape [4, 4] and data type INT32:

```
  {
    "data" :
     [
        {
          "INPUT0" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          "INPUT1" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
        ...
      ],
    "validation_data" :
     [
        {
          "OUTPUT0" : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          "OUTPUT1" : [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        }
        ...
      ]
  }
```

Besides the above example, the validation outputs can be specified in the same
variations described in "real input data" section.

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

### SSL/TLS Support

perf_analyzer can be used to benchmark Triton service behind SSL/TLS-enabled endpoints. These options can help in establishing secure connection with the endpoint and profile the server.

For gRPC, see the following options:

* `--ssl-grpc-use-ssl`
* `--ssl-grpc-root-certifications-file`
* `--ssl-grpc-private-key-file`
* `--ssl-grpc-certificate-chain-file`

More details here: https://grpc.github.io/grpc/cpp/structgrpc_1_1_ssl_credentials_options.html

The [inference protocol gRPC SSL/TLS section](inference_protocols.md#ssltls) describes server-side options to configure SSL/TLS in Triton's gRPC endpoint.

For HTTPS, the following options are exposed:

* `--ssl-https-verify-peer`
* `--ssl-https-verify-host`
* `--ssl-https-ca-certificates-file`
* `--ssl-https-client-certificate-file`
* `--ssl-https-client-certificate-type`
* `--ssl-https-private-key-file`
* `--ssl-https-private-key-type`

See `--help` for full documentation.

Unlike gRPC, Triton's HTTP server endpoint can not be configured with SSL/TLS support.

Note: Just providing these `--ssl-http-*` options to perf_analyzer does not ensure the SSL/TLS is used in communication. If SSL/TLS is not enabled on the service endpoint, these options have no effect. The intent of exposing these options to a user of perf_analyzer is to allow them to configure perf_analyzer to benchmark Triton service behind SSL/TLS-enabled endpoints. In other words, if Triton is running behind a HTTPS server proxy, then these options would allow perf_analyzer to profile Triton via exposed HTTPS proxy.

## Benchmarking Triton directly via C API

Besides using HTTP or gRPC server endpoints to communicate with Triton, perf_analyzer also allows user to benchmark Triton directly using C API. HTTP/gRPC endpoints introduce an additional latency in the pipeline which may not be of interest to the user who is using Triton via C API within their application. Specifically, this feature is useful to benchmark bare minimum Triton without additional overheads from HTTP/gRPC communication.

### Prerequisite
Pull the Triton SDK and the Inference Server container images on target machine.
Since you will need access to the Tritonserver install, it might be easier if 
you copy the perf_analyzer binary to the Inference Server container.

### Required Parameters
Use the --help option to see complete list of supported command line arguments.
By default perf_analyzer expects the Triton instance to already be running. You can configure the C API mode using the `--service-kind` option. In additon, you will need to point
perf_analyzer to the Triton server library path using the `--triton-server-directory` option and the model 
repository path using the `--model-repository` option.
If the server is run successfully, there is a prompt: "server is alive!" and perf_analyzer will print the stats, as normal.
An example run would look like:
```
perf_analyzer -m graphdef_int32_int32_int32 --service-kind=triton_c_api --triton-server-directory=/opt/tritonserver --model-repository=/workspace/qa/L0_perf_analyzer_capi/models
```

### Non-supported functionalities
There are a few functionalities that are missing from the C API. They are:
1. Async mode (`-a`)
2. Using shared memory mode (`--shared-memory=cuda` or `--shared-memory=system`)
3. Request rate range mode
4. For additonal known non-working cases, please refer to 
   [qa/L0_perf_analyzer_capi/test.sh](https://github.com/triton-inference-server/server/blob/main/qa/L0_perf_analyzer_capi/test.sh#L239-L277)


## Benchmarking TensorFlow Serving
perf_analyzer can also be used to benchmark models deployed on
[TensorFlow Serving](https://github.com/tensorflow/serving) using
the `--service-kind` option. The support is however only available
through gRPC protocol.
 
Following invocation demonstrates how to configure perf_analyzer
to issue requests to a running instance of
`tensorflow_model_server`:
 
```
$ perf_analyzer -m resnet50 --service-kind tfserving -i grpc -b 1 -p 5000 -u localhost:8500
*** Measurement Settings ***
  Batch size: 1
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using average latency
Request concurrency: 1
  Client: 
    Request count: 829
    Throughput: 165.8 infer/sec
    Avg latency: 6032 usec (standard deviation 569 usec)
    p50 latency: 5863 usec
    p90 latency: 6655 usec
    p95 latency: 6974 usec
    p99 latency: 8093 usec
    Avg gRPC time: 5984 usec ((un)marshal request/response 257 usec + response wait 5727 usec)
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 165.8 infer/sec, latency 6032 usec
```
 
You might have to specify a different url(`-u`) to access wherever
the server is running. The report of perf_analyzer will only
include statistics measured at the client-side.
 
**NOTE:** The support is still in **beta**. perf_analyzer does
not guarantee optimum tuning for TensorFlow Serving. However, a
single benchmarking tool that can be used to stress the inference
servers in an identical manner is important for performance
analysis.

 
The following points are important for interpreting the results:
1. `Concurrent Request Execution`:
TensorFlow Serving (TFS), as of version 2.8.0, by default creates
threads for each request that individually submits requests to
TensorFlow Session. There is a resource limit on the number of
concurrent threads serving requests. When benchmarking at a higher
request concurrency, you can see higher throughput because of this.  
Unlike TFS, by default Triton is configured with only a single
[instance count](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#instance-groups). Hence, at a higher request concurrency, most
of the requests are blocked on the instance availability. To
configure Triton to behave like TFS, set the instance count to a
reasonably high value and then set
[MAX_SESSION_SHARE_COUNT](https://github.com/triton-inference-server/tensorflow_backend#parameters)
parameter in the model confib.pbtxt to the same value.For some
context, the TFS sets its thread constraint to four times the
num of schedulable CPUs.
2. `Different library versions`:
The version of TensorFlow might differ between Triton and
TensorFlow Serving being benchmarked. Even the versions of cuda
libraries might differ between the two solutions. The performance
of models can be susceptible to the versions of these libraries.
For a single request concurrency, if the compute_infer time
reported by perf_analyzer when benchmarking Triton is as large as
the latency reported by perf_analyzer when benchmarking TFS, then
the performance difference is likely because of the difference in
the software stack and outside the scope of Triton.
3. `CPU Optimization`:
TFS has separate builds for CPU and GPU targets. They have
target-specific optimization. Unlike TFS, Triton has a single build
which is optimized for execution on GPUs. When collecting performance
on CPU models on Triton, try running Triton with the environment
variable `TF_ENABLE_ONEDNN_OPTS=1`.
 
 
## Benchmarking TorchServe
perf_analyzer can also be used to benchmark
[TorchServe](https://github.com/pytorch/serve) using the
`--service-kind` option. The support is however only available through
HTTP protocol. It also requires input to be provided via JSON file.
 
Following invocation demonstrates how to configure perf_analyzer to
issue requests to a running instance of `torchserve` assuming the
location holds `kitten_small.jpg`:
 
```
$ perf_analyzer -m resnet50 --service-kind torchserve -i http -u localhost:8080 -b 1 -p 5000 --input-data data.json
 Successfully read data for 1 stream/streams with 1 step/steps.
*** Measurement Settings ***
  Batch size: 1
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using average latency
Request concurrency: 1
  Client: 
    Request count: 799
    Throughput: 159.8 infer/sec
    Avg latency: 6259 usec (standard deviation 397 usec)
    p50 latency: 6305 usec
    p90 latency: 6448 usec
    p95 latency: 6494 usec
    p99 latency: 7158 usec
    Avg HTTP time: 6272 usec (send/recv 77 usec + response wait 6195 usec)
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 159.8 infer/sec, latency 6259 usec
```
 
The content of `data.json`:
 
```
 {
   "data" :
    [
       {
         "TORCHSERVE_INPUT" : ["kitten_small.jpg"]
       }
     ]
 }
```
 
You might have to specify a different url(`-u`) to access wherever
the server is running. The report of perf_analyzer will only include
statistics measured at the client-side.
 
**NOTE:** The support is still in **beta**. perf_analyzer does not
guarantee optimum tuning for TorchServe. However, a single benchmarking
tool that can be used to stress the inference servers in an identical
manner is important for performance analysis.

## Advantages of using Perf Analyzer over third-party benchmark suites

Triton Inference Server offers the entire serving solution which
includes [client libraries](https://github.com/triton-inference-server/client)
that are optimized for Triton.
Using third-party benchmark suites like jmeter fails to take advantage of the
optimized libraries. Some of these optimizations includes but are not limited
to:
1. Using [binary tensor data extension](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_binary_data.md) with HTTP requests.
2. Effective re-use of gRPC message allocation in subsequent requests.
3. Avoiding extra memory copy via libcurl interface.

These optimizations can have a tremendous impact on overall performance.
Using perf_analyzer for benchmarking directly allows a user to access
these optimizations in their study. 

Not only that, perf_analyzer is also very customizable and supports many
Triton features as described in this document. This, along with a detailed
report, allows a user to identify performance bottlenecks and experiment
with different features before deciding upon what works best for them.
