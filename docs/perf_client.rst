..
  # Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

.. _section-perf-client:

perf\_client
------------

A critical part of optimizing the inference performance of your model
is being able to measure changes in performance as you experiment with
different optimization strategies. The *perf\_client* application
performs this task for the TensorRT Inference Server. The perf\_client
is included with the client examples which are :ref:`available from
several sources <section-getting-the-client-examples>`.

The perf\_client generates inference requests to your model and
measures the throughput and latency of those requests. To get
representative results, the perf\_client measures the throughput and
latency over a time window, and then repeats the measurements until it
gets stable values. By default the perf\_client uses average latency
to determine stability but you can use the -\\-percentile flag to
stabilize results based on that confidence level. For example,
if -\\-percentile=95 is used the results will be stabilized using the
95-th percentile request latency. For example::

  $ perf_client -m resnet50_netdef --percentile=95
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

.. _section-perf-client-request-concurrency:

Request Concurrency
^^^^^^^^^^^^^^^^^^^

By default perf\_client measures your model's latency and throughput
using the lowest possible load on the model. To do this perf\_client
sends one inference request to the server and waits for the response.
When that response is received, the perf\_client immediately sends
another request, and then repeats this process during the measurement
windows. The number of outstanding inference requests is referred to
as the *request concurrency*, and so by default perf\_client uses a
request concurrency of 1.

Using the -\\-concurrency-range <start>:<end>:<step> option you can have
perf\_client collect data for a range of request concurrency
levels. Use the -\\-help option to see complete documentation for this
and other options. For example, to see the latency and throughput of
your model for request concurrency values from 1 to 4::

  $ perf_client -m resnet50_netdef --concurrency-range 1:4
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

Understanding The Output
^^^^^^^^^^^^^^^^^^^^^^^^

For each request concurrency level perf\_client reports latency and
throughput as seen from the *client* (that is, as seen by
perf\_client) and also the average request latency on the server.

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

Use the verbose (\-v) option to perf\_client to see more output,
including the stabilization passes run for each request concurrency
level.

.. _section-perf-client-visualize:

Visualizing Latency vs. Throughput
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The perf\_client provides the \-f option to generate a file containing
CSV output of the results::

  $ perf_client -m resnet50_netdef --concurrency-range 1:4 -f perf.csv
  $ cat perf.csv
  Concurrency,Inferences/Second,Client Send,Network+Server Send/Recv,Server Queue,Server Compute,Client Recv,p50 latency,p90 latency,p95 latency,p99 latency
  1,160.8,68,1291,38,4801,7,6212,6289,6328,7407
  3,207.8,70,1211,8346,4786,8,14379,14457,14536,15853
  4,208.4,71,1014,13314,4776,8,19168,19218,19265,19583
  2,209.2,67,1204,3511,4756,7,9545,9576,9588,9627

You can import the CSV file into a spreadsheet to help visualize
the latency vs inferences/second tradeoff as well as see some
components of the latency. Follow these steps:

- Open `this spreadsheet
  <https://docs.google.com/spreadsheets/d/1IsdW78x_F-jLLG4lTV0L-rruk0VEBRL7Mnb-80RGLL4>`_
- Make a copy from the File menu "Make a copy..."
- Open the copy
- Select the A1 cell on the "Raw Data" tab
- From the File menu select "Import..."
- Select "Upload" and upload the file
- Select "Replace data at selected cell" and then select the "Import data" button

Input Data
^^^^^^^^^^

Use the -\\-help option to see complete documentation for all input
data options. By default perf\_client sends random data to all the
inputs of your model. You can select a different input data mode with
the -\\-input-data option:

- *random*: (default) Send random data for each input.
- *zero*: Send zeros for each input.
- directory path: A path to a directory containing a binary file for each input, named the same as the input. Each binary file must contain the data required for that input for a batch-1 request. Each file should contain the raw binary representation of the input in row-major order.
- file path: A path to a JSON file containing data to be used with every inference request. See the "Real Input Data" section for further details. --input-data can be provided multiple times with different file paths to specific multiple JSON files.

For tensors with with STRING datatype there are additional options
-\\-string-length and -\\-string-data that may be used in some cases
(see -\\-help for full documentation).

For models that support batching you can use the \-b option to
indicate the batch-size of the requests that perf\_client should
send. For models with variable-sized inputs you must provide the
-\\-shape argument so that perf\_client knows what shape tensors to
use. For example, for a model that has an input called *IMAGE* that
has shape [ 3, N, M ], where N and M are variable-size dimensions, to
tell perf\_client to send batch-size 4 requests of shape [ 3, 224, 224 ]::

  $ perf_client -m mymodel -b 4 --shape IMAGE:3,224,224

Real Input Data
^^^^^^^^^^^^^^^

The performance of some models is highly dependent on the data used.
For such cases users can provide data to be used with every inference request
made by client in a JSON file. The perf_client will use the provided data when
sending inference requests in a round-robin fashion.

Each entry in the "data" array must specify all input tensors with the exact 
size expected by the model from a single batch. The following example describes
data for a model with inputs named, INPUT0 and INPUT1, shape [4, 4] and data 
type INT32: ::


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
        .
        .
        .
      ]
  }

Kindly note that the [4, 4] tensor has been flattened in a row-major format for the inputs.

A part from specifying explicit tensors, users can also provide Base64 encoded binary data 
for the tensors. Each data object must list its data in a row-major order. The following 
example highlights how this can be acheived: ::

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
        .
        .
        .
      ]
  }


In case of sequence models, multiple data streams can be specified in the JSON file. Each sequence
will get a data stream of its own and the client will ensure the data from each stream is
played back to the same correlation id. The below example highlights how to specify data for
multiple streams for a sequence model with a single input named INPUT, shape [1] and data type STRING: ::


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

The above example describes three data streams with lengths 4, 3 and 2 respectively.
The perf_client will hence produce sequences of length 4, 3 and 2 in this case.


Shared Memory
^^^^^^^^^^^^^

By default perf\_client sends input tensor data and receives output
tensor data over the network. You can instead instruct perf\_client to
use system shared memory or CUDA shared memory to communicate tensor
data. By using these options you can model the performance that you
can achieve by using shared memory in your application. Use
-\\-shared-memory=system to use system (CPU) shared memory or
-\\-shared-memory=cuda to use CUDA shared memory.

Communication Protocol
^^^^^^^^^^^^^^^^^^^^^^

By default perf\_client uses HTTP to communicate with the inference
server. The GRPC protocol can be specificed with the -i option. If
GRPC is selected the -\\-streaming option can also be specified for GRPC
streaming.
