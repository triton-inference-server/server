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

# Statistics Extension

This document describes Triton's statistics extension. The statistics
extension enables the reporting of per-model (per-version) statistics
which provide aggregate information about all activity occurring for a
specific model (version) since Triton started. Because this extension
is supported, Triton reports “statistics” in the extensions field of
its Server Metadata.

## HTTP/REST

In all JSON schemas shown in this document $number, $string, $boolean,
$object and $array refer to the fundamental JSON types. #optional
indicates an optional JSON field.

Triton exposes the statistics endpoint at the following URL. The
versions portion of the URL is optional; if not provided Triton will
return statistics for all versions of the model.

```
GET v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/stats
```

### Statistics Response JSON Object

A successful statistics request is indicated by a 200 HTTP status
code. For a model the statistics response object, identified as
$stats_model_response, is returned in the HTTP body for every
successful statistics request.

```
$stats_model_response =
{
  "version_stats" : [ $version_stat, ... ]
}
```

Each $version_stat object gives the statistics for a specific model version.

```
$version_stat =
{
  "version" : $string #optional,
  "stats" : $stats
}
```

```
$stats =
{
  "inference" : {
    "success" : $duration_stat,
    "fail" : $duration_stat,
    "queue" : $duration_stat,
    "compute_input" : $duration_stat,
    "compute_infer" : $duration_stat,
    "compute_output" : $duration_stat
  }
}
```

- “inference:success” : The count and cumulative duration for all
  successful inference requests.

- “inference:fail” : The count and cumulative duration for all failed
  inference requests.

- “inference:queue” : The count and cumulative duration that inference
  requests wait in scheduling or other queues.

- “inference:compute_input” : The count and cumulative duration to
  prepare input tensor data as required by the model framework /
  backend. For example, this duration should include the time to copy
  input tensor data to the GPU.

- “inference:compute_infer” : The count and cumulative duration to
  execute the model.

- “inference:compute_output” : The count and cumulative duration to
  extract output tensor data produced by the model framework /
  backend. For example, this duration should include the time to copy
  output tensor data from the GPU.

The $duration_stat object reports a count and a total time. This
format can be sampled to determine not only long-running averages but
also incremental averages between sample points.

```
$duration_stat =
{
  "count" : $number,
  "ns" : $number
}
```

- "count" : The number of times the statistic was collected.

- “ns” : The total duration for the statistic in nanoseconds.

### Statistics Response JSON Error Object

A failed statistics request will be indicated by an HTTP error status
(typically 400). The HTTP body must contain the
$repository_statistics_error_response object.

```
$repository_statistics_error_response =
{
  "error": $string
}
```

- “error” : The descriptive message for the error.

## GRPC

For the statistics extension Triton implements the following API:

```
service GRPCInferenceService
{
  …

  // Get the cumulative statistics for a model and version.
  rpc ModelStatistics(ModelStatisticsRequest)
          returns (ModelStatisticsResponse) {}
}
```

The ModelStatistics API returns model statistics. Errors are indicated
by the google.rpc.Status returned for the request. The OK code
indicates success and other codes indicate failure. The request and
response messages for ModelStatistics are:

```
message ModelStatisticsRequest
{
  // The name of the model.
  string name = 1;

  // The version of the model. If not given returns statistics for
  // all model versions.
  string version = 2;
}

message ModelStatisticsResponse
{
    // Map from version to inference statistics for that version.
    map<string, InferStatistics> inference = 1;
}
```

The statistics messages are:

```
// Statistic recording a cumulative duration metric.
message StatisticDuration
{
  // Cumulative number of times this metric occurred.
  uint64 count = 1;

  // Total collected duration of this metric in nanoseconds.
  uint64 ns = 2;
}

// Inference statistics.
message InferStatistics
{
  // Cumulative count and duration for successful inference requests,
  StatisticDuration success = 1;

  // Cumulative count and duration for failed inference requests,
  StatisticDuration fail = 2;

  // The count and cumulative duration that inference requests wait in
  // scheduling or other queues.
  StatisticDuration queue = 3;

  // The count and cumulative duration to prepare input tensor data as
  // required by the model framework / backend. For example, this duration
  // should include the time to copy input tensor data to the GPU.
  StatisticDuration compute_input = 4;

  // The count and cumulative duration to execute the model.
  StatisticDuration compute_infer = 5;

  // The count and cumulative duration to extract output tensor data
  // produced by the model framework / backend. For example, this duration
  // should include the time to copy output tensor data from the GPU.
  StatisticDuration compute_output = 6;
}
```
