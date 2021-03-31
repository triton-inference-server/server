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

# Model Repository Extension

This document describes Triton's model repository extension.  The
model-repository extension allows a client to query and control the
one or more model repositories being served by Triton.  Because this
extension is supported, Triton reports “model_repository” in the
extensions field of its Server Metadata.

## HTTP/REST

In all JSON schemas shown in this document $number, $string, $boolean,
$object and $array refer to the fundamental JSON types. #optional
indicates an optional JSON field.

The model-repository extension requires Index, Load and Unload
APIs. Triton exposes the endpoints at the following URLs.

```
POST v2/repository/index

POST v2/repository/models/${MODEL_NAME}/load

POST v2/repository/models/${MODEL_NAME}/unload[/cascading]
```

### Index

The index API returns information about every model available in a
model repository, even if it is not currently loaded into Triton. The
index API provides a way to determine which models can potentially be
loaded by the Load API. A model-repository index request is made with
an HTTP POST to the index endpoint. In the corresponding response the
HTTP body contains the JSON response.

The index request object, identified as $repository_index_request, is
required in the HTTP body of the POST request.

```
$repository_index_request =
{
  "ready" : $boolean #optional,
}
```

    "ready" : Optional, default is false. If true return only models ready for inferencing.

A successful index request is indicated by a 200 HTTP status code. The
response object, identified as $repository_index_response, is returned
in the HTTP body for every successful request.

```
$repository_index_response =
[
  {
    "name" : $string,
    "version" : $string #optional,
    "state" : $string,
    "reason" : $string
  },
  …
]
```

- “name” : The name of the model.
- “version” : The version of the model.
- “state” : The state of the model.
- “reason” : The reason, if any, that the model is in the current state.

A failed index request must be indicated by an HTTP error status
(typically 400). The HTTP body must contain the
$repository_index_error_response object.

```
$repository_index_error_response =
{
  "error": $string
}
```

- “error” : The descriptive message for the error.

### Load

The load API requests that a model be loaded into Triton, or reloaded
if the model is already loaded. A load request is made with an HTTP
POST to a load endpoint. The HTTP body must be empty. A successful
load request is indicated by a 200 HTTP status.

A failed load request must be indicated by an HTTP error status
(typically 400). The HTTP body must contain the
$repository_load_error_response object.

```
$repository_load_error_response =
{
  "error": $string
}
```
- “error” : The descriptive message for the error.

### Unload

The unload API requests that a model be unloaded from Triton. An
unload request is made with an HTTP POST to an unload endpoint. The
HTTP body must be empty. If cascading unload is specified and the model
to be unloaded is an ensemble model, the server will also unload the models
within the ensemble if they are only used by the ensemble.
A successful unload request is indicated by a 200 HTTP status.

A failed unload request must be indicated by an HTTP error status
(typically 400). The HTTP body must contain the
$repository_unload_error_response object.

```
$repository_unload_error_response =
{
  "error": $string
}
```

- “error” : The descriptive message for the error.

## GRPC

The model-repository extension requires the following API:

```
service GRPCInferenceService
{
  …

  // Get the index of model repository contents.
  rpc RepositoryIndex(RepositoryIndexRequest)
          returns (RepositoryIndexResponse) {}

  // Load or reload a model from a repository.
  rpc RepositoryModelLoad(RepositoryModeLoadRequest)
          returns (RepositoryModelLoadResponse) {}

  // Unload a model.
  rpc RepositoryModelUnload(RepositoryModelUnloadRequest)
          returns (RepositoryModelUnloadResponse) {}
}
```

### Index

The RepositoryIndex API returns information about every model
available in a model repository, even if it is not currently loaded
into Triton. Errors are indicated by the google.rpc.Status returned
for the request. The OK code indicates success and other codes
indicate failure. The request and response messages for
RepositoryIndex are:

```
message RepositoryIndexRequest
{
  // The name of the repository. If empty the index is returned
  // for all repositories.
  string repository_name = 1;

  // If true return only models currently ready for inferencing.
  bool ready = 2;
}

message RepositoryIndexResponse
{
  // Index entry for a model.
  message ModelIndex {
    // The name of the model.
    string name = 1;

    // The version of the model.
    string version = 2;

    // The state of the model.
    string state = 3;

    // The reason, if any, that the model is in the given state.
    string reason = 4;
  }

  // An index entry for each model.
  repeated ModelIndex models = 1;
}
```

### Load

The RepositoryModelLoad API requests that a model be loaded into
Triton, or reloaded if the model is already loaded. Errors are
indicated by the google.rpc.Status returned for the request. The OK
code indicates success and other codes indicate failure. The request
and response messages for RepositoryModelLoad are:

```
message RepositoryModelLoadRequest
{
  // The name of the repository to load from. If empty the model
  // is loaded from any repository.
  string repository_name = 1;

  // The name of the model to load, or reload.
  string model_name = 2;
}

message RepositoryModelLoadResponse
{
}
```

### Unload

The RepositoryModelUnload API requests that a model be unloaded from
Triton. Errors are indicated by the google.rpc.Status returned for the
request. The OK code indicates success and other codes indicate
failure. The request and response messages for RepositoryModelUnload
are:

```
message RepositoryModelUnloadRequest
{
  // The name of the repository from which the model was originally
  // loaded. If empty the repository is not considered.
  string repository_name = 1;

  // The name of the model to unload.
  string model_name = 2;
}

message RepositoryModelUnloadResponse
{
}
```
