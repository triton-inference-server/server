<!--
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Generate Extension

This document describes Triton's generate extension. The generate
extension provides a simple text oriented endpoint schema for interacting
with LLM models. This generate endpoint is specific to HTTP/REST frontend.

## HTTP/REST

In all JSON schemas shown in this document `$number`, `$string`, `$boolean`,
`$object` and `$array` refer to the fundamental JSON types. #optional
indicates an optional JSON field.

Triton exposes the generate endpoint at the following URLs. The client may use
HTTP POST request to different URLs for different response behavior, the
endpoint will return the generate results on success or an error in the case of
failure.

```
POST v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/generate

POST v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/generate_stream
```

### generate v.s. generate_stream

Both URLs expect the same request JSON object, and generate the same response
JSON object. However, `generate` return exactly 1 response JSON object, while
`generate_stream` may return various number of responses based on the inference
results. `generate_stream` returns the responses as
[Server-Sent Events](https://html.spec.whatwg.org/multipage/server-sent-events.html#server-sent-events)
(SSE), where each response will be a "data" chunk in the HTTP response body.
Also note that an error may be returned during inference, whereas the HTTP
response code has been set at first response of the SSE, which can result in
receiving error object while status code is success (200). Therefore the user
must always check whether an error object is received when generating responses
through `generate_stream`.

### Generate Request JSON Object

The generate request object, identified as *$generate_request*, is
required in the HTTP body of the POST request. The model name and
(optionally) version must be available in the URL. If a version is not
provided the server may choose a version based on its own policies or
return an error.

    $generate_request =
    {
      "text_input" : $string,
      "parameters" : $parameters #optional
    }

* "text_input" : The text input that the model should generate output from.
* "parameters" : An optional object containing zero or more parameters for this
  generate request expressed as key/value pairs. See
  [Parameters](#parameters) for more information.

#### Parameters

The *$parameters* JSON describes zero or more “name”/”value” pairs,
where the “name” is the name of the parameter and the “value” is a
$string, $number, or $boolean.

    $parameters =
    {
      $parameter, ...
    }

    $parameter = $string : $string | $number | $boolean

Parameters are model-specific. In other words, the accepting names and values
varies according to the model. The user should check with the model
specification to set the parameters.

### Generate Response JSON Object

A successful generate request is indicated by a 200 HTTP status code.
The generate response object, identified as *$generate_response*, is returned in
the HTTP body.

    $generate_response =
    {
      "text_output" : $string
    }

* "text_output" : The output of the inference.

### Generate Response JSON Error Object

A failed generate request must be indicated by an HTTP error status
(typically 400). The HTTP body must contain the
*$generate_error_response* object.

    $generate_error_response =
    {
      "error": <error message string>
    }

* “error” : The descriptive message for the error.