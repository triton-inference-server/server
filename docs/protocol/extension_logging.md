<!--
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Logging Extension

This document describes Triton's logging extension. The logging extension enables
the client to configure log settings during a Triton run. Triton reports "logging" 
in the extensions field of its Server Metadata.

## HTTP/REST

In all JSON schemas shown in this document $number, $string, $boolean,
$object and $array refer to the fundamental JSON types. #optional
indicates an optional JSON field.

Triton exposes the logging endpoint at the following URL. The client may use
HTTP GET request to retrieve the current log settings. A HTTP POST request
will modify the log settings, and the endpoint will return the updated log
settings on success or an error in the case of failure. 

```
GET v2/logging

POST v2/logging
```

### Log Setting Response JSON Object

A successful log setting request is indicated by a 200 HTTP status
code. The response object, identified as $log_setting_response, is
returned in the HTTP body for every successful log setting request.

```
$log_setting_response =
{
  $log_setting, ...
}

$log_setting = $string : $string | $boolean | $uint32
```

Each $log_setting JSON describes a “name”/”value” pair, where the “name” is
the $string representation of the log setting and the “value” is a $string, $bool, or $uint32 representation of the
setting value. Currently, the following log settings are defined:

- "log_file" : a $string parameter defining the file where the log outputs will be saved. If not specified,
log outputs will stream to the console.

- "log_info" : a $boolean parameter that controls whether the Triton server logs INFO level messages. 

- "log_warnings" : a $boolean parameter that controls whether the Triton server logs WARNING level messages. 

- "log_errors" : a $boolean parameter that controls whether the Triton server logs ERROR level messages. 

- "log_verbose_level" : a $uint32 parameter that controls whether the Triton server outputs verbose messages
of varying degrees. This value can be any integer >= 0. If "log_verbose_level" is 0, no verbose messages
will be output by the Triton server. If "log_verbose_level" is 1, level 1 verbose messages will be output
by the Triton server. If "log_verbose_level" is 2, the Triton server will output all verbose messages of 
level <= 2, etc.

- "log_format" : a $string parameter that controls the format of Triton server log messages. There are currently
2 formats: "default" and "ISO8601".


### Log Setting Response JSON Error Object

A failed log setting request will be indicated by an HTTP error status
(typically 400). The HTTP body will contain a
$log_setting_error_response object.

```
$log_setting_error_response =
{
  "error": $string
}
```

- “error” : The descriptive message for the error.

### Log Setting Request JSON Object

A log setting request is made with a HTTP POST to
the logging endpoint. In the corresponding response, the HTTP body contains the
response JSON. A successful request is indicated by a 200 HTTP status code.

The request object, identified as $log_setting_request must be provided in the HTTP
body.

```
$log_setting_request =
{
  $log_setting, ...
}
```

When a $log_setting JSON is received (defined above), only the specified
settings will be updated.

### Example Usage
The logging protocol extension can be invoked using the curl library in the following manner (assuming 
a Triton server is running at `localhost:8000`):
```
curl -s -w '\n%{http_code}\n' -d '{"log_verbose_level":1}' localhost:8000/v2/logging
```
This command should return a $log_setting_response JSON object with the following format:
```
{"log_file":"","log_info":true,"log_warnings":true,"log_errors":true,"log_verbose_level":1,"log_format":"default"}
200
```
Note that the current values for all parameter fields are returned even though `log_verbose_level` 
was the only parameter that was modified.

## GRPC

For the logging extension, Triton implements the following API:

```
service GRPCInferenceService
{
  …

  // Update and get the log setting of the Triton server.
  rpc LogSettings(LogSettingsRequest)
          returns (LogSettingsResponse) {}
}
```

The Log Setting API returns the latest log settings. Errors are indicated
by the `google.rpc.Status` returned for the request. The OK code
indicates success and other codes indicate failure. The request and
response messages for Log Settings are:

```
message LogSettingsRequest
{
  message SettingValue
  {
    oneof parameter_choice
    {
      // bool param option
      bool bool_param = 1;

      // uint32 param option
      uint32 uint32_param = 2;

      // string param option
      string string_param = 3;
    }
  }
  // The new setting values to be updated.
  // Unspecified settings will remain unchanged.
  map<string, SettingValue> settings = 1;
}

message LogSettingsResponse
{
  message SettingValue
  {
    oneof parameter_choice
    {
      // bool param option
      bool bool_param = 1;

      // uint32 param option
      uint32 uint32_param = 2;

      // string param option
      string string_param = 3;
    }
  }
  // The latest log settings values.
  map<string, SettingValue> settings = 1;
}
```
