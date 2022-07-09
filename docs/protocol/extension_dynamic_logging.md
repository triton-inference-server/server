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

# Dynamic Logging Extension

This document describes Triton's dynamic logging extension. The logging extension enables
the client to configure the log settings during a Triton run. Because this
extension is supported, Triton reports TODO in the extensions field of
its Server Metadata.

## HTTP/REST

In all JSON schemas shown in this document $number, $string, $boolean,
$object and $array refer to the fundamental JSON types. #optional
indicates an optional JSON field.

Triton exposes the dynamic logging endpoint at the following URL. The client may use
HTTP GET request to retrieve the current log settings. A HTTP POST request
will modify the log settings, and the endpoint will return the updated log
settings on success or an error in the case of failure. 

```
GET v2/dynamic_logging

POST v2/dynamic_logging
```

### Log Setting Response JSON Object

A successful log setting request is indicated by a 200 HTTP status
code. The response object, identified as TODO, is
returned in the HTTP body for every successful trace setting request.

```
$log_setting_response =
{
  $log_setting, ...
}

$log_setting = $string : $string | [ $string, ...]
```

Each $log_setting JSON describes a “name”/”value” pair, where the “name” is
the name of the trace setting and the “value” is a $string representation of the
setting value. Currently the following log settings are defined:

- "log_file" : the file where the log outputs will be saved. If not specified,
log outputs will continue to stream to the console. Info, warning, and error messages,
as well verbose messages, can be redirected into this log file, depending on the parameter
settings below. By default "log_file" is unspecified -- "".

- "log_info" : this parameter controls whether the Triton server outputs info messages or
writes them to the "log_file" (if specified). A value of "false" will stop the Triton server
from outputting info messages. By default "log_info" is "true".

- "log_warnings" : this parameter controls whether the Triton server outputs warning messages or
writes them to the "log_file" (if specified). A value of "false" will stop the Triton server
from outputting warning messages. By default "log_warnings" is "true".

- "log_errors" : this parameter controls whether the Triton server outputs error messages or
writes them to the "log_file" (if specified). A value of "false" will stop the Triton server
from outputting error messages. By default "log_errors" is "true".

- "log_verbose_level" : this parameter controls whether the Triton server outputs verbose messages
of varying degress. This value can be any integer >= 0. If "log_verbose_level" is 0, no verbose messages
will be output by the Triton server. If "log_verbose_level" is 1, level 1 verbose messages will be output
by the Triton server. If "log_verbose_level" is 2, the Triton server will output all verbose messages of 
level <= 2, etc. By default "log_verbose_level" is 0.

- "log_format" : this parameter controls the format of Triton server log messages. There are currently
2 formats: "default" and "ISO8601".


### Log Setting Response JSON Error Object

A failed log setting request will be indicated by an HTTP error status
(typically 400). The HTTP body must contain the
$log_setting_error_response object.

```
$log_setting_error_response =
{
  "error": $string
}
```

- “error” : The descriptive message for the error.

#### Log Setting Request JSON Object

A log setting request is made with a HTTP POST to
the dynamic logging endpoint. In the corresponding response, the HTTP body contains the
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

## GRPC

For the trace extension Triton implements the following API:

```
service GRPCInferenceService
{
  …

  // Update and get the trace setting of the Triton server.
  rpc LogSettings(LogSettingsRequest)
          returns (LogSettingsResponse) {}
}
```

The Log Setting API returns the latest log settings. Errors are indicated
by the google.rpc.Status returned for the request. The OK code
indicates success and other codes indicate failure. The request and
response messages for Log Settings are:

```
message LogSettingsRequest
{
  // The values to be associated with a log setting.
  message SettingValue
  {
    repeated string value = 1;
  }

  // The new setting values to be updated,
  // settings that are not specified will remain unchanged.
  map<string, SettingValue> settings = 1;
}

message LogSettingsResponse
{
  message SettingValue
  {
    repeated string value = 1;
  }

  // The latest trace settings.
  map<string, SettingValue> settings = 1;
}
```
