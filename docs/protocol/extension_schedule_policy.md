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

# Schedule Policy Extension

This document describes Triton's schedule policy extension. The
schedule-policy extension allows an inference request to provide
parameters that influence how Triton handles and schedules the
request.  Because this extension is supported, Triton reports
“schedule_policy” in the extensions field of its Server Metadata.

The schedule-policy extension uses request parameters to indicate the
policy. The parameters and their type are:

- "priority" : int64 value indicating the priority of the
  request. Priority value zero indicates that the default priority
  level should be used (i.e. same behavior as not specifying the
  priority parameter). Lower value priorities indicate higher priority
  levels. Thus the highest priority level is indicated by setting the
  parameter to 1, the next highest is 2, etc.

- "timeout" : int64 value indicating the timeout value for the
  request, in microseconds. If the request cannot be completed within
  the time Triton will take a model-specific action such as
  terminating the request.

Both parameters are optional and if not specified Triton will handle
the request using the default priority and timeout values appropriate
for the model.
