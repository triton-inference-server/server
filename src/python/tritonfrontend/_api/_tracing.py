# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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


import sys
from typing import Union

import tritonserver
from pydantic import Field
from pydantic.dataclasses import dataclass

#


class Tracing:
    @dataclass
    class Options:
        filepath: str = ""
        # trace_level = TRITONSERVER_TRACE_LEVEL_DISABLED
        rate: int = Field(1000, ge=0, help="Specifies sampling rate")
        count: int = -1
        # trace_log_frequency: int =
        # trace_mode = TRACE_MODE_TRITON

    #   std::string trace_filepath_{};


#   TRITONSERVER_InferenceTraceLevel trace_level_{
#       TRITONSERVER_TRACE_LEVEL_DISABLED};
#   int32_t trace_rate_{1000};
#   int32_t trace_count_{-1};
#   int32_t trace_log_frequency_{0};
#   InferenceTraceMode trace_mode_{TRACE_MODE_TRITON};
#   TraceConfigMap trace_config_map_;


# using TraceConfig = std::vector<
#     std::pair<std::string, std::variant<std::string, int, uint32_t>>>;
# // Key is trace mode,
# using TraceConfigMap = std::unordered_map<std::string, TraceConfig>;

# /// Trace modes.
# typedef enum tracemode_enum {
#   /// Default is Triton tracing API
#   TRACE_MODE_TRITON = 0,
#   /// OpenTelemetry API for tracing
#   TRACE_MODE_OPENTELEMETRY = 1
# } InferenceTraceMode;


# Current C++ flow of tracer arguments:
# TraceConfigMap
