# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from enum import IntEnum

import tritonserver
from _typeshed import Incomplete as Incomplete

class Grpc_compression_level(IntEnum):
    NONE = 0
    LOW = 1
    MED = 2
    HIGH = 3
    COUNT = 4

class KServeGrpc:
    Grpc_compression_level = Grpc_compression_level
    class Options:
        address: str
        port: int
        reuse_port: bool
        use_ssl: bool
        server_cert: str
        server_key: str
        root_cert: str
        use_mutual_auth: bool
        keepalive_time_ms: int
        keepalive_timeout_ms: int
        keepalive_permit_without_calls: bool
        http2_max_pings_without_data: int
        http2_min_recv_ping_interval_without_data_ms: int
        http2_max_ping_strikes: int
        max_connection_age_ms: int
        max_connection_age_grace_ms: int
        infer_compression_level: int | Grpc_compression_level
        infer_allocation_pool_size: int
        forward_header_pattern: str
        def __post_init__(self) -> None: ...
    triton_frontend: Incomplete
    def __init__(self, server: tritonserver, options: KServeGrpc.Options = None) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
