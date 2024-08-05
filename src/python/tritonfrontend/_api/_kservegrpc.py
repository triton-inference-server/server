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


from dataclasses import asdict, dataclass
from enum import Enum
from typing import Union

import tritonserver
from tritonfrontend._api.validation import Validation
from tritonfrontend._c.tritonfrontend_bindings import TritonFrontendGrpc


# Enum (mirroring C++ format)
class Grpc_compression_level(Enum):
    NONE = 0
    LOW = 1
    MED = 2
    HIGH = 3
    COUNT = 4


class KServeGrpc:
    Grpc_compression_level = (
        Grpc_compression_level  # Include the enum as a class attribute
    )

    # triton::server::grpc::Options
    @dataclass
    class Options(Validation):
        # triton::server::grpc::SocketOptions
        address: str = "0.0.0.0"
        port: int = 8001
        reuse_port: bool = False
        # triton::server::grpc::SslOptions
        use_ssl: bool = False
        server_cert: str = ""
        server_key: str = ""
        root_cert: str = ""
        use_mutual_auth: bool = False
        # triton::server::grpc::KeepAliveOptions
        keepalive_time_ms: int = 7_200_000
        keepalive_timeout_ms: int = 20_000
        keepalive_permit_without_calls: bool = False
        http2_max_pings_without_data: int = 2
        http2_min_recv_ping_interval_without_data_ms: int = 300_000
        http2_max_ping_strikes: int = 2
        max_connection_age_ms: int = 0
        max_connection_age_grace_ms: int = 0

        # triton::server::grpc::Options

        infer_compression_level: Union[
            int, Grpc_compression_level
        ] = Grpc_compression_level.NONE
        infer_allocation_pool_size: int = 8
        forward_header_pattern: str = ""

        def __post_init__(self):
            if isinstance(self.infer_compression_level, Grpc_compression_level):
                self.infer_compression_level = self.infer_compression_level.value

            self.validate()

    class Server:
        def __init__(self, server: tritonserver, options: "KServeGrpc.Options"):
            server_ptr = server.get_c_ptr()
            options_dict: dict[str, Union[int, bool, str]] = asdict(options)
            # Converts dataclass instance -> python dictionary -> unordered_map<string, std::variant<...>>

            self.triton_c_object = TritonFrontendGrpc(server_ptr, options_dict)

        def __del__(self):
            # Delete called on C++ side, so assigning to None to prevent double-free
            self.triton_c_object = None

        def start(self):
            return self.triton_c_object.start()

        def stop(self):
            return self.triton_c_object.stop()
