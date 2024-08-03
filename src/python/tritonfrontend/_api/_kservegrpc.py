from dataclasses import dataclass, asdict
from typing import Any, Union, Optional
from enum import Enum
from tritonfrontend._api.validation import Validation

from tritonfrontend._c.tritonfrontend_bindings import TritonFrontendGrpc
import tritonserver

# Enum (mirroring C++ format)
class Grpc_compression_level(Enum):
    NONE  = 0
    LOW   = 1
    MED   = 2
    HIGH  = 3 
    COUNT = 4

class KServeGrpc:
    Grpc_compression_level = Grpc_compression_level  # Include the enum as a class attribute

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
        
        infer_compression_level: Union[int, Grpc_compression_level] = Grpc_compression_level.NONE
        infer_allocation_pool_size: int = 8
        forward_header_pattern: str = ""

        def __post_init__(self):
            if isinstance(self.infer_compression_level, Grpc_compression_level):
                self.infer_compression_level = self.infer_compression_level.value
            
            self.validate()

    class Server:
        def __init__(self, server: tritonserver, options: "KServeHTTP.KServeGrpcOptions"):
            server_ptr = server.get_c_ptr()
            options_dict: dict[str, Union[int, bool, str]] = asdict(options)
            # Converts dataclass instance -> python dictionary -> unordered_map<string, std::variant<...>> 

            self.triton_c_object = TritonFrontendGrpc(server_ptr, options_dict)

        def __del__(self):
            # Delete called on C++ side, so assigning to None to prevent double-free
            self.triton_c_object = None
            pass
            # Need to bind a function which is
        
        def start(self):
            return self.triton_c_object.start()
        
        def stop(self):
            return self.triton_c_object.stop()

