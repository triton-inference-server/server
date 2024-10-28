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
