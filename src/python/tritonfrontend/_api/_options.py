from dataclasses import dataclass
from typing import Any, Union
from enum import Enum

class Validation:
    def validate_type(self, value: Any, expected_type: Any, param_name: str):
        if not isinstance(value, expected_type):
            raise TypeError(f"Incorrect Type for {param_name}. Expected {expected_type}, got {type(value)}")
    
    #TODO: implement to catch ints in python that are too big for int32_t in C++
    def validate_range(self, value, lb, ub, param_name):
        pass
    
    def validate(self):
        for param_name, param_type in self.__annotations__.items():
            value = getattr(self, param_name)
            self.validate_type(value, param_type, param_name)

@dataclass
class KServeHttpOptions(Validation):
    address: str = "0.0.0.0"
    port: int = 8000
    reuse_port: bool = False
    thread_count: int = 8
    header_forward_pattern: str = ""
    # restricted_protocols: list

class Grpc_compression_level(Enum):
    NONE  = 0
    LOW   = 1
    MED   = 2
    HIGH  = 3 
    COUNT = 4

# triton::server::grpc::Options
@dataclass
class KServeGrpcOptions(Validation): 
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
    
    #triton::server::grpc::Options
    
    infer_compression_level: Union[int, Grpc_compression_level] = Grpc_compression_level.NONE
    infer_allocation_pool_size: int = 8
    forward_header_pattern: str = ""

    def __post_init__(self):
        if isinstance(self.infer_compression_level, Grpc_compression_level):
            self.infer_compression_level = self.infer_compression_level.value

@dataclass
class MetricsOptions(Validation):
    address: str = ""
    port: int = 8002
    interval_ms: int = 2000
    allow_gpu_metrics: bool = True
    allow_cpu_metrics: bool  = True
    # metrics_config_settings?: [ ("", "", ""), ("", "", ""), ...]

@dataclass
class SageMakerOptions(Validation):
    address: str = "0.0.0.0"
    port: int = 8080
    sage_range_set: bool = False
    sage_range: tuple = (-1, -1)  # Need to be binded to std::pair<int32_t, int32_t>
    thread_count: int = 8

@dataclass
class VertexAIOptions(Validation):
    address: str = "0.0.0.0"
    port: int = 8080
    thread_count: int = 8
    default_model: str = ""



    