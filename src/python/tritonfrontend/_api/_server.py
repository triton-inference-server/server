from typing import Union
from dataclasses import asdict
from tritonfrontend._api._options import KServeHttpOptions, KServeGrpcOptions
from tritonfrontend._api._options import MetricsOptions, SageMakerOptions, VertexAIOptions
from tritonfrontend._c.tritonfrontend_bindings import create
from tritonfrontend._c.tritonfrontend_bindings import parse_options
from tritonfrontend._c.tritonfrontend_bindings import set_options
import tritonserver
optionsGroup = Union[KServeHttpOptions, KServeGrpcOptions, MetricsOptions, SageMakerOptions, VertexAIOptions]

class Server:
    # def __init__(self, triton_core: Union[TritonCore, int], options: KServeHttpOptions):
    def __init():
        pass
    def __del__():
        pass
        # Need to bind a function which is
    def createServer(server: tritonserver, options: dict):
        
        # Converts dataclass instance -> python dictionary -> unordered_map<string, std::variant<...>>
        options_dict = asdict(optionsGroup)
        c_options = parse_options(options_dict)
        
        server_ptr = server.get_c_ptr()
        create(server_ptr, c_options)

    
