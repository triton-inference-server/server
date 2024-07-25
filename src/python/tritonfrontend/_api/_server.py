from _options import KServeHttpOptions, KServeGrpcOptions
from _options import MetricsOptions, SageMakerOptions, VertexAIOptions
from tritonfrontend._c.tritonfrontend_bindings import create
from typing import Union

optionsGroup = Union[KServeHttpOptions, KServeGrpcOptions, MetricsOptions, SageMakerOptions, VertexAIOptions]
class Server:
    # def __init__(self, triton_core: Union[TritonCore, int], options: KServeHttpOptions):
    def __init():
        pass
    def __del__():
        pass
        # Need to bind a function which is
    def createServer(server_ptr: int, options: dict):
        create(server_ptr, asdict(optionsGroup))

    
