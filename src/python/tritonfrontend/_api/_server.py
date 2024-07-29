from typing import Union
from dataclasses import asdict
from tritonfrontend._api._options import KServeHttpOptions, KServeGrpcOptions
from tritonfrontend._api._options import MetricsOptions, SageMakerOptions, VertexAIOptions
from tritonfrontend._c.tritonfrontend_bindings import TritonFrontendCWrapper
import tritonserver


optionsGroup = Union[KServeHttpOptions, KServeGrpcOptions, MetricsOptions, SageMakerOptions, VertexAIOptions]

class Frontend:
    # def __init__(self, triton_core: Union[TritonCore, int], options: KServeHttpOptions):
    def __init__(self, server: tritonserver, options: dict):
        server_ptr = server.get_c_ptr()
        options_dict: dict[str, Union[int, bool, str]] = asdict(options)

        # Converts dataclass instance -> python dictionary -> unordered_map<string, std::variant<...>> 
        print(f"Options: {options_dict}, Type: {type(options_dict)}")
        self.triton_c_object = TritonFrontendCWrapper(server, options)
        self.triton_c_object.CreateWrapper(server_ptr, options_dict)

    def __del__(self):
        print("Need to delete pointers and assign them to nullptr?")
        pass
        # Need to bind a function which is
    
    def start():
        triton_c_object.start()
    
    def stop()
        triton_c_object.stop()

    def createFrontend(self):
        pass
        
        
        
        


    
