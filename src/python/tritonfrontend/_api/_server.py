# External Imports
from typing import Union
from dataclasses import asdict
import tritonserver

# Internal Imports
from tritonfrontend._api._options import KServeHttpOptions, KServeGrpcOptions
from tritonfrontend._api._options import MetricsOptions, SageMakerOptions, VertexAIOptions
from tritonfrontend._c.tritonfrontend_bindings import TritonFrontend


dataclassGroup = Union[KServeHttpOptions, KServeGrpcOptions, MetricsOptions, SageMakerOptions, VertexAIOptions]

class Frontend:
    # def __init__(self, triton_core: Union[TritonCore, int], options: KServeHttpOptions):
    def __init__(self, server: tritonserver, options: dataclassGroup):
        print("Frontend()")
        server_ptr = server.get_c_ptr()
        options_dict: dict[str, Union[int, bool, str]] = asdict(options)

        # Converts dataclass instance -> python dictionary -> unordered_map<string, std::variant<...>> 
        print(f"Options: {options_dict}, Type: {type(options_dict)}")
        self.triton_c_object = TritonFrontend(server_ptr, options_dict)

    def __del__(self):
        print("Need to delete pointers and assign them to nullptr?")
        pass
        # Need to bind a function which is
    
    def start(self):
        return self.triton_c_object.start()
    
    def stop(self):
        return self.triton_c_object.stop()

    def createFrontend(self):
        pass
        
        
        
        


    
