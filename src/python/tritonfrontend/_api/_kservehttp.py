from dataclasses import dataclass, asdict
from typing import Any, Union
from enum import Enum
from tritonfrontend._api.validation import Validation
import tritonserver

from tritonfrontend._c.tritonfrontend_bindings import TritonFrontendHttp

class KServeHttp:
    @dataclass
    class Options(Validation):
        address: str = "0.0.0.0"
        port: int = 8000
        reuse_port: bool = False
        thread_count: int = 8
        header_forward_pattern: str = ""
        # restricted_protocols: list

    class Server:
        def __init__(self, server: tritonserver, options: "KServeHTTP.KServeHttpOptions"):

            server_ptr = server.get_c_ptr()
            options_dict: dict[str, Union[int, bool, str]] = asdict(options)
            # Converts dataclass instance -> python dictionary -> unordered_map<string, std::variant<...>> 

            self.triton_c_object = TritonFrontendHttp(server_ptr, options_dict)

        def __del__(self):
            # Delete called on C++ side, so assigning to None to prevent double-free
            self.triton_c_object = None
            pass
            # Need to bind a function which is
        
        def start(self):
            return self.triton_c_object.start()
        
        def stop(self):
            return self.triton_c_object.stop()


