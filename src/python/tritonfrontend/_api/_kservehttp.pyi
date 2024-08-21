import tritonserver
from _typeshed import Incomplete
from tritonfrontend._c.tritonfrontend_bindings import (
    InvalidArgumentError as InvalidArgumentError,
)
from tritonfrontend._c.tritonfrontend_bindings import (
    TritonFrontendHttp as TritonFrontendHttp,
)

class KServeHttp:
    class Options:
        address: str
        port: int
        reuse_port: bool
        thread_count: int
        header_forward_pattern: str
    class Server:
        triton_frontend: Incomplete
        def __init__(self, server: tritonserver, options: KServeHttp.Options = None) -> None: ...
        def __enter__(self): ...
        def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None: ...
        def start(self) -> None: ...
        def stop(self) -> None: ...
