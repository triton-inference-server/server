import tritonserver
from _typeshed import Incomplete
from tritonfrontend._api._error_mapping import (
    handle_triton_error as handle_triton_error,
)
from tritonfrontend._c.tritonfrontend_bindings import (
    InvalidArgumentError as InvalidArgumentError,
)
from tritonfrontend._c.tritonfrontend_bindings import (
    TritonFrontendMetrics as TritonFrontendMetrics,
)

class Metrics:
    class Options:
        address: str
        port: int
        thread_count: int
    triton_frontend: Incomplete
    def __init__(self, server: tritonserver, options: Metrics.Options = None) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
